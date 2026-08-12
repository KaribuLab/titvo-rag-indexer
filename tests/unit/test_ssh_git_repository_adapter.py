import stat
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from rag_indexer.infra.adapters.ssh_git_repository_adapter import (
    SshGitRepositoryAdapter,
    SubprocessGitRunner,
)

FROM_SHA = "a" * 40
TO_SHA = "b" * 40


class FakeGitRunner:
    def __init__(self):
        self.calls: list[tuple[list[str], str | None]] = []
        self.responses: dict[str, bytes] = {}
        self.blobs: dict[str, bytes] = {}

    def run(self, args: list[str], cwd: str | None = None) -> bytes:
        self.calls.append((args, cwd))
        if args[0] == "cat-file":
            return self.blobs[args[2]]
        return self.responses.get(args[0], b"")


@pytest.fixture
def adapter():
    runner = FakeGitRunner()
    repository = SshGitRepositoryAdapter(
        clone_url="git@github.com:acme/service.git",
        private_key="private-key",
        runner=runner,
    )
    yield repository, runner
    repository.close()


def test_subprocess_runner_uses_hardened_non_interactive_environment(tmp_path):
    key_path = tmp_path / "id_ed25519"
    known_hosts_path = tmp_path / "known_hosts"
    runner = SubprocessGitRunner(key_path, known_hosts_path, timeout_seconds=17)
    completed = MagicMock(stdout=b"ok")

    with patch("subprocess.run", return_value=completed) as run:
        assert runner.run(["version"]) == b"ok"

    command = run.call_args.args[0]
    options = run.call_args.kwargs
    assert command == ["git", "version"]
    assert options["shell"] is False
    assert options["timeout"] == 17
    assert options["env"]["GIT_TERMINAL_PROMPT"] == "0"
    assert options["env"]["GIT_CONFIG_NOSYSTEM"] == "1"
    ssh_command = options["env"]["GIT_SSH_COMMAND"]
    assert "BatchMode=yes" in ssh_command
    assert "IdentitiesOnly=yes" in ssh_command
    assert "StrictHostKeyChecking=yes" in ssh_command
    assert str(known_hosts_path) in ssh_command


def test_subprocess_runner_reports_failure_without_command_arguments(tmp_path):
    runner = SubprocessGitRunner(tmp_path / "key", tmp_path / "known_hosts")
    failure = subprocess.CalledProcessError(
        128,
        ["git", "fetch", "secret-argument"],
        stderr=b"Permission denied",
    )

    with patch("subprocess.run", side_effect=failure):
        with pytest.raises(RuntimeError, match="git fetch: Permission denied") as error:
            runner.run(["fetch", "secret-argument"])

    assert "secret-argument" not in str(error.value)


def test_private_key_permissions_and_idempotent_cleanup(adapter):
    repository, _ = adapter
    key_path = repository._key_path
    key_dir = repository._key_dir

    assert stat.S_IMODE(key_path.stat().st_mode) == 0o600
    assert key_path.read_text() == "private-key"

    repository.close()
    repository.close()

    assert not key_path.exists()
    assert not key_dir.exists()


def test_resolve_branch_sha_uses_exact_head_ref(adapter):
    repository, runner = adapter
    runner.responses["ls-remote"] = (
        f"{TO_SHA}\trefs/heads/main\n{FROM_SHA}\trefs/heads/main-old\n".encode()
    )

    assert repository.resolve_branch_sha("ignored", "main") == TO_SHA
    assert runner.calls[0][0] == [
        "ls-remote",
        "--heads",
        "git@github.com:acme/service.git",
        "refs/heads/main",
    ]


def test_resolve_branch_sha_rejects_missing_branch(adapter):
    repository, runner = adapter
    runner.responses["ls-remote"] = b""

    with pytest.raises(ValueError, match="Could not resolve branch missing"):
        repository.resolve_branch_sha("ignored", "missing")


def test_get_files_reads_blobs_without_checkout_and_filters_entries(adapter):
    repository, runner = adapter
    text_oid = "1" * 40
    spaced_oid = "2" * 40
    invalid_oid = "3" * 40
    symlink_oid = "4" * 40
    excluded_oid = "5" * 40
    runner.responses["ls-tree"] = b"\0".join(
        [
            f"100644 blob {text_oid}\tsrc/main.py".encode(),
            f"100644 blob {spaced_oid}\tdocs/file name.md".encode(),
            f"100644 blob {invalid_oid}\tsrc/invalid.txt".encode(),
            f"120000 blob {symlink_oid}\tlink".encode(),
            f"100644 blob {excluded_oid}\timage.png".encode(),
            f"160000 commit {FROM_SHA}\tdependency".encode(),
            b"",
        ]
    )
    runner.blobs = {
        text_oid: b"print('hello')",
        spaced_oid: b"documentation",
        invalid_oid: b"\xff\xfe",
        symlink_oid: b"../outside",
    }

    files = repository.get_files("ignored", TO_SHA)

    assert [(file.path, file.content) for file in files] == [
        ("src/main.py", "print('hello')"),
        ("docs/file name.md", "documentation"),
        ("link", "../outside"),
    ]
    commands = [call[0][0] for call in runner.calls]
    assert "checkout" not in commands
    assert excluded_oid not in runner.blobs


def test_get_changed_files_fetches_both_commits_and_maps_rename(adapter):
    repository, runner = adapter
    runner.responses["diff"] = b"\0".join(
        [
            b"A",
            b"src/new.py",
            b"M",
            b"src/changed.py",
            b"D",
            b"src/deleted.py",
            b"R100",
            b"src/old name.py",
            b"src/new name.py",
            b"A",
            b"image.png",
            b"",
        ]
    )

    result = repository.get_changed_files("ignored", FROM_SHA, TO_SHA)

    assert result.added == ["src/new.py", "src/new name.py"]
    assert result.modified == ["src/changed.py"]
    assert result.deleted == ["src/deleted.py", "src/old name.py"]
    fetches = [call for call in runner.calls if call[0][0] == "fetch"]
    assert [call[0][-1] for call in fetches] == [FROM_SHA, TO_SHA]


def test_target_commit_is_reused_after_delta(adapter):
    repository, runner = adapter
    runner.responses["diff"] = b""
    runner.responses["ls-tree"] = b""

    repository.get_changed_files("ignored", FROM_SHA, TO_SHA)
    repository.get_files("ignored", TO_SHA)

    target_fetches = [
        call for call in runner.calls if call[0][0] == "fetch" and call[0][-1] == TO_SHA
    ]
    init_calls = [call for call in runner.calls if call[0][0] == "init"]
    assert len(target_fetches) == 1
    assert len(init_calls) == 1


@pytest.mark.parametrize("sha", ["", "abc123", "g" * 40])
def test_fetch_rejects_invalid_commit_sha(adapter, sha):
    repository, _ = adapter

    with pytest.raises(ValueError, match="Invalid commit SHA"):
        repository.get_files("ignored", sha)


def test_cleanup_removes_repository_after_git_failure(adapter):
    repository, runner = adapter
    runner.run = MagicMock(side_effect=RuntimeError("git failed"))

    with pytest.raises(RuntimeError, match="git failed"):
        repository.get_files("ignored", TO_SHA)

    repo_dir = repository._repo_dir
    repository.close()
    assert repo_dir is not None
    assert not repo_dir.exists()
    assert not repository._key_dir.exists()


def test_5_4_get_files_with_exclude_paths_skips_cat_file(adapter):
    """If exclude_paths contains a path, git cat-file is not called for it."""
    repository, runner = adapter
    keep_oid = "1" * 40
    skip_oid = "2" * 40
    runner.responses["ls-tree"] = b"\0".join(
        [
            f"100644 blob {keep_oid}\tsrc/keep.py".encode(),
            f"100644 blob {skip_oid}\tsrc/skip.py".encode(),
            b"",
        ]
    )
    runner.blobs = {keep_oid: b"keep-content"}

    files = repository.get_files("ignored", TO_SHA, exclude_paths={"src/skip.py"})

    assert [(f.path, f.content) for f in files] == [("src/keep.py", "keep-content")]
    # The skip file's blob was never read
    assert skip_oid not in runner.blobs


def test_5_5_restore_from_snapshot_uses_extracted_git(adapter):
    """restore_from_snapshot moves .git from snapshot to a new mkdtemp."""
    import tempfile
    from pathlib import Path

    repository, _ = adapter

    # Build a fake snapshot directory containing a .git
    snapshot_root = Path(tempfile.mkdtemp(prefix="rag-snapshot-"))
    fake_git = snapshot_root / ".git"
    fake_git.mkdir()
    (fake_git / "HEAD").write_text("ref: refs/heads/main")

    repository.restore_from_snapshot(str(snapshot_root), TO_SHA)

    # After restore, _repo_dir points to a new dir with .git
    assert repository._repo_dir is not None
    assert (repository._repo_dir / ".git").is_dir()
    head_content = (repository._repo_dir / ".git" / "HEAD").read_text()
    assert head_content == "ref: refs/heads/main"
    # And the commit is marked as already fetched
    assert TO_SHA in repository._fetched_commits

    repository.close()
