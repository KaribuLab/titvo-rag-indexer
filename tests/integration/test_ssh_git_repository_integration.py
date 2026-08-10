import subprocess

from rag_indexer.infra.adapters.ssh_git_repository_adapter import (
    SshGitRepositoryAdapter,
)


def _git(repository, *args):
    return subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def test_fetch_files_and_diff_from_real_git_repository(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "--quiet")
    _git(source, "config", "user.name", "Test User")
    _git(source, "config", "user.email", "test@example.com")

    (source / "old name.py").write_text("version one")
    (source / "unchanged.py").write_text("unchanged")
    _git(source, "add", ".")
    _git(source, "commit", "--quiet", "-m", "first")
    from_sha = _git(source, "rev-parse", "HEAD")

    _git(source, "mv", "old name.py", "new name.py")
    (source / "new name.py").write_text("version two")
    (source / "added.py").write_text("added")
    _git(source, "add", ".")
    _git(source, "commit", "--quiet", "-m", "second")
    to_sha = _git(source, "rev-parse", "HEAD")

    adapter = SshGitRepositoryAdapter(
        clone_url=source.resolve().as_uri(),
        private_key="unused-for-local-transport",
    )
    try:
        diff = adapter.get_changed_files("ignored", from_sha, to_sha)
        files = adapter.get_files("ignored", to_sha)
    finally:
        key_dir = adapter._key_dir
        repo_dir = adapter._repo_dir
        adapter.close()

    assert diff.added == ["added.py", "new name.py"]
    assert diff.modified == []
    assert diff.deleted == ["old name.py"]
    assert {file.path: file.content for file in files} == {
        "added.py": "added",
        "new name.py": "version two",
        "unchanged.py": "unchanged",
    }
    assert repo_dir is not None and not repo_dir.exists()
    assert not key_dir.exists()
