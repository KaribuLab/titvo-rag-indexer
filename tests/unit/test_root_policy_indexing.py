"""Public indexing regressions: transport policy is not embedding source.

Use real tar/manifest validation with in-memory transport and vector storage;
no cloud, Git subprocess, or model is used. Partial-chunk cases exercise
real local sqlite-vec storage through its public interface.
"""
from copy import deepcopy
import shutil
import hashlib
import io
import json
import tarfile
from unittest.mock import MagicMock

import pytest

from rag_indexer.infra.adapters.sqlite_vec_store_adapter import SqliteVecStoreAdapter

from rag_indexer.application.index_repository_use_case import IndexRepositoryUseCase
from rag_indexer.infra.adapters.cli_snapshot_repository_adapter import CliSnapshotRepositoryAdapter
from rag_indexer.infra.adapters.ssh_git_repository_adapter import SshGitRepositoryAdapter

ROOT = ".titvoignore.yml"
NESTED = "config/.titvoignore.yml"
URL = "https://github.com/acme/repo"
SOURCE = {"a.py": b"print('ordinary')", NESTED: b"ordinary: yaml"}


def manifest_for(entries):
    return {
        "version": 1, "scope": "full", "commit": "a" * 40,
        "branch": "main", "focus_files": ["a.py"],
        "files": [{"path": p, "sha256": hashlib.sha256(v).hexdigest(), "size": len(v)}
                  for p, v in entries.items()],
    }


def snapshot_provider(entries, manifest):
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        for path, data in entries.items():
            member = tarfile.TarInfo(path)
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
    objects = {"scan-manifest.json": json.dumps(manifest).encode(),
               "snapshot.tar.gz": archive.getvalue()}
    s3 = MagicMock()
    s3.get_object.side_effect = lambda **kw: {
        "Body": io.BytesIO(objects[kw["Key"].rsplit("/", 1)[-1]])}
    provider = CliSnapshotRepositoryAdapter(s3, "fake-bucket", "batch")
    # Observes the source-materialization boundary, not transport hashing.
    provider.get_files_for_paths = MagicMock(wraps=provider.get_files_for_paths)
    return provider


class Vectors:
    def __init__(self, paths=()):
        # Both rows and resume markers must receive the same inventory, even
        # when the caller supplies a one-shot generator.
        paths = tuple(paths)
        self.rows = {path: "old:" + path for path in paths}
        self.indexed = set(paths)
        self.deleted = []
        self.inserted = []

    def is_file_indexed(self, path):
        return path in self.indexed

    def indexed_files_set(self):
        return set(self.indexed)

    def stored_file_paths(self, paths):
        return set(paths) & (set(self.rows) | self.indexed)

    def delete_by_file_paths(self, paths):
        self.deleted.extend(paths)
        for path in paths:
            self.rows.pop(path, None)
            self.indexed.discard(path)

    def insert_one(self, *, doc_id, file_path, chunk_text, embedding):
        self.inserted.append(file_path)
        self.rows[file_path] = chunk_text

    def mark_file_indexed(self, path):
        self.indexed.add(path)


def harness(monkeypatch, tmp_path, provider, previous=None):
    import rag_indexer.application.index_repository_use_case as module
    monkeypatch.setattr(module, "_CLI_DB_PATH", str(tmp_path / "cli.db"))
    vectors = Vectors(item["path"] for item in (previous or {}).get("files", []))
    store = MagicMock()
    store.acquire_cli_lock.return_value = True
    store.acquire_lock.return_value = True
    store.get_cli_lock.return_value = {"etag": "fake"}
    store.get_lock.return_value = {"etag": "fake"}
    store.get_cli_latest_manifest.return_value = deepcopy(previous)
    store.download_cli_latest_db.return_value = True
    store.get_latest_commit_sha.return_value = None
    store.download_checkpoint.return_value = None
    store.download_source_snapshot.return_value = None
    splitter = MagicMock()
    splitter.iter_chunks.side_effect = lambda f: iter([f.content])
    embedder = MagicMock()
    embedder.embed_iter.side_effect = lambda chunks: ((c, [0.1]) for c in chunks)
    use_case = IndexRepositoryUseCase(provider, splitter, embedder, store,
                                      job_id="test", db_path=str(tmp_path / "git.db"))
    opened = MagicMock(return_value=vectors)
    monkeypatch.setattr(use_case, "_open_db", opened)
    return use_case, store, vectors, splitter, embedder, opened


@pytest.mark.parametrize("with_policy", [False, True])
def test_cli_source_inventory_and_processing(monkeypatch, tmp_path, with_policy):
    entries = dict(SOURCE)
    if with_policy:
        entries[ROOT] = b"exceptions: []"
    manifest = manifest_for(entries)
    provider = snapshot_provider(entries, manifest)
    uc, store, vectors, split, embed, _ = harness(monkeypatch, tmp_path, provider)
    result = uc.execute_cli(URL, "batch")
    assert provider.get_files_for_paths.call_args.args == (set(SOURCE),)
    assert {call.args[0].path for call in split.iter_chunks.call_args_list} == set(SOURCE)
    assert embed.embed_iter.call_count == 2
    assert vectors.inserted == sorted(SOURCE)
    assert vectors.indexed == set(SOURCE)
    assert result.files_processed == result.chunks_indexed == 2
    assert result.files_skipped_resume == 0
    published = store.upload_cli_latest.call_args.args[2]
    assert published == {**manifest, "files": manifest_for(SOURCE)["files"]}
    # Validation inventory remains complete; source filtering must not mutate it.
    assert provider.scan_manifest == manifest
    store.release_cli_lock.assert_called_once_with(URL, "test")


@pytest.mark.parametrize("old_policy", [False, True])
@pytest.mark.parametrize("incoming_policy", [False, True])
def test_cli_unchanged_sources_retire_only_old_root(monkeypatch, tmp_path, old_policy, incoming_policy):
    entries = {**SOURCE, ROOT: b"exceptions: []"}
    previous = manifest_for(entries if old_policy else SOURCE)
    incoming = entries if incoming_policy else SOURCE
    provider = snapshot_provider(incoming, manifest_for(incoming))
    uc, store, vectors, split, embed, opened = harness(monkeypatch, tmp_path, provider, previous)
    result = uc.execute_cli(URL, "batch")
    assert vectors.rows == {p: "old:" + p for p in SOURCE}
    assert vectors.deleted == ([ROOT] if old_policy else [])
    assert vectors.inserted == []
    split.iter_chunks.assert_not_called()
    embed.embed_iter.assert_not_called()
    assert result.files_processed == result.chunks_indexed == 0
    assert result.files_skipped_resume == 2
    if old_policy:
        assert opened.call_count == 1
        assert store.upload_cli_latest.call_args.args[2]["files"] == manifest_for(SOURCE)["files"]
    else:
        opened.assert_not_called()
        store.upload_cli_latest.assert_not_called()
    store.release_cli_lock.assert_called_once_with(URL, "test")


@pytest.mark.parametrize("corruption", ["hash", "size", "missing", "extra", "duplicate", "unsafe"])
def test_invalid_policy_transport_has_no_index_effects(monkeypatch, tmp_path, corruption, capsys):
    entries = {**SOURCE, ROOT: b"exceptions: []"}
    manifest = manifest_for(entries)
    policy = manifest["files"][-1]
    if corruption == "hash":
        policy["sha256"] = "0" * 64
    elif corruption == "size":
        policy["size"] += 1
    elif corruption == "missing":
        entries.pop(ROOT)
    elif corruption == "extra":
        manifest["files"].pop()
    elif corruption == "duplicate":
        manifest["files"].append(dict(policy))
    else:
        entries["../" + ROOT] = entries.pop(ROOT)
    provider = snapshot_provider(entries, manifest)
    previous = manifest_for({**SOURCE, ROOT: b"old policy"})
    uc, store, vectors, split, embed, opened = harness(monkeypatch, tmp_path, provider, previous)
    before = deepcopy(vectors.__dict__)
    expected_error = {
        "hash": f"Snapshot integrity mismatch: {ROOT}",
        "size": f"Snapshot integrity mismatch: {ROOT}",
        "missing": f"Snapshot does not match scan-manifest files[] (missing=['{ROOT}'], extra=[])",
        "extra": f"Snapshot does not match scan-manifest files[] (missing=[], extra=['{ROOT}'])",
        "duplicate": "scan-manifest.json has duplicate paths in files[]",
        "unsafe": f"Unsafe archive path: ../{ROOT}",
    }[corruption]
    with pytest.raises(ValueError) as error:
        uc.execute_cli(URL, "batch")
    assert str(error.value) == expected_error
    assert vectors.__dict__ == before
    assert store.get_cli_latest_manifest.return_value == previous
    opened.assert_not_called()
    store.download_cli_latest_db.assert_not_called()
    store.upload_cli_latest.assert_not_called()
    provider.get_files_for_paths.assert_not_called()
    split.iter_chunks.assert_not_called()
    embed.embed_iter.assert_not_called()
    # execute_cli's capability probe evaluates scan_manifest before locking.
    # Invalid transport must therefore touch neither lock ownership nor index.
    store.acquire_cli_lock.assert_not_called()
    store.release_cli_lock.assert_not_called()
    assert capsys.readouterr() == ("", "")


def test_git_full_source_boundary(monkeypatch, tmp_path):
    entries = {**SOURCE, ROOT: b"exceptions: []"}
    runner = MagicMock()
    ids = {p: str(i) * 40 for i, p in enumerate(entries, 1)}
    def run(args, cwd=None):
        if args[0] == "ls-remote":
            return ("a" * 40 + "\trefs/heads/main\n").encode()
        if args[0] == "ls-tree":
            return b"".join(f"100644 blob {ids[p]}\t{p}\0".encode() for p in entries)
        if args[0] == "cat-file":
            return next(entries[p] for p in entries if ids[p] == args[2])
        return b""
    runner.run.side_effect = run
    provider = SshGitRepositoryAdapter("git@github.com:acme/repo.git", "fake-key", runner=runner)
    try:
        uc, store, vectors, split, embed, _ = harness(monkeypatch, tmp_path, provider)
        result = uc.execute(URL, branch="main")
        assert {c.args[0].path for c in split.iter_chunks.call_args_list} == set(SOURCE)
        assert set(vectors.inserted) == set(SOURCE)
        assert embed.embed_iter.call_count == 2
        assert result.files_processed == result.chunks_indexed == 2
        store.upload_db.assert_called_once()
        store.release_lock.assert_called_once_with(repository_url=URL, branch="main", owner="test")
    finally:
        provider.close()


@pytest.fixture
def git_source():
    """Real SSH adapter over a deterministic Git protocol double."""
    entries = {**SOURCE, ROOT: b"exceptions: []"}
    runner = MagicMock()
    runner.diff = b""
    ids = {p: str(i) * 40 for i, p in enumerate(entries, 1)}

    def run(args, cwd=None):
        if args[0] == "ls-remote":
            return ("a" * 40 + "\trefs/heads/main\n").encode()
        if args[0] == "ls-tree":
            return b"".join(f"100644 blob {ids[p]}\t{p}\0".encode() for p in entries)
        if args[0] == "cat-file":
            return next(entries[p] for p in entries if ids[p] == args[2])
        if args[0] == "diff":
            return runner.diff
        return b""

    runner.run.side_effect = run
    provider = SshGitRepositoryAdapter("git@github.com:acme/repo.git", "fake-key", runner=runner)
    try:
        yield provider, runner
    finally:
        provider.close()


def seed_git_db(tmp_path, store, *, checkpoint=False):
    """A local transport placeholder: vectors themselves live in Vectors."""
    path = tmp_path / "download.db"
    path.write_bytes(b"fake database transport")
    if checkpoint:
        store.download_checkpoint.return_value = str(path)
    else:
        store.download_latest_db.return_value = str(path)


def assert_git_cleanup(vectors, split, embed, *, legacy_root):
    assert vectors.rows == {p: "old:" + p for p in SOURCE}
    assert vectors.indexed == set(SOURCE)
    assert vectors.deleted == ([ROOT] if legacy_root else [])
    assert vectors.inserted == []
    split.iter_chunks.assert_not_called()
    embed.embed_iter.assert_not_called()


@pytest.mark.parametrize("legacy_root", [False, True])
def test_git_resume_legacy_cleanup(monkeypatch, tmp_path, git_source, legacy_root):
    provider, runner = git_source
    previous = manifest_for({**SOURCE, **({ROOT: b"old policy"} if legacy_root else {})})
    uc, store, vectors, split, embed, _ = harness(monkeypatch, tmp_path, provider, previous)
    seed_git_db(tmp_path, store, checkpoint=True)
    result = uc.execute(URL, branch="main")
    assert_git_cleanup(vectors, split, embed, legacy_root=legacy_root)
    assert result.files_processed == result.chunks_indexed == 0
    assert result.is_delta is False
    store.upload_db.assert_called_once_with(URL, "main", "a" * 40, uc.db_path)
    store.delete_checkpoint.assert_called_once_with(URL, "main", "a" * 40)
    store.release_lock.assert_called_once_with(repository_url=URL, branch="main", owner="test")


@pytest.mark.parametrize("legacy_root", [False, True])
@pytest.mark.parametrize("explicit_commit", [False, True])
def test_git_identical_commit_legacy_cleanup(monkeypatch, tmp_path, git_source, legacy_root, explicit_commit):
    provider, runner = git_source
    previous = manifest_for({**SOURCE, **({ROOT: b"old policy"} if legacy_root else {})})
    uc, store, vectors, split, embed, _ = harness(monkeypatch, tmp_path, provider, previous)
    store.get_latest_commit_sha.return_value = "a" * 40
    seed_git_db(tmp_path, store)
    result = uc.execute(URL, branch="main", commit_sha="a" * 40 if explicit_commit else None)
    assert_git_cleanup(vectors, split, embed, legacy_root=legacy_root)
    assert result.files_processed == result.chunks_indexed == 0
    assert result.files_skipped_resume == 0
    if legacy_root:
        store.upload_db.assert_called_once()
    else:
        store.upload_db.assert_not_called()
    store.release_lock.assert_called_once_with(repository_url=URL, branch="main", owner="test")


@pytest.mark.parametrize("legacy_root", [False, True])
@pytest.mark.parametrize("change", ["unchanged", "root_modified", "source_modified"])
def test_git_delta_legacy_cleanup(monkeypatch, tmp_path, git_source, legacy_root, change):
    import rag_indexer.application.index_repository_use_case as module
    monkeypatch.setattr(module, "_DELTA_DB_PATH", str(tmp_path / "delta.db"))
    provider, runner = git_source
    runner.diff = {"unchanged": b"", "root_modified": f"M\0{ROOT}\0".encode(),
                   "source_modified": b"M\0a.py\0"}[change]
    previous = manifest_for({**SOURCE, **({ROOT: b"old policy"} if legacy_root else {})})
    uc, store, vectors, split, embed, _ = harness(monkeypatch, tmp_path, provider, previous)
    store.get_latest_commit_sha.return_value = "b" * 40
    seed_git_db(tmp_path, store)
    result = uc.execute(URL, branch="main", commit_sha="a" * 40)
    assert ROOT not in vectors.rows
    assert vectors.indexed == set(SOURCE)
    assert vectors.rows[NESTED] == "old:" + NESTED
    expected_deleted = ({ROOT} if legacy_root else set()) | ({"a.py"} if change == "source_modified" else set())
    assert set(vectors.deleted) == expected_deleted
    if change == "source_modified":
        assert vectors.inserted == ["a.py"]
        assert vectors.rows["a.py"] == SOURCE["a.py"].decode()
        assert [c.args[0].path for c in split.iter_chunks.call_args_list] == ["a.py"]
        assert embed.embed_iter.call_count == 1
        assert result.files_processed == result.chunks_indexed == 1
    else:
        assert_git_cleanup(vectors, split, embed, legacy_root=legacy_root)
        assert result.files_processed == result.chunks_indexed == 0
    assert result.is_delta is True
    assert result.files_skipped_resume == 0
    if legacy_root or change == "source_modified":
        store.upload_db.assert_called_once_with(URL, "main", "a" * 40, uc.db_path)
    else:
        store.upload_db.assert_not_called()
    store.release_lock.assert_called_once_with(repository_url=URL, branch="main", owner="test")


@pytest.mark.parametrize("partial_root", [False, True])
@pytest.mark.parametrize("mode", ["identical_full", "identical_delta", "delta", "resume", "cli", "cli_unlisted_present", "cli_unlisted_absent"])
def test_partial_root_chunks_cleanup_public_use_case(
    monkeypatch, tmp_path, git_source, partial_root, mode, capsys,
):
    """Unmarked chunks must not evade cleanup; other partial chunks survive.

    The False cases pin ordinary no-policy/no-op behavior. Storage setup,
    inspection and cleanup all use the actual adapter's public methods.
    """
    import rag_indexer.application.index_repository_use_case as module
    monkeypatch.setattr(module, "_DELTA_DB_PATH", str(tmp_path / "delta.db"))
    provider, _ = git_source
    inventory = {**SOURCE, **({ROOT: b"old policy"} if partial_root else {})}
    if mode.startswith("cli"):
        incoming = {**SOURCE, **({ROOT: b"exceptions: []"} if mode.endswith("present") else {})}
        provider = snapshot_provider(incoming, manifest_for(incoming))
        if mode.startswith("cli_unlisted"):
            # Persisted inventory and markers cannot reveal an interrupted
            # first-index root chunk; only the DB can.
            inventory = dict(SOURCE)
    uc, artifacts, _, split, embed, _ = harness(
        monkeypatch, tmp_path, provider, manifest_for(inventory),
    )
    # Query embeddings are deterministic and local; embedding source is never
    # needed in these unchanged-source runs.
    embed.embed.side_effect = lambda texts: [[0.1] * 1536 for _ in texts]
    deleted = []

    def open_store(path):
        vectors = SqliteVecStoreAdapter(str(path), embed, artifacts, URL)
        delete = vectors.delete_by_file_paths

        def record_delete(paths):
            acquire = artifacts.acquire_cli_lock if mode.startswith("cli") else artifacts.acquire_lock
            release = artifacts.release_cli_lock if mode.startswith("cli") else artifacts.release_lock
            assert acquire.call_count == 1
            release.assert_not_called()
            deleted.extend(paths)
            delete(paths)

        monkeypatch.setattr(vectors, "delete_by_file_paths", record_delete)
        return vectors

    source_path = tmp_path / "stored.db"
    seed = open_store(source_path)
    expected = {p: "old:" + p for p in SOURCE}
    # An unrelated unfinished source chunk must remain unmarked and untouched.
    expected["unrelated.py"] = "unfinished ordinary chunk"
    for i, (path, text) in enumerate(expected.items()):
        seed.insert_one(str(i), path, text, [0.1] * 1536)
        if path in SOURCE:
            seed.mark_file_indexed(path)
    if partial_root:
        seed.insert_one("root-unfinished", ROOT, "unfinished root chunk", [0.1] * 1536)
    assert seed.indexed_files_set() == set(SOURCE)
    assert seed.is_file_indexed(ROOT) is False
    monkeypatch.setattr(uc, "_open_db", open_store)

    def download_latest(*args):
        target = tmp_path / "download.db"
        shutil.copy(source_path, target)
        return str(target)

    def download_checkpoint(*args):
        if mode != "resume":
            return None
        target = args[-1]
        shutil.copy(source_path, target)
        return target

    def download_cli(_url, target):
        shutil.copy(source_path, target)
        return True

    artifacts.download_latest_db.side_effect = download_latest
    artifacts.download_checkpoint.side_effect = download_checkpoint
    artifacts.download_cli_latest_db.side_effect = download_cli
    published = []

    def publish(path):
        vectors = open_store(path)
        published.append((
            {r["file_path"]: r["chunk_text"] for r in vectors.search("local query", k=10)},
            vectors.indexed_files_set(),
        ))

    artifacts.upload_db.side_effect = lambda _url, _branch, _sha, path: publish(path)
    artifacts.upload_cli_latest.side_effect = lambda _url, path, _manifest, _meta: publish(path)
    artifacts.get_latest_commit_sha.return_value = (
        None if mode == "resume" else "b" * 40 if mode == "delta" else "a" * 40
    )
    if mode.startswith("cli"):
        result = uc.execute_cli(URL, "batch")
    else:
        result = uc.execute(URL, branch="main", commit_sha=(
            "a" * 40 if mode in {"identical_delta", "delta"} else None
        ))
    assert result.files_processed == result.chunks_indexed == 0
    assert result.files_skipped_resume == (len(SOURCE) if mode.startswith("cli") else 0)
    assert deleted == ([ROOT] if partial_root else [])
    assert published == ([(expected, set(SOURCE))] if partial_root or mode == "resume" else [])
    split.iter_chunks.assert_not_called()
    embed.embed_iter.assert_not_called()
    if mode.startswith("cli"):
        artifacts.release_cli_lock.assert_called_once_with(URL, "test")
        if partial_root:
            assert artifacts.upload_cli_latest.call_args.args[2]["files"] == manifest_for(SOURCE)["files"]
    else:
        artifacts.release_lock.assert_called_once_with(repository_url=URL, branch="main", owner="test")
    assert capsys.readouterr() == ("", "")
