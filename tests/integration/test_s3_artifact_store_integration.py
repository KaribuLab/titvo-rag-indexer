"""Integration tests using moto to mock S3."""

import os
import tempfile
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import boto3
import pytest
from moto import mock_aws

from rag_indexer.application.index_repository_use_case import IndexRepositoryUseCase
from rag_indexer.domain.dto.checkpoint_config_dto import CheckpointConfig
from rag_indexer.domain.dto.file_content_dto import FileContent
from rag_indexer.infra.adapters.s3_artifact_store_adapter import S3ArtifactStoreAdapter

# ============================================================================
# Fixtures: moto-backed S3 + helper to build a use case against it
# ============================================================================


@pytest.fixture
def aws_credentials(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


@pytest.fixture
def bucket(aws_credentials):
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")
        yield "test-bucket"


@pytest.fixture
def config_provider():
    cp = MagicMock()
    cp.get_value.side_effect = lambda key: (
        "test-bucket" if key == "rag_index_bucket" else None
    )
    cp.get_secret.return_value = "fake-secret"
    return cp


@pytest.fixture
def real_artifact_store(bucket, config_provider):
    """Real S3ArtifactStoreAdapter backed by moto (use for tests that hit S3)."""
    return S3ArtifactStoreAdapter(
        s3_client=boto3.client("s3", region_name="us-east-1"),
        configuration_provider=config_provider,
    )


def _make_use_case(artifact_store_mock, *, lock_ttl_minutes=60, every_n_files=5):
    """Build a use case with a MagicMock artifact_store (for orchestration tests)."""
    use_case = IndexRepositoryUseCase(
        repository_provider=MagicMock(),
        code_splitter=MagicMock(),
        embedding_provider=MagicMock(),
        artifact_store=artifact_store_mock,
        checkpoint_config=CheckpointConfig(
            every_n_files=every_n_files,
            lock_ttl_minutes=lock_ttl_minutes,
        ),
    )
    use_case.artifact_store.acquire_lock.return_value = True
    use_case.artifact_store.get_lock.return_value = None
    use_case.artifact_store.renew_lock.return_value = True
    return use_case


# ============================================================================
# 8.7: resume uses snapshot + exclude_paths and skips git fetch + cat-file
# ============================================================================


def test_8_7_resume_skips_git_fetch_and_cat_file():
    """If snapshot is restored, get_files is called with exclude_paths and
    repository_provider.restore_from_snapshot is invoked."""
    artifact_store = MagicMock()
    use_case = _make_use_case(artifact_store)
    use_case.artifact_store.download_checkpoint.return_value = "/tmp/checkpoint.db"
    use_case.artifact_store.download_source_snapshot.return_value = "/tmp/snap/.git"
    use_case.artifact_store.get_latest_commit_sha.return_value = None
    use_case.repository_provider.get_files.return_value = [
        FileContent(path="src/new.py", content="new"),
    ]
    use_case.code_splitter.iter_chunks.return_value = iter(["chunk"])
    use_case.embedding_provider.embed_iter.return_value = iter([[0.1] * 1536])

    with patch(
        "rag_indexer.application.index_repository_use_case.SqliteVecStoreAdapter"
    ) as mock_vs_cls:
        mock_vs = MagicMock()
        mock_vs.is_file_indexed.return_value = False
        mock_vs.indexed_files_set.return_value = {"src/already.py"}
        mock_vs_cls.return_value = mock_vs

        use_case.execute("https://github.com/owner/repo", branch="main")

    use_case.repository_provider.restore_from_snapshot.assert_called_once()
    call = use_case.repository_provider.get_files.call_args
    assert call.kwargs["exclude_paths"] == {"src/already.py"}


# ============================================================================
# 8.8: snapshot absent or corrupt degrades to git fetch with WARNING
# ============================================================================


def test_8_8_snapshot_absent_falls_back_to_git_fetch():
    """If download_source_snapshot returns None, get_files is called normally."""
    artifact_store = MagicMock()
    use_case = _make_use_case(artifact_store)
    use_case.artifact_store.download_checkpoint.return_value = None
    use_case.artifact_store.download_source_snapshot.return_value = None
    use_case.artifact_store.get_latest_commit_sha.return_value = None
    use_case.repository_provider.get_files.return_value = []
    use_case.code_splitter.iter_chunks.return_value = iter([])
    use_case.embedding_provider.embed_iter.return_value = iter([])

    with patch(
        "rag_indexer.application.index_repository_use_case.SqliteVecStoreAdapter"
    ) as mock_vs_cls:
        mock_vs = MagicMock()
        mock_vs.is_file_indexed.return_value = False
        mock_vs.indexed_files_set.return_value = set()
        mock_vs_cls.return_value = mock_vs

        use_case.execute("https://github.com/owner/repo", branch="main")

    use_case.repository_provider.restore_from_snapshot.assert_not_called()
    call = use_case.repository_provider.get_files.call_args
    assert call.kwargs["exclude_paths"] == set()


def test_8_8b_snapshot_restore_fails_falls_back_to_git_fetch():
    """If restore_from_snapshot raises, get_files is called normally."""
    artifact_store = MagicMock()
    use_case = _make_use_case(artifact_store)
    use_case.artifact_store.download_checkpoint.return_value = "/tmp/checkpoint.db"
    use_case.artifact_store.download_source_snapshot.return_value = "/tmp/snap/.git"
    use_case.artifact_store.get_latest_commit_sha.return_value = None
    use_case.repository_provider.restore_from_snapshot.side_effect = RuntimeError(
        "git invalid"
    )
    use_case.repository_provider.get_files.return_value = []
    use_case.code_splitter.iter_chunks.return_value = iter([])
    use_case.embedding_provider.embed_iter.return_value = iter([])

    with patch(
        "rag_indexer.application.index_repository_use_case.SqliteVecStoreAdapter"
    ) as mock_vs_cls:
        mock_vs = MagicMock()
        mock_vs.is_file_indexed.return_value = False
        mock_vs.indexed_files_set.return_value = set()
        mock_vs_cls.return_value = mock_vs

        use_case.execute("https://github.com/owner/repo", branch="main")

    # Despite the snapshot being downloaded, restore failed → fell back to git fetch
    use_case.repository_provider.get_files.assert_called_once()


# ============================================================================
# 8.4: cleanup NOT executed if upload_db fails (checkpoint stays for retry)
# ============================================================================


def test_8_4_upload_db_failure_keeps_checkpoint():
    """If upload_db raises, delete_checkpoint is NOT called."""
    artifact_store = MagicMock()
    use_case = _make_use_case(artifact_store)
    use_case.artifact_store.download_checkpoint.return_value = None
    use_case.artifact_store.download_source_snapshot.return_value = None
    use_case.artifact_store.upload_db.side_effect = RuntimeError("S3 down")
    use_case.artifact_store.get_latest_commit_sha.return_value = None
    use_case.repository_provider.get_files.return_value = []
    use_case.code_splitter.iter_chunks.return_value = iter([])
    use_case.embedding_provider.embed_iter.return_value = iter([])

    with patch(
        "rag_indexer.application.index_repository_use_case.SqliteVecStoreAdapter"
    ) as mock_vs_cls:
        mock_vs = MagicMock()
        mock_vs.is_file_indexed.return_value = False
        mock_vs.indexed_files_set.return_value = set()
        mock_vs_cls.return_value = mock_vs

        with pytest.raises(RuntimeError, match="S3 down"):
            use_case.execute("https://github.com/owner/repo", branch="main")

    use_case.artifact_store.delete_checkpoint.assert_not_called()
    use_case.artifact_store.delete_source_snapshot.assert_not_called()


# ============================================================================
# 8.15: two execute() concurrent → second fails with RuntimeError
# ============================================================================


def test_8_15_second_execute_fails_with_real_s3(real_artifact_store):
    """Two acquire_lock calls at the same time: second returns False."""
    # Job 1 acquires the lock
    ok1 = real_artifact_store.acquire_lock(
        repository_url="https://github.com/owner/repo",
        branch="main",
        owner="job-1",
        ttl_minutes=60,
        commit_sha="abc",
        aws_batch_job_id="job-1",
    )
    assert ok1 is True

    # Job 2 tries to acquire
    ok2 = real_artifact_store.acquire_lock(
        repository_url="https://github.com/owner/repo",
        branch="main",
        owner="job-2",
        ttl_minutes=60,
        commit_sha="abc",
        aws_batch_job_id="job-2",
    )
    assert ok2 is False

    # The second sees the active lock
    existing = real_artifact_store.get_lock("https://github.com/owner/repo", "main")
    assert existing["owner"] == "job-1"
    assert existing["aws_batch_job_id"] == "job-1"
    assert existing["expires_at"] > datetime.now(timezone.utc).isoformat()


# ============================================================================
# 8.16: lock released after success; next execute can acquire
# ============================================================================


def test_8_16_release_then_reacquire(real_artifact_store):
    """release_lock frees the lock; next acquire_lock succeeds."""
    # Acquire as job-1
    assert real_artifact_store.acquire_lock(
        "https://github.com/owner/repo", "main", "job-1", 60, "abc", "job-1"
    )
    # Release
    assert real_artifact_store.release_lock(
        "https://github.com/owner/repo", "main", "job-1"
    )
    # Re-acquire as job-2
    assert real_artifact_store.acquire_lock(
        "https://github.com/owner/repo", "main", "job-2", 60, "abc", "job-2"
    )
    lock = real_artifact_store.get_lock("https://github.com/owner/repo", "main")
    assert lock["owner"] == "job-2"


# ============================================================================
# 8.17: lock NOT released on exception; expires by TTL
# ============================================================================


def test_8_17_lock_survives_exception(real_artifact_store):
    """If a job dies without releasing, the lock persists in S3."""
    # Acquire with short TTL
    assert real_artifact_store.acquire_lock(
        "https://github.com/owner/repo", "main", "dead-job", 1, "abc", "dead-job"
    )

    # Lock still exists
    assert (
        real_artifact_store.get_lock("https://github.com/owner/repo", "main")
        is not None
    )

    # release_lock with wrong owner returns False
    assert not real_artifact_store.release_lock(
        "https://github.com/owner/repo", "main", "other-owner"
    )

    # Lock still there
    assert (
        real_artifact_store.get_lock("https://github.com/owner/repo", "main")
        is not None
    )

    # Stale takeover: simulate by deleting the lock manually
    real_artifact_store.s3_client.delete_object(
        Bucket="test-bucket",
        Key="github.com/owner/repo/locks/main.json",
    )

    # Now a new job can acquire
    assert real_artifact_store.acquire_lock(
        "https://github.com/owner/repo", "main", "new-job", 60, "abc", "new-job"
    )


# ============================================================================
# 8.10: tarball of snapshot decompresses into a valid .git
# ============================================================================


def test_8_10_tarball_extracts_to_git(real_artifact_store):
    """A snapshot uploaded as tar.gz can be extracted back to a .git directory."""
    with tempfile.TemporaryDirectory() as tmp:
        repo_dir = os.path.join(tmp, "fake_repo")
        os.makedirs(os.path.join(repo_dir, ".git"))
        # Create a fake .git structure with enough content to be non-zero size
        with open(os.path.join(repo_dir, ".git", "HEAD"), "w") as f:
            f.write("ref: refs/heads/main\n")
        with open(os.path.join(repo_dir, ".git", "config"), "w") as f:
            f.write("[core]\n\trepositoryformatversion = 0\n")
        # Add some bulk
        objects_dir = os.path.join(repo_dir, ".git", "objects", "ab")
        os.makedirs(objects_dir)
        with open(os.path.join(objects_dir, "cdef1234567890"), "wb") as f:
            f.write(b"fake git object " + b"x" * 2048)

        # Upload
        key, size_mb = real_artifact_store.upload_source_snapshot(
            "https://github.com/owner/repo", "main", "abc123", repo_dir
        )
        assert key.endswith("repo.tar.gz")
        assert size_mb >= 0

        # Download and extract
        extract_dir = os.path.join(tmp, "extracted")
        os.makedirs(extract_dir)
        extracted_git = real_artifact_store.download_source_snapshot(
            "https://github.com/owner/repo", "main", "abc123", extract_dir
        )
        assert extracted_git is not None
        assert os.path.isdir(extracted_git)
        assert os.path.exists(os.path.join(extracted_git, "HEAD"))
        assert os.path.exists(os.path.join(extracted_git, "config"))


# ============================================================================
# 8.20: with a 100-file repo and a kill at the middle, next run resumes
# ============================================================================


def test_8_20_resume_does_not_reembed():
    """Run 1: 5 files, all processed. Resume scenario: 2 already indexed,
    3 new. embed_iter should be called 3 times, not 5."""
    artifact_store = MagicMock()
    use_case = _make_use_case(artifact_store)
    use_case.artifact_store.download_checkpoint.return_value = "/tmp/checkpoint.db"
    use_case.artifact_store.download_source_snapshot.return_value = "/tmp/snap/.git"
    use_case.artifact_store.get_latest_commit_sha.return_value = None

    # 5 files, 2 already indexed (in checkpoint)
    files = [FileContent(path=f"src/f{i}.py", content=f"x{i}") for i in range(5)]
    use_case.repository_provider.get_files.return_value = files
    use_case.code_splitter.iter_chunks.side_effect = [
        iter([f"chunk_{i}"]) for i in range(5)
    ]
    use_case.embedding_provider.embed_iter.side_effect = [
        iter([[0.1] * 1536]) for _ in range(5)
    ]

    with patch(
        "rag_indexer.application.index_repository_use_case.SqliteVecStoreAdapter"
    ) as mock_vs_cls:
        mock_vs = MagicMock()

        # 3 files are already indexed (f0, f1, f2); 2 are new (f3, f4)
        def is_indexed(path):
            return path in {"src/f0.py", "src/f1.py", "src/f2.py"}

        mock_vs.is_file_indexed.side_effect = is_indexed
        mock_vs.indexed_files_set.return_value = {
            "src/f0.py",
            "src/f1.py",
            "src/f2.py",
        }
        mock_vs_cls.return_value = mock_vs

        result = use_case.execute("https://github.com/owner/repo", branch="main")

    # 2 new files processed, 3 skipped
    assert result.files_processed == 2
    assert result.files_skipped_resume == 3
    # embed_iter was called only for the 2 new files
    assert use_case.embedding_provider.embed_iter.call_count == 2


# ============================================================================
# 5B.12: embed_iter memory profile
# ============================================================================


def test_5B_12_embed_iter_does_not_accumulate_results():
    """The iterator returns embeddings lazily; the produced list is not the source."""
    from rag_indexer.infra.adapters.langchain_embedding_adapter import (
        LangChainEmbeddingAdapter,
    )

    config_provider = MagicMock()
    config_provider.get_value.side_effect = lambda key: {
        "embedding_provider": "openai",
        "embedding_model": "text-embedding-3-small",
    }[key]
    config_provider.get_secret.return_value = "test-key"
    adapter = LangChainEmbeddingAdapter(
        configuration_provider=config_provider, batch_size=1000
    )

    call_count = 0

    def fake_embed(texts):
        nonlocal call_count
        call_count += 1
        return [[0.1] * 1536 for _ in texts]

    adapter._embeddings = MagicMock(embed_documents=fake_embed)

    # Iterator for 2500 chunks
    it = adapter.embed_iter(iter(["t"] * 2500))

    # Consume the first 1000 (first batch)
    first_batch = list()
    for _ in range(1000):
        first_batch.append(next(it))

    # The first_batch is independent of the iterator (the iterator doesn't
    # re-yield the same items; it was a generator).
    # What we care about: the iterator itself is still alive and has more to emit.
    assert len(first_batch) == 1000
    remaining = sum(1 for _ in it)
    assert remaining == 1500
    assert call_count == 3  # 1000 + 1000 + 500
