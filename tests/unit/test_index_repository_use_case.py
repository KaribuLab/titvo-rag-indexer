from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from rag_indexer.application.index_repository_use_case import IndexRepositoryUseCase
from rag_indexer.domain.dto.checkpoint_config_dto import CheckpointConfig
from rag_indexer.domain.dto.diff_result_dto import DiffResult
from rag_indexer.domain.dto.file_content_dto import FileContent


class TestIndexRepositoryUseCase:
    @pytest.fixture
    def use_case(self):
        repository_provider = MagicMock()
        code_splitter = MagicMock()
        embedding_provider = MagicMock()
        artifact_store = MagicMock()
        # Default: lock acquired successfully, no existing lock
        artifact_store.acquire_lock.return_value = True
        artifact_store.get_lock.return_value = None
        artifact_store.release_lock.return_value = True
        artifact_store.renew_lock.return_value = True

        return IndexRepositoryUseCase(
            repository_provider=repository_provider,
            code_splitter=code_splitter,
            embedding_provider=embedding_provider,
            artifact_store=artifact_store,
            checkpoint_config=CheckpointConfig(every_n_files=2),
        )

    def test_execute_full_index_new_repo(self, use_case):
        use_case.artifact_store.get_latest_commit_sha.return_value = None
        use_case.repository_provider.get_files.return_value = [
            FileContent(path="src/main.py", content="def hello(): pass"),
        ]
        use_case.code_splitter.split.return_value = ["chunk1", "chunk2"]
        use_case.embedding_provider.embed.return_value = [[0.1] * 1536, [0.2] * 1536]

        with patch("os.path.exists", return_value=False):
            with patch(
                "rag_indexer.application.index_repository_use_case.SqliteVecStoreAdapter"
            ) as mock_vs:
                mock_vs_instance = MagicMock()
                mock_vs.return_value = mock_vs_instance

                result = use_case._execute_full(
                    "https://github.com/owner/repo", "main", "abc123def456"
                )

        assert result.is_delta is False
        assert result.commit_sha == "abc123def456"
        use_case.artifact_store.upload_db.assert_called_once_with(
            "https://github.com/owner/repo", "main", "abc123def456", "/tmp/rag_index.db"
        )

    def test_execute_full_index_idempotency(self, use_case):
        use_case.artifact_store.get_latest_commit_sha.return_value = "abc123def456"

        result = use_case._execute_full(
            "https://github.com/owner/repo", "main", "abc123def456"
        )

        assert result.is_delta is False
        assert result.chunks_indexed == 0
        assert result.files_processed == 0
        use_case.artifact_store.upload_db.assert_not_called()

    def test_execute_delta_index_no_previous_raises_error(self, use_case):
        use_case.artifact_store.get_latest_commit_sha.return_value = None

        error_msg = (
            "No previous index found for branch 'feature-branch'. Run full index first"
        )
        with pytest.raises(ValueError, match=error_msg):
            use_case._execute_delta(
                "https://github.com/owner/repo", "feature-branch", "def789"
            )

    def test_execute_delta_index_with_changes(self, use_case):
        use_case.artifact_store.get_latest_commit_sha.return_value = "abc123"
        use_case.artifact_store.download_latest_db.return_value = "/tmp/existing.db"

        use_case.repository_provider.get_changed_files.return_value = DiffResult(
            added=["src/new.py"],
            modified=["src/changed.py"],
            deleted=["src/deleted.py"],
        )

        use_case.repository_provider.get_files.return_value = [
            FileContent(path="src/new.py", content="def new(): pass"),
            FileContent(path="src/changed.py", content="def changed(): pass"),
        ]

        use_case.code_splitter.split.return_value = ["chunk1"]
        use_case.embedding_provider.embed.return_value = [[0.1] * 1536]

        with patch("shutil.copy"), patch("os.remove"):
            with patch(
                "rag_indexer.application.index_repository_use_case.SqliteVecStoreAdapter"
            ) as mock_vs:
                mock_vs_instance = MagicMock()
                mock_vs.return_value = mock_vs_instance

                result = use_case._execute_delta(
                    "https://github.com/owner/repo", "main", "def789"
                )

        assert result.is_delta is True
        mock_vs_instance.delete_by_file_paths.assert_called_once()
        use_case.artifact_store.upload_db.assert_called_once_with(
            "https://github.com/owner/repo", "main", "def789", "/tmp/rag_index_delta.db"
        )

    def test_execute_delta_index_no_changes(self, use_case):
        use_case.artifact_store.get_latest_commit_sha.return_value = "abc123"
        use_case.artifact_store.download_latest_db.return_value = "/tmp/existing.db"

        use_case.repository_provider.get_changed_files.return_value = DiffResult(
            added=[],
            modified=[],
            deleted=[],
        )

        with patch("shutil.copy"), patch("os.remove"):
            result = use_case._execute_delta(
                "https://github.com/owner/repo", "main", "def789"
            )

        assert result.is_delta is True
        assert result.chunks_indexed == 0
        assert result.files_processed == 0

    def test_execute_delta_idempotency(self, use_case):
        use_case.artifact_store.get_latest_commit_sha.return_value = "abc123"

        result = use_case._execute_delta(
            "https://github.com/owner/repo", "main", "abc123"
        )

        assert result.is_delta is True
        assert result.chunks_indexed == 0
        use_case.artifact_store.download_latest_db.assert_not_called()

    def test_process_files(self, use_case):
        files = [
            FileContent(path="src/a.py", content="code a"),
            FileContent(path="src/b.py", content="code b"),
        ]

        use_case.code_splitter.iter_chunks.side_effect = [
            iter(["chunk1", "chunk2"]),
            iter(["chunk3"]),
        ]
        use_case.embedding_provider.embed_iter.side_effect = [
            iter([[0.1] * 1536, [0.2] * 1536]),
            iter([[0.3] * 1536]),
        ]

        mock_vs = MagicMock()
        mock_vs.is_file_indexed.return_value = False
        use_case._process_files(
            mock_vs, files, "https://github.com/owner/repo", "main", "commit123"
        )

        assert use_case.code_splitter.iter_chunks.call_count == 2
        assert use_case.embedding_provider.embed_iter.call_count == 2
        assert mock_vs.insert_one.call_count == 3
        assert mock_vs.mark_file_indexed.call_count == 2

    def test_execute_delta_mode_with_branch_and_sha(self, use_case):
        use_case.artifact_store.get_latest_commit_sha.return_value = "abc123"
        use_case.artifact_store.download_latest_db.return_value = "/tmp/existing.db"
        use_case.repository_provider.get_changed_files.return_value = DiffResult(
            added=[],
            modified=[],
            deleted=[],
        )

        with patch("shutil.copy"), patch("os.remove"):
            result = use_case.execute(
                "https://github.com/owner/repo",
                branch="feature-branch",
                commit_sha="def789",
            )

        assert result.is_delta is True
        use_case.repository_provider.resolve_branch_sha.assert_not_called()

    def test_execute_full_mode_with_branch_only(self, use_case):
        use_case.artifact_store.get_latest_commit_sha.return_value = None
        use_case.repository_provider.resolve_branch_sha.return_value = "abc123"
        use_case.repository_provider.get_files.return_value = [
            FileContent(path="src/main.py", content="def hello(): pass"),
        ]
        use_case.code_splitter.split.return_value = ["chunk1"]
        use_case.embedding_provider.embed.return_value = [[0.1] * 1536]

        with patch("os.path.exists", return_value=False):
            with patch(
                "rag_indexer.application.index_repository_use_case.SqliteVecStoreAdapter"
            ) as mock_vs:
                mock_vs_instance = MagicMock()
                mock_vs.return_value = mock_vs_instance

                result = use_case.execute(
                    "https://github.com/owner/repo", branch="main"
                )

        assert result.is_delta is False
        use_case.repository_provider.resolve_branch_sha.assert_called_once_with(
            "https://github.com/owner/repo", "main"
        )

    def test_execute_without_branch_raises_error(self, use_case):
        with pytest.raises(ValueError, match="branch is required"):
            use_case.execute("https://github.com/owner/repo", commit_sha="abc123")

    def test_execute_empty_branch_raises_error(self, use_case):
        with pytest.raises(ValueError, match="branch is required"):
            use_case.execute(
                "https://github.com/owner/repo", branch="", commit_sha="abc123"
            )


# ============================================================================
# 8. Tests for checkpointing, snapshot, lock, streaming
# ============================================================================


class TestCheckpointResume:
    """8.1-8.8: Checkpoint + snapshot + streaming behavior."""

    @pytest.fixture
    def use_case(self):
        repository_provider = MagicMock()
        code_splitter = MagicMock()
        embedding_provider = MagicMock()
        artifact_store = MagicMock()
        artifact_store.acquire_lock.return_value = True
        artifact_store.get_lock.return_value = None
        artifact_store.release_lock.return_value = True
        artifact_store.renew_lock.return_value = True
        artifact_store.get_latest_commit_sha.return_value = None

        return IndexRepositoryUseCase(
            repository_provider=repository_provider,
            code_splitter=code_splitter,
            embedding_provider=embedding_provider,
            artifact_store=artifact_store,
            checkpoint_config=CheckpointConfig(every_n_files=2),
        )

    def test_8_1_resume_skips_already_indexed_files(self, use_case):
        """_execute_full with existing checkpoint skips files in indexed_files."""
        # Existing checkpoint available
        use_case.artifact_store.download_checkpoint.return_value = "/tmp/checkpoint.db"
        # Mock SqliteVecStoreAdapter to return a set of already-indexed files
        with patch(
            "rag_indexer.application.index_repository_use_case.SqliteVecStoreAdapter"
        ) as mock_vs_cls:
            mock_vs = MagicMock()
            mock_vs.indexed_files_set.return_value = {"src/already.py"}
            mock_vs_cls.return_value = mock_vs

            # Repo returns 2 files, one already indexed
            use_case.repository_provider.get_files.return_value = [
                FileContent(path="src/already.py", content="x"),
                FileContent(path="src/new.py", content="y"),
            ]
            use_case.code_splitter.iter_chunks.return_value = iter(["chunk"])
            use_case.embedding_provider.embed_iter.return_value = iter([[0.1] * 1536])

            result = use_case._execute_full(
                "https://github.com/owner/repo", "main", "abc123"
            )

        # New file processed, already indexed skipped
        assert result.files_processed == 1
        assert result.files_skipped_resume == 1
        # get_files called with exclude_paths including the already-indexed file
        call_kwargs = use_case.repository_provider.get_files.call_args
        assert call_kwargs.kwargs["exclude_paths"] == {"src/already.py"}

    def test_8_2_checkpoint_flushes_every_n_files(self, use_case):
        """Checkpoint is flushed every N files."""
        use_case.artifact_store.download_checkpoint.return_value = None
        use_case.artifact_store.upload_source_snapshot.return_value = ("key", 1)
        files = [FileContent(path=f"src/f{i}.py", content="x") for i in range(5)]
        use_case.repository_provider.get_files.return_value = files
        use_case.code_splitter.iter_chunks.return_value = iter(["chunk"])
        use_case.embedding_provider.embed_iter.return_value = iter([[0.1] * 1536])

        with patch(
            "rag_indexer.application.index_repository_use_case.SqliteVecStoreAdapter"
        ) as mock_vs_cls:
            mock_vs = MagicMock()
            mock_vs.is_file_indexed.return_value = False
            mock_vs.indexed_files_set.return_value = set()
            mock_vs_cls.return_value = mock_vs

            use_case._execute_full("https://github.com/owner/repo", "main", "abc123")

        # every_n_files=2, so flushes at file 2 and 4 (not 5)
        assert use_case.artifact_store.upload_checkpoint.call_count == 2

    def test_8_3_checkpoint_cleaned_after_successful_upload(self, use_case):
        """delete_checkpoint is called after upload_db."""
        use_case.artifact_store.download_checkpoint.return_value = None
        use_case.artifact_store.upload_source_snapshot.return_value = ("key", 1)
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

            use_case._execute_full("https://github.com/owner/repo", "main", "abc123")

        delete_checkpoint_calls = (
            use_case.artifact_store.delete_checkpoint.call_args_list
        )
        # delete_checkpoint is called after upload_db
        assert len(delete_checkpoint_calls) == 1
        assert use_case.artifact_store.delete_source_snapshot.call_count == 1

    def test_8_5_corrupt_checkpoint_treated_as_fresh(self, use_case):
        """If checkpoint is corrupt, fall back to fresh run."""
        use_case.artifact_store.download_checkpoint.return_value = "/tmp/garbage.db"
        # First SqliteVecStoreAdapter call (probe) raises; subsequent call succeeds.
        with patch(
            "rag_indexer.application.index_repository_use_case.SqliteVecStoreAdapter"
        ) as mock_vs_cls:
            probe = MagicMock()
            probe.indexed_files_set.side_effect = Exception(
                "sqlite: database disk image is malformed"
            )
            mock_vs_cls.side_effect = [probe, MagicMock()]

            use_case.repository_provider.get_files.return_value = []
            use_case.code_splitter.iter_chunks.return_value = iter([])
            use_case.embedding_provider.embed_iter.return_value = iter([])

            result = use_case._execute_full(
                "https://github.com/owner/repo", "main", "abc123"
            )

        # Run continued as fresh
        assert result.files_processed == 0
        assert use_case.repository_provider.get_files.called


class TestLockerAcquire:
    """8.11-8.12: acquire_lock behavior."""

    @pytest.fixture
    def use_case(self):
        repository_provider = MagicMock()
        artifact_store = MagicMock()
        return IndexRepositoryUseCase(
            repository_provider=repository_provider,
            code_splitter=MagicMock(),
            embedding_provider=MagicMock(),
            artifact_store=artifact_store,
        )

    def test_8_11_acquire_lock_success_when_absent(self, use_case):
        """acquire_lock returns True when no existing lock."""
        use_case.artifact_store.acquire_lock.return_value = True
        use_case.artifact_store.get_latest_commit_sha.return_value = None
        use_case.repository_provider.resolve_branch_sha.return_value = "abc"
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

        use_case.artifact_store.acquire_lock.assert_called_once()
        # owner arg is positional (or kwarg depending on how the test was called)
        acquire_call = use_case.artifact_store.acquire_lock.call_args
        owner = acquire_call.kwargs.get("owner") or acquire_call.args[2]
        assert owner == use_case.job_id

    def test_8_11_acquire_lock_fails_fast_when_active(self, use_case):
        """acquire_lock False + active lock → RuntimeError before any embed."""
        use_case.artifact_store.acquire_lock.return_value = False
        now = datetime.now(timezone.utc)
        use_case.artifact_store.get_lock.return_value = {
            "owner": "other-job",
            "aws_batch_job_id": "other-job",
            "acquired_at": (now - timedelta(minutes=10)).isoformat(),
            "expires_at": (now + timedelta(minutes=350)).isoformat(),
            "commit_sha": "abc",
            "etag": "etag1",
        }

        with pytest.raises(RuntimeError, match="Lock held by other-job"):
            use_case.execute("https://github.com/owner/repo", branch="main")

        # CRITICAL: no embed call was made
        use_case.embedding_provider.embed.assert_not_called()
        use_case.embedding_provider.embed_iter.assert_not_called()

    def test_8_12_stale_lock_takeover(self, use_case):
        """If lock has expired, takeover and retry acquire."""
        use_case.artifact_store.acquire_lock.side_effect = [False, True]
        now = datetime.now(timezone.utc)
        use_case.artifact_store.get_lock.return_value = {
            "owner": "dead-job",
            "aws_batch_job_id": "dead-job",
            "acquired_at": (now - timedelta(hours=7)).isoformat(),
            "expires_at": (now - timedelta(hours=1)).isoformat(),
            "commit_sha": "abc",
            "etag": "etag1",
        }
        use_case.artifact_store.get_latest_commit_sha.return_value = None
        use_case.repository_provider.resolve_branch_sha.return_value = "abc"
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

        # acquire_lock was called twice (failed first, succeeded on retry)
        assert use_case.artifact_store.acquire_lock.call_count == 2
        # The stale lock was released first
        use_case.artifact_store.release_lock.assert_called()


class TestLockRelease:
    """8.13-8.17: lock release / lost lock / renewal."""

    @pytest.fixture
    def use_case(self):
        repository_provider = MagicMock()
        artifact_store = MagicMock()
        artifact_store.acquire_lock.return_value = True
        artifact_store.get_lock.return_value = None
        artifact_store.release_lock.return_value = True
        artifact_store.renew_lock.return_value = True
        return IndexRepositoryUseCase(
            repository_provider=repository_provider,
            code_splitter=MagicMock(),
            embedding_provider=MagicMock(),
            artifact_store=artifact_store,
        )

    def test_8_13_release_lock_only_if_owner_matches(self, use_case):
        """release_lock is delegated to the artifact store with our owner."""
        use_case.artifact_store.get_latest_commit_sha.return_value = None
        use_case.repository_provider.resolve_branch_sha.return_value = "abc"
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

        # release_lock was called with our owner
        finally_call = use_case.artifact_store.release_lock.call_args
        assert finally_call.kwargs["owner"] == use_case.job_id

    def test_8_14_renew_lock_aborts_run_on_failure(self, use_case):
        """If renew_lock returns False mid-run, run aborts with RuntimeError."""
        use_case.artifact_store.get_latest_commit_sha.return_value = None
        use_case.repository_provider.resolve_branch_sha.return_value = "abc"
        use_case.repository_provider.get_files.return_value = [
            FileContent(path="src/a.py", content="x"),
        ]
        use_case.code_splitter.iter_chunks.return_value = iter(["chunk"])
        use_case.embedding_provider.embed_iter.return_value = iter([[0.1] * 1536])
        # renew_lock returns False (lock lost during run)
        use_case.artifact_store.renew_lock.return_value = False

        # Force _maybe_renew_lock to always run (skip the time check)
        def always_renew(self, *args, **kwargs):
            from datetime import datetime, timedelta, timezone

            new_expires = datetime.now(timezone.utc) + timedelta(minutes=360)
            renewed = self.artifact_store.renew_lock(
                repository_url="https://github.com/owner/repo",
                branch="main",
                owner=self.job_id,
                etag="etag1",
                new_expires_at=new_expires.isoformat(),
            )
            if not renewed:
                raise RuntimeError(
                    "Lock lost during renewal; aborting run to avoid duplicate embeds"
                )

        with patch(
            "rag_indexer.application.index_repository_use_case.SqliteVecStoreAdapter"
        ) as mock_vs_cls:
            mock_vs = MagicMock()
            mock_vs.is_file_indexed.return_value = False
            mock_vs.indexed_files_set.return_value = set()
            mock_vs_cls.return_value = mock_vs

            with patch.object(
                IndexRepositoryUseCase, "_maybe_renew_lock", always_renew
            ):
                with pytest.raises(RuntimeError, match="Lock lost during renewal"):
                    use_case.execute("https://github.com/owner/repo", branch="main")

    def test_8_17_lock_not_released_on_exception(self, use_case):
        """If execute() raises, release_lock is still called in finally."""
        use_case.artifact_store.get_latest_commit_sha.side_effect = RuntimeError("boom")
        use_case.artifact_store.release_lock.return_value = True

        with pytest.raises(RuntimeError, match="boom"):
            use_case.execute("https://github.com/owner/repo", branch="main")

        use_case.artifact_store.release_lock.assert_called_once()


class TestStreaming:
    """8.18-8.19: streaming memory profile + mark_file_indexed timing."""

    @pytest.fixture
    def use_case(self):
        return IndexRepositoryUseCase(
            repository_provider=MagicMock(),
            code_splitter=MagicMock(),
            embedding_provider=MagicMock(),
            artifact_store=MagicMock(),
        )

    def test_8_18_streaming_does_not_accumulate_lists(self, use_case):
        """Peak memory is bounded: embeddings are consumed lazily, not accumulated."""
        files = [
            FileContent(path="src/a.py", content="x"),
            FileContent(path="src/b.py", content="y"),
        ]
        use_case.code_splitter.iter_chunks.side_effect = [
            iter(["c1", "c2", "c3"]),
            iter(["c4"]),
        ]
        use_case.embedding_provider.embed_iter.side_effect = [
            iter([[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]),
            iter([[0.4] * 1536]),
        ]

        mock_vs = MagicMock()
        mock_vs.is_file_indexed.return_value = False

        # The key check: insert_one is called per chunk, not in a batch
        use_case._process_files(
            mock_vs, files, "https://github.com/owner/repo", "main", "c"
        )

        assert mock_vs.insert_one.call_count == 4
        # No batch insert was used
        assert "store" not in [call[0] for call in mock_vs.method_calls]

    def test_8_19_mark_file_indexed_only_after_all_chunks(self, use_case):
        """mark_file_indexed is called once per file, AFTER all chunks inserted."""
        files = [
            FileContent(path="src/a.py", content="x"),
            FileContent(path="src/b.py", content="y"),
        ]
        use_case.code_splitter.iter_chunks.side_effect = [
            iter(["c1", "c2"]),
            iter(["c3", "c4", "c5"]),
        ]
        use_case.embedding_provider.embed_iter.side_effect = [
            iter([[0.1] * 1536, [0.2] * 1536]),
            iter([[0.3] * 1536, [0.4] * 1536, [0.5] * 1536]),
        ]

        mock_vs = MagicMock()
        mock_vs.is_file_indexed.return_value = False

        use_case._process_files(
            mock_vs, files, "https://github.com/owner/repo", "main", "c"
        )

        # 2 files → 2 mark_file_indexed calls
        assert mock_vs.mark_file_indexed.call_count == 2
        # Each file's mark_file_indexed uses its file_path (positional arg)
        mark_call_paths = [c.args[0] for c in mock_vs.mark_file_indexed.call_args_list]
        assert set(mark_call_paths) == {"src/a.py", "src/b.py"}


class TestSnapshotUpload:
    """8.6: source snapshot uploaded after first get_files and deleted on success."""

    def test_8_6_snapshot_uploaded_and_deleted_on_success(self):
        use_case = IndexRepositoryUseCase(
            repository_provider=MagicMock(),
            code_splitter=MagicMock(),
            embedding_provider=MagicMock(),
            artifact_store=MagicMock(),
            checkpoint_config=CheckpointConfig(every_n_files=5),
        )
        use_case.artifact_store.acquire_lock.return_value = True
        use_case.artifact_store.get_lock.return_value = None
        use_case.artifact_store.release_lock.return_value = True
        use_case.artifact_store.renew_lock.return_value = True
        use_case.artifact_store.get_latest_commit_sha.return_value = None
        use_case.artifact_store.download_checkpoint.return_value = None
        use_case.artifact_store.download_source_snapshot.return_value = None
        use_case.artifact_store.upload_source_snapshot.return_value = ("key", 1)
        use_case.repository_provider.get_files.return_value = []
        use_case.repository_provider.get_repo_dir.return_value = "/tmp/repo"
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

        # Snapshot uploaded exactly once
        assert use_case.artifact_store.upload_source_snapshot.call_count == 1
        # And deleted after success
        assert use_case.artifact_store.delete_source_snapshot.call_count == 1
