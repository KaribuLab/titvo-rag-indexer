from unittest.mock import MagicMock, patch

import pytest

from rag_indexer.application.index_repository_use_case import IndexRepositoryUseCase
from rag_indexer.domain.dto.diff_result_dto import DiffResult
from rag_indexer.domain.dto.file_content_dto import FileContent


class TestIndexRepositoryUseCase:
    @pytest.fixture
    def use_case(self):
        repository_provider = MagicMock()
        code_splitter = MagicMock()
        embedding_provider = MagicMock()
        artifact_store = MagicMock()

        return IndexRepositoryUseCase(
            repository_provider=repository_provider,
            code_splitter=code_splitter,
            embedding_provider=embedding_provider,
            artifact_store=artifact_store,
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
                    "https://github.com/owner/repo",
                    "main",
                    "abc123def456"
                )

        assert result.is_delta is False
        assert result.commit_sha == "abc123def456"
        use_case.artifact_store.upload_db.assert_called_once_with(
            "https://github.com/owner/repo",
            "main",
            "abc123def456",
            "/tmp/rag_index.db"
        )

    def test_execute_full_index_idempotency(self, use_case):
        use_case.artifact_store.get_latest_commit_sha.return_value = "abc123def456"

        result = use_case._execute_full(
            "https://github.com/owner/repo",
            "main",
            "abc123def456"
        )

        assert result.is_delta is False
        assert result.chunks_indexed == 0
        assert result.files_processed == 0
        use_case.artifact_store.upload_db.assert_not_called()

    def test_execute_delta_index_no_previous_raises_error(self, use_case):
        """Delta without existing index for branch should raise error."""
        use_case.artifact_store.get_latest_commit_sha.return_value = None

        error_msg = (
            "No previous index found for branch 'feature-branch'. "
            "Run full index first"
        )
        with pytest.raises(ValueError, match=error_msg):
            use_case._execute_delta(
                "https://github.com/owner/repo",
                "feature-branch",
                "def789"
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
                    "https://github.com/owner/repo",
                    "main",
                    "def789"
                )

        assert result.is_delta is True
        mock_vs_instance.delete_by_file_paths.assert_called_once()
        use_case.artifact_store.upload_db.assert_called_once_with(
            "https://github.com/owner/repo",
            "main",
            "def789",
            "/tmp/rag_index_delta.db"
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
                "https://github.com/owner/repo",
                "main",
                "def789"
            )

        assert result.is_delta is True
        assert result.chunks_indexed == 0
        assert result.files_processed == 0

    def test_execute_delta_idempotency(self, use_case):
        use_case.artifact_store.get_latest_commit_sha.return_value = "abc123"

        result = use_case._execute_delta(
            "https://github.com/owner/repo",
            "main",
            "abc123"
        )

        assert result.is_delta is True
        assert result.chunks_indexed == 0
        use_case.artifact_store.download_latest_db.assert_not_called()

    def test_process_files(self, use_case):
        files = [
            FileContent(path="src/a.py", content="code a"),
            FileContent(path="src/b.py", content="code b"),
        ]

        use_case.code_splitter.split.side_effect = [
            ["chunk1", "chunk2"],
            ["chunk3"],
        ]

        mock_vs = MagicMock()
        use_case._process_files(mock_vs, files, "commit123")

        assert use_case.code_splitter.split.call_count == 2
        mock_vs.store.assert_called_once()

    def test_execute_delta_mode_with_branch_and_sha(self, use_case):
        """Execute with both branch and commit_sha → delta mode."""
        use_case.artifact_store.get_latest_commit_sha.return_value = "abc123"
        use_case.artifact_store.download_latest_db.return_value = "/tmp/existing.db"

        use_case.repository_provider.get_changed_files.return_value = DiffResult(
            added=[],
            modified=[],
            deleted=[],
        )

        with patch("shutil.copy"):
            with patch("os.remove"):
                result = use_case.execute(
                    "https://github.com/owner/repo",
                    branch="feature-branch",
                    commit_sha="def789"
                )

        assert result.is_delta is True
        use_case.repository_provider.resolve_branch_sha.assert_not_called()

    def test_execute_full_mode_with_branch_only(self, use_case):
        """Execute with only branch → full mode."""
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
                    "https://github.com/owner/repo",
                    branch="main"
                )

        assert result.is_delta is False
        use_case.repository_provider.resolve_branch_sha.assert_called_once_with(
            "https://github.com/owner/repo",
            "main"
        )

    def test_execute_without_branch_raises_error(self, use_case):
        """Execute without branch should raise ValueError."""
        with pytest.raises(ValueError, match="branch is required"):
            use_case.execute(
                "https://github.com/owner/repo",
                commit_sha="abc123"
            )

    def test_execute_empty_branch_raises_error(self, use_case):
        """Execute with empty branch should raise ValueError."""
        with pytest.raises(ValueError, match="branch is required"):
            use_case.execute(
                "https://github.com/owner/repo",
                branch="",
                commit_sha="abc123"
            )
