import logging
import os
import tempfile
from typing import Any, Optional

from rag_indexer.domain.dto.diff_result_dto import DiffResult
from rag_indexer.domain.dto.file_content_dto import FileContent
from rag_indexer.domain.dto.index_result_dto import IndexResultDto
from rag_indexer.domain.ports.artifact_store_port import IArtifactStorePort
from rag_indexer.domain.ports.code_splitter_port import ICodeSplitter
from rag_indexer.domain.ports.embedding_provider import IEmbeddingProvider
from rag_indexer.domain.ports.repository_provider import IRepositoryProvider
from rag_indexer.domain.ports.vector_store_port import IVectorStorePort
from rag_indexer.infra.adapters.sqlite_vec_store_adapter import SqliteVecStoreAdapter

LOGGER = logging.getLogger(__name__)

_DEFAULT_DB_PATH = "/tmp/rag_index.db"


class IndexRepositoryUseCase:
    def __init__(
        self,
        repository_provider: IRepositoryProvider,
        code_splitter: ICodeSplitter,
        embedding_provider: IEmbeddingProvider,
        artifact_store: IArtifactStorePort,
        db_path: str = _DEFAULT_DB_PATH,
    ):
        self.repository_provider = repository_provider
        self.code_splitter = code_splitter
        self.embedding_provider = embedding_provider
        self.artifact_store = artifact_store
        self.db_path = db_path

    def execute(
        self,
        repository_url: str,
        branch: Optional[str] = None,
        commit_sha: Optional[str] = None,
    ) -> IndexResultDto:
        LOGGER.info("Starting indexing for repository: %s", repository_url)

        # Determine mode: delta if commit_sha provided, full if branch provided
        if commit_sha:
            return self._execute_delta(repository_url, commit_sha)
        elif branch:
            commit_sha = self.repository_provider.resolve_branch_sha(repository_url, branch)
            return self._execute_full(repository_url, commit_sha)
        else:
            raise ValueError("Either branch or commit_sha must be provided")

    def _execute_full(self, repository_url: str, commit_sha: str) -> IndexResultDto:
        """Execute full indexing for a commit."""
        LOGGER.info("Executing full index for %s@%s", repository_url, commit_sha[:7])

        # Check idempotency
        existing_sha = self.artifact_store.get_latest_commit_sha(repository_url)
        if existing_sha == commit_sha:
            LOGGER.info("Commit %s already indexed, skipping", commit_sha[:7])
            return IndexResultDto(
                repository_url=repository_url,
                commit_sha=commit_sha,
                is_delta=False,
                chunks_indexed=0,
                files_processed=0,
            )

        # Clean up any existing temp DB
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

        # Create new vector store
        vector_store = self._create_vector_store()

        # Get all files
        files = self.repository_provider.get_files(repository_url, commit_sha)
        LOGGER.info("Found %d files to index", len(files))

        # Process files
        chunks_indexed = self._process_files(vector_store, files)

        # Upload database
        self.artifact_store.upload_db(repository_url, commit_sha, self.db_path)

        LOGGER.info("Full index complete: %d chunks from %d files", chunks_indexed, len(files))
        return IndexResultDto(
            repository_url=repository_url,
            commit_sha=commit_sha,
            is_delta=False,
            chunks_indexed=chunks_indexed,
            files_processed=len(files),
        )

    def _execute_delta(self, repository_url: str, commit_sha: str) -> IndexResultDto:
        """Execute delta indexing from previous commit."""
        LOGGER.info("Executing delta index for %s@%s", repository_url, commit_sha[:7])

        # Get previous commit SHA
        prev_sha = self.artifact_store.get_latest_commit_sha(repository_url)

        if not prev_sha:
            LOGGER.info("No previous index found, falling back to full index")
            return self._execute_full(repository_url, commit_sha)

        # Check idempotency
        if prev_sha == commit_sha:
            LOGGER.info("Commit %s already indexed as latest, skipping", commit_sha[:7])
            return IndexResultDto(
                repository_url=repository_url,
                commit_sha=commit_sha,
                is_delta=True,
                chunks_indexed=0,
                files_processed=0,
            )

        # Download existing database
        temp_db_path = self.artifact_store.download_latest_db(repository_url)
        if not temp_db_path:
            LOGGER.info("Could not download latest DB, falling back to full index")
            return self._execute_full(repository_url, commit_sha)

        # Copy to working location
        work_db_path = "/tmp/rag_index_delta.db"
        import shutil
        shutil.copy(temp_db_path, work_db_path)
        os.remove(temp_db_path)  # Clean up temp download
        self.db_path = work_db_path

        # Create vector store with downloaded DB
        vector_store = self._create_vector_store()

        # Get changed files
        diff = self.repository_provider.get_changed_files(repository_url, prev_sha, commit_sha)

        if not diff.added and not diff.modified and not diff.deleted:
            LOGGER.info("No changes detected between %s..%s", prev_sha[:7], commit_sha[:7])
            os.remove(self.db_path)  # Clean up
            return IndexResultDto(
                repository_url=repository_url,
                commit_sha=commit_sha,
                is_delta=True,
                chunks_indexed=0,
                files_processed=0,
            )

        LOGGER.info("Delta: %d added, %d modified, %d deleted", len(diff.added), len(diff.modified), len(diff.deleted))

        # Delete chunks for modified and deleted files
        files_to_delete = diff.modified + diff.deleted
        if files_to_delete:
            vector_store.delete_by_file_paths(files_to_delete)

        # Get all files from new commit and filter to only added/modified
        all_files = self.repository_provider.get_files(repository_url, commit_sha)
        files_to_fetch = set(diff.added + diff.modified)
        files = [f for f in all_files if f.path in files_to_fetch]

        LOGGER.info("Processing %d files from delta", len(files))

        # Process new/modified files
        chunks_indexed = self._process_files(vector_store, files, commit_sha)

        # Upload updated database
        self.artifact_store.upload_db(repository_url, commit_sha, self.db_path)

        # Clean up
        os.remove(self.db_path)

        LOGGER.info("Delta index complete: %d chunks from %d files", chunks_indexed, len(files))
        return IndexResultDto(
            repository_url=repository_url,
            commit_sha=commit_sha,
            is_delta=True,
            chunks_indexed=chunks_indexed,
            files_processed=len(files),
        )

    def _create_vector_store(self) -> SqliteVecStoreAdapter:
        return SqliteVecStoreAdapter(
            db_path=self.db_path,
            embedding_provider=self.embedding_provider,
            artifact_store=self.artifact_store,
            repository_url="",  # Not used in this context
        )

    def _process_files(
        self,
        vector_store: SqliteVecStoreAdapter,
        files: list[FileContent],
        commit_sha: str = "",
    ) -> int:
        """Process files and store chunks. Returns number of chunks indexed."""
        total_chunks = 0
        documents: list[dict[str, Any]] = []

        for file in files:
            chunks = self.code_splitter.split(file.path, file.content)
            for chunk_text in chunks:
                documents.append({
                    "file_path": file.path,
                    "text": chunk_text,
                })
            total_chunks += len(chunks)

        if documents:
            # Store with empty repository_url as it's encoded in the DB path
            vector_store.store("", commit_sha, documents)

        return total_chunks
        """Parse repository URL to (owner/workspace, repo)."""
        # This is a simplified version - in practice would use the same logic as adapters
        url = url.rstrip("/").replace(".git", "")
        parts = url.replace("https://", "").replace("http://", "").split("/")
        if len(parts) < 3:
            raise ValueError(f"Invalid repository URL: {url}")
        return parts[1], parts[2]
