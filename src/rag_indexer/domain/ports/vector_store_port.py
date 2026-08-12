import abc
from typing import Any, Optional, Set


class IVectorStorePort(abc.ABC):
    @abc.abstractmethod
    def store(self, repository_url: str, commit_sha: str, documents: list[Any]) -> None:
        """Store documents with their embeddings for a specific commit.

        NOTE: Prefer insert_one() for streaming use cases. This method is kept for
        backwards compatibility and bulk loads."""
        pass

    @abc.abstractmethod
    def insert_one(
        self,
        doc_id: str,
        file_path: str,
        chunk_text: str,
        embedding: list[float],
    ) -> None:
        """Insert a single chunk with its embedding. Autocommit per call."""
        pass

    @abc.abstractmethod
    def delete_by_file_paths(self, file_paths: list[str]) -> None:
        """Delete all vectors for the given file paths."""
        pass

    @abc.abstractmethod
    def search(self, query: str, k: int) -> list[Any]:
        """Search for k most similar chunks to the query."""
        pass

    @abc.abstractmethod
    def get_latest_indexed_commit(self, repository_url: str) -> Optional[str]:
        """Get the latest indexed commit SHA for a repository."""
        pass

    @abc.abstractmethod
    def is_file_indexed(self, file_path: str) -> bool:
        """Return True if file_path is already in the indexed_files tracking table."""
        pass

    @abc.abstractmethod
    def mark_file_indexed(self, file_path: str) -> None:
        """Record file_path as fully indexed (insert OR IGNORE)."""
        pass

    @abc.abstractmethod
    def count_indexed_files(self) -> int:
        """Return the number of files currently in the indexed_files tracking table."""
        pass

    @abc.abstractmethod
    def indexed_files_set(self) -> Set[str]:
        """Return the full set of file paths in the indexed_files tracking table."""
        pass
