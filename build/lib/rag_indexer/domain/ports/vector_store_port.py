import abc
from typing import Any, Optional


class IVectorStorePort(abc.ABC):
    @abc.abstractmethod
    def store(self, repository_url: str, commit_sha: str, documents: list[Any]) -> None:
        """Store documents with their embeddings for a specific commit."""
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
