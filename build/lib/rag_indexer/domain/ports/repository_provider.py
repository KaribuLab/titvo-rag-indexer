import abc

from rag_indexer.domain.dto.diff_result_dto import DiffResult
from rag_indexer.domain.dto.file_content_dto import FileContent


class IRepositoryProvider(abc.ABC):
    @abc.abstractmethod
    def resolve_branch_sha(self, url: str, branch: str) -> str:
        """Resolve a branch name to its HEAD commit SHA."""
        pass

    @abc.abstractmethod
    def get_files(self, url: str, commit_sha: str) -> list[FileContent]:
        """Get all files from a commit."""
        pass

    @abc.abstractmethod
    def get_changed_files(self, url: str, from_sha: str, to_sha: str) -> DiffResult:
        """Get list of changed files between two commits."""
        pass

    @abc.abstractmethod
    def close(self) -> None:
        """Release repository data and credentials held by the provider."""
        pass
