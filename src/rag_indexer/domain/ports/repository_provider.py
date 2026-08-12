import abc
from typing import Optional, Set

from rag_indexer.domain.dto.diff_result_dto import DiffResult
from rag_indexer.domain.dto.file_content_dto import FileContent


class IRepositoryProvider(abc.ABC):
    @abc.abstractmethod
    def resolve_branch_sha(self, url: str, branch: str) -> str:
        """Resolve a branch name to its HEAD commit SHA."""
        pass

    @abc.abstractmethod
    def get_files(
        self,
        url: str,
        commit_sha: str,
        exclude_paths: Optional[Set[str]] = None,
    ) -> list[FileContent]:
        """Get all files from a commit. Paths in exclude_paths are skipped at the
        cat-file-blob level (adapter still runs ls-tree but skips reading content)."""
        pass

    @abc.abstractmethod
    def get_changed_files(self, url: str, from_sha: str, to_sha: str) -> DiffResult:
        """Get list of changed files between two commits."""
        pass

    @abc.abstractmethod
    def restore_from_snapshot(
        self,
        snapshot_path: str,
        commit_sha: str,
    ) -> None:
        """Restore the local repo from a previously uploaded tarball snapshot.

        After this call, get_files() for the same commit_sha should skip git fetch
        and use the restored local objects."""
        pass

    @abc.abstractmethod
    def close(self) -> None:
        """Release repository data and credentials held by the provider."""
        pass
