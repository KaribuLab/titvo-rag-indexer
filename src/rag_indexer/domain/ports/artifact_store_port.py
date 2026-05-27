import abc
from typing import Optional


class IArtifactStorePort(abc.ABC):
    @abc.abstractmethod
    def upload_db(self, repository_url: str, commit_sha: str, db_path: str) -> None:
        """Upload database to S3 at commit path and update latest pointer."""
        pass

    @abc.abstractmethod
    def download_latest_db(self, repository_url: str) -> Optional[str]:
        """Download latest database from S3 to local path. Returns local path or None."""
        pass

    @abc.abstractmethod
    def get_latest_commit_sha(self, repository_url: str) -> Optional[str]:
        """Read commit SHA from latest/meta.json without downloading DB."""
        pass
