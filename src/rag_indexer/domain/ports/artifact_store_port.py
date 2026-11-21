import abc
from typing import Optional


class IArtifactStorePort(abc.ABC):
    @abc.abstractmethod
    def upload_db(
        self,
        repository_url: str,
        branch: str,
        commit_sha: str,
        db_path: str,
    ) -> None:
        """Upload database to S3 at commit path and update latest pointer."""
        pass

    @abc.abstractmethod
    def download_latest_db(
        self,
        repository_url: str,
        branch: str,
    ) -> Optional[str]:
        """Download latest database from S3 to local path."""
        pass

    @abc.abstractmethod
    def get_latest_commit_sha(self, repository_url: str, branch: str) -> Optional[str]:
        """Read commit SHA from latest/meta.json without downloading DB."""
        pass

    @abc.abstractmethod
    def upload_checkpoint(
        self,
        repository_url: str,
        branch: str,
        commit_sha: str,
        db_path: str,
    ) -> str:
        """Upload DB as a checkpoint for an in-progress run. Returns the S3 key."""
        pass

    @abc.abstractmethod
    def download_checkpoint(
        self,
        repository_url: str,
        branch: str,
        commit_sha: str,
        target_path: str,
    ) -> Optional[str]:
        """Download checkpoint DB to target_path. Returns path or None if absent."""
        pass

    @abc.abstractmethod
    def delete_checkpoint(
        self,
        repository_url: str,
        branch: str,
        commit_sha: str,
    ) -> bool:
        """Delete the checkpoint object. Returns True on success."""
        pass

    @abc.abstractmethod
    def upload_source_snapshot(
        self,
        repository_url: str,
        branch: str,
        commit_sha: str,
        repo_dir_path: str,
    ) -> tuple[str, int]:
        """Tar+gzip the local .git and upload as repo.tar.gz. Returns (key, size_mb)."""
        pass

    @abc.abstractmethod
    def download_source_snapshot(
        self,
        repository_url: str,
        branch: str,
        commit_sha: str,
        target_dir: str,
    ) -> Optional[str]:
        """Download repo.tar.gz and extract. Returns path to .git or None."""
        pass

    @abc.abstractmethod
    def delete_source_snapshot(
        self,
        repository_url: str,
        branch: str,
        commit_sha: str,
    ) -> bool:
        """Delete the source snapshot. Returns True on success."""
        pass

    @abc.abstractmethod
    def acquire_lock(
        self,
        repository_url: str,
        branch: str,
        owner: str,
        ttl_minutes: int,
        commit_sha: str,
        aws_batch_job_id: Optional[str] = None,
    ) -> bool:
        """Acquire the branch lock. Returns True if acquired, False if locked."""
        pass

    @abc.abstractmethod
    def release_lock(
        self,
        repository_url: str,
        branch: str,
        owner: str,
    ) -> bool:
        """Release the lock if owned by `owner`. Returns True on success."""
        pass

    @abc.abstractmethod
    def renew_lock(
        self,
        repository_url: str,
        branch: str,
        owner: str,
        etag: str,
        new_expires_at: str,
    ) -> bool:
        """Renew the lock expiry with IfMatch=etag. Returns False if owner changed."""
        pass

    @abc.abstractmethod
    def get_lock(
        self,
        repository_url: str,
        branch: str,
    ) -> Optional[dict]:
        """Read and return the lock info dict, or None if no lock exists."""
        pass

    # ---------- CLI index ({repo}/cli/latest/) ----------

    @abc.abstractmethod
    def upload_cli_latest(
        self,
        repository_url: str,
        db_path: str,
        manifest: dict,
        meta: dict,
    ) -> None:
        """Upload index.db, manifest.json and meta.json under {repo}/cli/latest/."""
        pass

    @abc.abstractmethod
    def download_cli_latest_db(
        self,
        repository_url: str,
        target_path: str,
    ) -> Optional[str]:
        """Download {repo}/cli/latest/index.db to target_path. None if absent."""
        pass

    @abc.abstractmethod
    def get_cli_latest_manifest(self, repository_url: str) -> Optional[dict]:
        """Read {repo}/cli/latest/manifest.json or None if absent."""
        pass

    @abc.abstractmethod
    def get_cli_latest_meta(self, repository_url: str) -> Optional[dict]:
        """Read {repo}/cli/latest/meta.json or None if absent."""
        pass

    @abc.abstractmethod
    def acquire_cli_lock(
        self,
        repository_url: str,
        owner: str,
        ttl_minutes: int,
        batch_id: str = "",
        aws_batch_job_id: Optional[str] = None,
    ) -> bool:
        """Acquire {repo}/cli/lock.json. True if acquired, False if held."""
        pass

    @abc.abstractmethod
    def release_cli_lock(self, repository_url: str, owner: str) -> bool:
        """Release CLI lock if owned by ``owner``."""
        pass

    @abc.abstractmethod
    def renew_cli_lock(
        self,
        repository_url: str,
        owner: str,
        etag: str,
        new_expires_at: str,
    ) -> bool:
        """Renew CLI lock expiry with IfMatch=etag."""
        pass

    @abc.abstractmethod
    def get_cli_lock(self, repository_url: str) -> Optional[dict]:
        """Read CLI lock info dict, or None if no lock exists."""
        pass
