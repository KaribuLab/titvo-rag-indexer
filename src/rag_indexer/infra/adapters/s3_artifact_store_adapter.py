import json
import logging
import os
import re
import tarfile
import tempfile
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import botocore.exceptions

from rag_indexer.domain.ports.artifact_store_port import IArtifactStorePort

LOGGER = logging.getLogger(__name__)

_DEFAULT_BUCKET_KEY = "rag_index_bucket"

_DEFAULT_CHECKPOINT_KEY_TEMPLATE = (
    "{repo_host}/{owner}/{repo}/branches/{branch}/checkpoints/{commit_sha}/index.db"
)


class S3ArtifactStoreAdapter(IArtifactStorePort):
    def __init__(
        self,
        s3_client: Any,
        configuration_provider: Any,
    ):
        self.s3_client = s3_client
        self.configuration_provider = configuration_provider
        self._bucket_name: Optional[str] = None

    def _get_bucket_name(self) -> str:
        if self._bucket_name is not None:
            return self._bucket_name

        bucket = self.configuration_provider.get_value(_DEFAULT_BUCKET_KEY)
        if not bucket:
            raise ValueError(f"{_DEFAULT_BUCKET_KEY} not found in configuration")
        self._bucket_name = bucket
        return bucket

    def _build_repo_path(self, repository_url: str) -> str:
        """Convert repository URL to S3 path."""
        url = re.sub(r"^https?://", "", repository_url)
        url = url.rstrip("/").replace(".git", "")
        return url

    def _build_checkpoint_key(
        self,
        repository_url: str,
        branch: str,
        commit_sha: str,
        template: str = _DEFAULT_CHECKPOINT_KEY_TEMPLATE,
    ) -> str:
        repo_path = self._build_repo_path(repository_url)
        return template.format(
            repo_host=repo_path.split("/", 1)[0] if "/" in repo_path else "",
            owner=repo_path.split("/", 2)[1] if repo_path.count("/") >= 1 else "",
            repo=repo_path.split("/", 2)[2] if repo_path.count("/") >= 2 else "",
            branch=branch,
            commit_sha=commit_sha,
        )

    # ---------- final DB upload/download (existing) ----------

    def upload_db(
        self,
        repository_url: str,
        branch: str,
        commit_sha: str,
        db_path: str,
    ) -> None:
        bucket = self._get_bucket_name()
        repo_path = self._build_repo_path(repository_url)

        commit_key = f"{repo_path}/branches/{branch}/{commit_sha}/index.db"
        LOGGER.info("Uploading DB to s3://%s/%s", bucket, commit_key)
        self.s3_client.upload_file(db_path, bucket, commit_key)

        meta = {
            "commit_sha": commit_sha,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }
        meta_key = f"{repo_path}/branches/{branch}/{commit_sha}/meta.json"
        self.s3_client.put_object(
            Bucket=bucket,
            Key=meta_key,
            Body=json.dumps(meta),
            ContentType="application/json",
        )

        latest_db_key = f"{repo_path}/branches/{branch}/latest/index.db"
        latest_meta_key = f"{repo_path}/branches/{branch}/latest/meta.json"

        LOGGER.info(
            "Copying to s3://%s/%s/branches/%s/latest/",
            bucket,
            repo_path,
            branch,
        )
        self.s3_client.copy_object(
            CopySource={"Bucket": bucket, "Key": commit_key},
            Bucket=bucket,
            Key=latest_db_key,
        )
        self.s3_client.copy_object(
            CopySource={"Bucket": bucket, "Key": meta_key},
            Bucket=bucket,
            Key=latest_meta_key,
        )

        LOGGER.info(
            "Successfully uploaded database for %s@%s (branch: %s)",
            repository_url,
            commit_sha[:7],
            branch,
        )

    def download_latest_db(
        self,
        repository_url: str,
        branch: str,
    ) -> Optional[str]:
        bucket = self._get_bucket_name()
        repo_path = self._build_repo_path(repository_url)
        key = f"{repo_path}/branches/{branch}/latest/index.db"

        try:
            temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
            temp_path = temp_file.name
            temp_file.close()

            LOGGER.info("Downloading s3://%s/%s to %s", bucket, key, temp_path)
            self.s3_client.download_file(bucket, key, temp_path)
            return temp_path
        except botocore.exceptions.ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "404" or error_code == "NoSuchKey":
                LOGGER.debug(
                    "Latest DB not found for %s (branch: %s)",
                    repository_url,
                    branch,
                )
                return None
            LOGGER.error("Error downloading DB: %s", e)
            raise

    def get_latest_commit_sha(
        self,
        repository_url: str,
        branch: str,
    ) -> Optional[str]:
        bucket = self._get_bucket_name()
        repo_path = self._build_repo_path(repository_url)
        key = f"{repo_path}/branches/{branch}/latest/meta.json"

        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            body = response["Body"].read()
            meta = json.loads(body)
            return meta.get("commit_sha")
        except botocore.exceptions.ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "404" or error_code == "NoSuchKey":
                LOGGER.debug(
                    "Latest meta.json not found for %s (branch: %s)",
                    repository_url,
                    branch,
                )
                return None
            LOGGER.error("Error reading meta.json: %s", e)
            raise

    # ---------- checkpoint DB ----------

    def upload_checkpoint(
        self,
        repository_url: str,
        branch: str,
        commit_sha: str,
        db_path: str,
    ) -> str:
        bucket = self._get_bucket_name()
        key = self._build_checkpoint_key(repository_url, branch, commit_sha)
        LOGGER.info("Uploading checkpoint to s3://%s/%s", bucket, key)
        self.s3_client.upload_file(db_path, bucket, key)
        return key

    def download_checkpoint(
        self,
        repository_url: str,
        branch: str,
        commit_sha: str,
        target_path: str,
    ) -> Optional[str]:
        bucket = self._get_bucket_name()
        key = self._build_checkpoint_key(repository_url, branch, commit_sha)
        try:
            LOGGER.info(
                "Downloading checkpoint s3://%s/%s to %s", bucket, key, target_path
            )
            self.s3_client.download_file(bucket, key, target_path)
            return target_path
        except botocore.exceptions.ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code in ("404", "NoSuchKey"):
                LOGGER.debug("Checkpoint not found at s3://%s/%s", bucket, key)
                return None
            LOGGER.error("Error downloading checkpoint: %s", e)
            raise

    def delete_checkpoint(
        self,
        repository_url: str,
        branch: str,
        commit_sha: str,
    ) -> bool:
        bucket = self._get_bucket_name()
        key = self._build_checkpoint_key(repository_url, branch, commit_sha)
        try:
            self.s3_client.delete_object(Bucket=bucket, Key=key)
            LOGGER.info("Deleted checkpoint s3://%s/%s", bucket, key)
            return True
        except botocore.exceptions.ClientError as e:
            LOGGER.error("Error deleting checkpoint: %s", e)
            return False

    # ---------- source snapshot (.git tarball) ----------

    def _build_snapshot_key(
        self, repository_url: str, branch: str, commit_sha: str
    ) -> str:
        repo_path = self._build_repo_path(repository_url)
        return f"{repo_path}/branches/{branch}/checkpoints/{commit_sha}/repo.tar.gz"

    def upload_source_snapshot(
        self,
        repository_url: str,
        branch: str,
        commit_sha: str,
        repo_dir_path: str,
    ) -> tuple[str, int]:
        bucket = self._get_bucket_name()
        key = self._build_snapshot_key(repository_url, branch, commit_sha)
        git_dir = os.path.join(repo_dir_path, ".git")
        if not os.path.isdir(git_dir):
            raise ValueError(f"No .git directory found at {git_dir}")

        # tar+gzip into a temp file
        t0 = time.time()
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            with tarfile.open(tmp_path, "w:gz") as tar:
                tar.add(git_dir, arcname=".git")
            size_bytes = os.path.getsize(tmp_path)
            size_mb = round(size_bytes / (1024 * 1024), 2)
            LOGGER.info(
                "Source snapshot tarred: size_mb=%s tar_ms=%.0f",
                size_mb,
                (time.time() - t0) * 1000,
            )
            self.s3_client.upload_file(tmp_path, bucket, key)
            LOGGER.info(
                "Source snapshot uploaded s3://%s/%s size_mb=%s",
                bucket,
                key,
                size_mb,
            )
            return key, size_mb
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def download_source_snapshot(
        self,
        repository_url: str,
        branch: str,
        commit_sha: str,
        target_dir: str,
    ) -> Optional[str]:
        bucket = self._get_bucket_name()
        key = self._build_snapshot_key(repository_url, branch, commit_sha)
        try:
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                self.s3_client.download_file(bucket, key, tmp_path)
                with tarfile.open(tmp_path, "r:gz") as tar:
                    tar.extractall(path=target_dir)
                extracted_git = os.path.join(target_dir, ".git")
                if not os.path.isdir(extracted_git):
                    raise ValueError(
                        "Snapshot tarball did not contain a .git directory"
                    )
                LOGGER.info("Source snapshot extracted to %s", extracted_git)
                return extracted_git
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
        except botocore.exceptions.ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code in ("404", "NoSuchKey"):
                LOGGER.debug("Source snapshot not found at s3://%s/%s", bucket, key)
                return None
            LOGGER.error("Error downloading source snapshot: %s", e)
            raise

    def delete_source_snapshot(
        self,
        repository_url: str,
        branch: str,
        commit_sha: str,
    ) -> bool:
        bucket = self._get_bucket_name()
        key = self._build_snapshot_key(repository_url, branch, commit_sha)
        try:
            self.s3_client.delete_object(Bucket=bucket, Key=key)
            LOGGER.info("Deleted source snapshot s3://%s/%s", bucket, key)
            return True
        except botocore.exceptions.ClientError as e:
            LOGGER.error("Error deleting source snapshot: %s", e)
            return False

    # ---------- lock ----------

    def _build_lock_key(self, repository_url: str, branch: str) -> str:
        repo_path = self._build_repo_path(repository_url)
        return f"{repo_path}/locks/{branch}.json"

    def acquire_lock(
        self,
        repository_url: str,
        branch: str,
        owner: str,
        ttl_minutes: int,
        commit_sha: str,
        aws_batch_job_id: Optional[str] = None,
    ) -> bool:
        bucket = self._get_bucket_name()
        key = self._build_lock_key(repository_url, branch)

        now = datetime.now(timezone.utc)
        expires = now + timedelta(minutes=ttl_minutes)
        body = {
            "owner": owner,
            "aws_batch_job_id": aws_batch_job_id,
            "acquired_at": now.isoformat(),
            "expires_at": expires.isoformat(),
            "commit_sha": commit_sha,
        }

        try:
            self.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=json.dumps(body),
                ContentType="application/json",
                IfNoneMatch="*",
            )
            LOGGER.info(
                "Lock acquired owner=%s bucket=%s key=%s expires_at=%s job_id=%s",
                owner,
                bucket,
                key,
                expires.isoformat(),
                aws_batch_job_id,
            )
            return True
        except botocore.exceptions.ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            status_code = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if error_code == "PreconditionFailed" or status_code == 412:
                LOGGER.info("Lock already exists at s3://%s/%s", bucket, key)
                return False
            LOGGER.error("Error acquiring lock: %s", e)
            raise

    def release_lock(
        self,
        repository_url: str,
        branch: str,
        owner: str,
    ) -> bool:
        bucket = self._get_bucket_name()
        key = self._build_lock_key(repository_url, branch)

        # Read the lock to validate ownership and get the etag
        try:
            head = self.s3_client.head_object(Bucket=bucket, Key=key)
        except botocore.exceptions.ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code in ("404", "NoSuchKey"):
                LOGGER.debug("Lock not found at s3://%s/%s", bucket, key)
                return False
            raise

        etag = head.get("ETag", "")
        body_resp = self.s3_client.get_object(Bucket=bucket, Key=key)
        body = json.loads(body_resp["Body"].read())
        if body.get("owner") != owner:
            LOGGER.warning(
                "Lock release refused: owner mismatch (current=%s, caller=%s)",
                body.get("owner"),
                owner,
            )
            return False

        try:
            self.s3_client.delete_object(Bucket=bucket, Key=key, IfMatch=etag)
            LOGGER.info("Lock released owner=%s key=%s", owner, key)
            return True
        except botocore.exceptions.ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "PreconditionFailed":
                LOGGER.warning("Lock release lost race: etag changed")
                return False
            raise

    def renew_lock(
        self,
        repository_url: str,
        branch: str,
        owner: str,
        etag: str,
        new_expires_at: str,
    ) -> bool:
        bucket = self._get_bucket_name()
        key = self._build_lock_key(repository_url, branch)

        try:
            body_resp = self.s3_client.get_object(Bucket=bucket, Key=key)
            body = json.loads(body_resp["Body"].read())
        except botocore.exceptions.ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code in ("404", "NoSuchKey"):
                LOGGER.warning("Lock gone at s3://%s/%s", bucket, key)
                return False
            raise

        if body.get("owner") != owner:
            LOGGER.error(
                "Lock renewal refused: owner changed (current=%s, caller=%s)",
                body.get("owner"),
                owner,
            )
            return False

        body["expires_at"] = new_expires_at
        try:
            self.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=json.dumps(body),
                ContentType="application/json",
                IfMatch=etag,
            )
            LOGGER.info(
                "Lock renewed owner=%s new_expires_at=%s", owner, new_expires_at
            )
            return True
        except botocore.exceptions.ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "PreconditionFailed":
                LOGGER.error("Lock renewal lost race: etag changed")
                return False
            raise

    def get_lock(
        self,
        repository_url: str,
        branch: str,
    ) -> Optional[dict]:
        bucket = self._get_bucket_name()
        key = self._build_lock_key(repository_url, branch)
        try:
            resp = self.s3_client.get_object(Bucket=bucket, Key=key)
            etag = resp.get("ETag", "")
            body = json.loads(resp["Body"].read())
            return {
                "owner": body.get("owner"),
                "aws_batch_job_id": body.get("aws_batch_job_id"),
                "acquired_at": body.get("acquired_at"),
                "expires_at": body.get("expires_at"),
                "commit_sha": body.get("commit_sha"),
                "etag": etag,
            }
        except botocore.exceptions.ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code in ("404", "NoSuchKey"):
                return None
            raise
