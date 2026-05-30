import json
import logging
import re
import tempfile
from typing import Any, Optional

import botocore.exceptions

from rag_indexer.domain.ports.artifact_store_port import IArtifactStorePort

LOGGER = logging.getLogger(__name__)

_DEFAULT_BUCKET_KEY = "rag_index_bucket"


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
        # Remove protocol
        url = re.sub(r"^https?://", "", repository_url)
        # Remove .git suffix
        url = url.rstrip("/").replace(".git", "")
        return url

    def upload_db(
        self,
        repository_url: str,
        branch: str,
        commit_sha: str,
        db_path: str,
    ) -> None:
        bucket = self._get_bucket_name()
        repo_path = self._build_repo_path(repository_url)

        # Upload to commit-specific path under branch
        commit_key = f"{repo_path}/branches/{branch}/{commit_sha}/index.db"
        LOGGER.info("Uploading DB to s3://%s/%s", bucket, commit_key)
        self.s3_client.upload_file(db_path, bucket, commit_key)

        # Create and upload metadata
        meta = {
            "commit_sha": commit_sha,
            "indexed_at": str(__import__("datetime").datetime.utcnow().isoformat()),
        }
        meta_key = f"{repo_path}/branches/{branch}/{commit_sha}/meta.json"
        self.s3_client.put_object(
            Bucket=bucket,
            Key=meta_key,
            Body=json.dumps(meta),
            ContentType="application/json",
        )

        # Copy to branch latest/
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
            # Create a temp file to download to
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
