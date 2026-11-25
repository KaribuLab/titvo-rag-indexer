"""Load CLI scan inventory from S3 temp batch (tar + scan-manifest). No Git/SSH."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import tarfile
from pathlib import PurePosixPath
from typing import Any, Optional, Set

import botocore.exceptions

from rag_indexer.application.cli_manifest_merge import is_root_policy_path
from rag_indexer.domain.dto.diff_result_dto import DiffResult
from rag_indexer.domain.dto.file_content_dto import FileContent
from rag_indexer.domain.ports.repository_provider import IRepositoryProvider

LOGGER = logging.getLogger(__name__)

_INTERNAL_MANIFEST = ".titvo-manifest.json"
_MAX_SNAPSHOT_BYTES = 100 * 1024 * 1024
_SNAPSHOT_NAME = "snapshot.tar.gz"
_SCAN_MANIFEST_NAME = "scan-manifest.json"


class CliSnapshotRepositoryAdapter(IRepositoryProvider):
    """Reads ``temp/{batch_id}/snapshot.tar.gz`` + ``scan-manifest.json`` from S3."""

    def __init__(
        self,
        s3_client: Any,
        bucket: str,
        batch_id: str,
    ):
        if not batch_id:
            raise ValueError("TITVO_SOURCE_BATCH_ID is required for CLI indexing")
        if not bucket:
            raise ValueError("CLI files bucket is required for CLI indexing")
        self.s3_client = s3_client
        self.bucket = bucket
        self.batch_id = batch_id
        self._manifest: Optional[dict[str, Any]] = None
        self._entries: Optional[dict[str, bytes]] = None

    @property
    def scan_manifest(self) -> dict[str, Any]:
        self._ensure_loaded()
        assert self._manifest is not None
        return self._manifest

    def _prefix(self) -> str:
        return f"temp/{self.batch_id}"

    def _snapshot_key(self) -> str:
        return f"{self._prefix()}/{_SNAPSHOT_NAME}"

    def _manifest_key(self) -> str:
        return f"{self._prefix()}/{_SCAN_MANIFEST_NAME}"

    def _get_object_bytes(self, key: str) -> bytes:
        try:
            resp = self.s3_client.get_object(Bucket=self.bucket, Key=key)
            body = resp["Body"]
            try:
                data = body.read(_MAX_SNAPSHOT_BYTES + 1)
            finally:
                body.close()
        except botocore.exceptions.ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "Unknown")
            if error_code in ("404", "NoSuchKey", "NotFound"):
                raise FileNotFoundError(
                    f"Missing CLI batch object s3://{self.bucket}/{key}"
                ) from exc
            raise
        if len(data) > _MAX_SNAPSHOT_BYTES:
            raise ValueError(
                f"CLI object s3://{self.bucket}/{key} exceeds "
                f"{_MAX_SNAPSHOT_BYTES} bytes"
            )
        return data

    def _ensure_loaded(self) -> None:
        if self._manifest is not None and self._entries is not None:
            return

        manifest_raw = self._get_object_bytes(self._manifest_key())
        try:
            manifest = json.loads(manifest_raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid scan-manifest.json for batch {self.batch_id}"
            ) from exc
        if not isinstance(manifest, dict) or "files" not in manifest:
            raise ValueError(
                f"scan-manifest.json for batch {self.batch_id} lacks files[]"
            )

        archive = self._get_object_bytes(self._snapshot_key())
        entries = self._read_tar_entries(archive)
        self._validate_against_manifest(entries, manifest["files"])
        self._manifest = manifest
        self._entries = entries
        LOGGER.info(
            "Loaded CLI batch %s: %d files from s3://%s/%s",
            self.batch_id,
            len(entries),
            self.bucket,
            self._prefix(),
        )

    @staticmethod
    def _read_tar_entries(archive: bytes) -> dict[str, bytes]:
        files: dict[str, bytes] = {}
        total = 0
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
            for member in tar:
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts or "\\" in member.name:
                    raise ValueError(f"Unsafe archive path: {member.name}")
                if member.isdir():
                    continue
                if not member.isfile() or not path.parts:
                    raise ValueError(f"Unsupported archive entry: {member.name}")
                name = path.as_posix()
                if name in files:
                    raise ValueError(f"Duplicate archive path: {name}")
                total += member.size
                if total > _MAX_SNAPSHOT_BYTES:
                    raise ValueError("Snapshot exceeds 100 MiB uncompressed")
                stream = tar.extractfile(member)
                assert stream is not None
                files[name] = stream.read()
        files.pop(_INTERNAL_MANIFEST, None)
        return files

    @staticmethod
    def _validate_against_manifest(
        entries: dict[str, bytes], files_list: list[dict[str, Any]]
    ) -> None:
        expected = {item["path"]: item for item in files_list}
        if len(expected) != len(files_list):
            raise ValueError("scan-manifest.json has duplicate paths in files[]")
        if set(expected) != set(entries):
            missing = set(expected) - set(entries)
            extra = set(entries) - set(expected)
            raise ValueError(
                f"Snapshot does not match scan-manifest files[] "
                f"(missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]})"
            )
        for path, data in entries.items():
            item = expected[path]
            sha = hashlib.sha256(data).hexdigest()
            if sha != item.get("sha256") or len(data) != item.get("size"):
                raise ValueError(f"Snapshot integrity mismatch: {path}")

    def inventory_by_path(self) -> dict[str, str]:
        """Return ``{path: sha256}`` from the loaded scan-manifest."""
        manifest = self.scan_manifest
        return {item["path"]: item["sha256"] for item in manifest["files"]}

    def get_files(
        self,
        url: str,
        commit_sha: str,
        exclude_paths: Optional[Set[str]] = None,
    ) -> list[FileContent]:
        self._ensure_loaded()
        assert self._entries is not None
        exclude = exclude_paths or set()
        result: list[FileContent] = []
        for path in sorted(self._entries):
            if path in exclude or is_root_policy_path(path):
                continue
            data = self._entries[path]
            if b"\x00" in data:
                LOGGER.warning("Skipping CLI file %r: binary content (NUL byte)", path)
                continue
            try:
                content = data.decode("utf-8")
            except UnicodeDecodeError:
                LOGGER.warning(
                    "Skipping CLI file %r: content is not valid UTF-8", path
                )
                continue
            result.append(FileContent(path=path, content=content))
        return result

    def get_files_for_paths(self, paths: Set[str]) -> list[FileContent]:
        """Return UTF-8 file contents for the given paths only."""
        if not paths:
            return []
        all_files = self.get_files(url="", commit_sha="")
        return [f for f in all_files if f.path in paths]

    def resolve_branch_sha(self, url: str, branch: str) -> str:
        raise RuntimeError(
            "CLI indexing does not use Git; resolve_branch_sha is disabled"
        )

    def get_changed_files(self, url: str, from_sha: str, to_sha: str) -> DiffResult:
        raise RuntimeError(
            "CLI indexing does not use Git; get_changed_files is disabled"
        )

    def restore_from_snapshot(self, snapshot_path: str, commit_sha: str) -> None:
        raise RuntimeError(
            "CLI indexing does not use Git; restore_from_snapshot is disabled"
        )

    def close(self) -> None:
        self._manifest = None
        self._entries = None
