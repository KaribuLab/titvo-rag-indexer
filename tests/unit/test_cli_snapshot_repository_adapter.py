"""Unit tests for CLI snapshot provider (S3 tar + scan-manifest; no Git)."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from unittest.mock import MagicMock

import botocore.exceptions
import pytest

from rag_indexer.infra.adapters.cli_snapshot_repository_adapter import (
    CliSnapshotRepositoryAdapter,
)


def _client_error(code: str) -> botocore.exceptions.ClientError:
    return botocore.exceptions.ClientError(
        {"Error": {"Code": code, "Message": code}},
        "GetObject",
    )


def _tar_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path, data in files.items():
            info = tarfile.TarInfo(name=path)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _manifest_for(files: dict[str, bytes], **extra) -> dict:
    inventory = [
        {
            "path": path,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
        }
        for path, data in sorted(files.items())
    ]
    body = {
        "version": 1,
        "scope": "full",
        "commit": "abc123",
        "branch": "main",
        "focus_files": [],
        "files": inventory,
    }
    body.update(extra)
    return body


def _s3_with_objects(objects: dict[str, bytes]) -> MagicMock:
    s3 = MagicMock()

    def get_object(*, Bucket, Key):
        if Key not in objects:
            raise _client_error("NoSuchKey")
        body = MagicMock()
        body.read.return_value = objects[Key]
        return {"Body": body}

    s3.get_object.side_effect = get_object
    return s3


@pytest.fixture
def sample_files():
    return {
        "src/a.py": b"print('a')\n",
        "src/b.py": b"print('b')\n",
    }


def test_loads_tar_and_manifest_without_git(sample_files):
    manifest = _manifest_for(sample_files)
    archive = _tar_bytes(sample_files)
    s3 = _s3_with_objects(
        {
            "temp/batch-1/scan-manifest.json": json.dumps(manifest).encode(),
            "temp/batch-1/snapshot.tar.gz": archive,
        }
    )
    provider = CliSnapshotRepositoryAdapter(s3, "cli-bucket", "batch-1")

    files = provider.get_files(url="https://github.com/acme/repo", commit_sha="")

    assert {f.path for f in files} == {"src/a.py", "src/b.py"}
    assert provider.scan_manifest["commit"] == "abc123"
    # Never touch git: no runner / subprocess; only S3 get_object
    assert s3.get_object.call_count == 2
    keys = {c.kwargs["Key"] for c in s3.get_object.call_args_list}
    assert keys == {
        "temp/batch-1/scan-manifest.json",
        "temp/batch-1/snapshot.tar.gz",
    }


def test_git_methods_raise(sample_files):
    manifest = _manifest_for(sample_files)
    s3 = _s3_with_objects(
        {
            "temp/batch-1/scan-manifest.json": json.dumps(manifest).encode(),
            "temp/batch-1/snapshot.tar.gz": _tar_bytes(sample_files),
        }
    )
    provider = CliSnapshotRepositoryAdapter(s3, "cli-bucket", "batch-1")

    with pytest.raises(RuntimeError, match="does not use Git"):
        provider.resolve_branch_sha("https://github.com/acme/repo", "main")
    with pytest.raises(RuntimeError, match="does not use Git"):
        provider.get_changed_files("u", "a", "b")
    with pytest.raises(RuntimeError, match="does not use Git"):
        provider.restore_from_snapshot("/tmp/x", "sha")


def test_missing_manifest_raises(sample_files):
    s3 = _s3_with_objects(
        {"temp/batch-1/snapshot.tar.gz": _tar_bytes(sample_files)}
    )
    provider = CliSnapshotRepositoryAdapter(s3, "cli-bucket", "batch-1")

    with pytest.raises(FileNotFoundError, match="scan-manifest.json"):
        provider.get_files(url="", commit_sha="")


def test_missing_snapshot_raises(sample_files):
    manifest = _manifest_for(sample_files)
    s3 = _s3_with_objects(
        {"temp/batch-1/scan-manifest.json": json.dumps(manifest).encode()}
    )
    provider = CliSnapshotRepositoryAdapter(s3, "cli-bucket", "batch-1")

    with pytest.raises(FileNotFoundError, match="snapshot.tar.gz"):
        provider.get_files(url="", commit_sha="")


def test_integrity_mismatch_raises(sample_files):
    manifest = _manifest_for(sample_files)
    manifest["files"][0]["sha256"] = "0" * 64
    s3 = _s3_with_objects(
        {
            "temp/batch-1/scan-manifest.json": json.dumps(manifest).encode(),
            "temp/batch-1/snapshot.tar.gz": _tar_bytes(sample_files),
        }
    )
    provider = CliSnapshotRepositoryAdapter(s3, "cli-bucket", "batch-1")

    with pytest.raises(ValueError, match="integrity mismatch"):
        provider.get_files(url="", commit_sha="")


def test_get_files_for_paths_filters(sample_files):
    manifest = _manifest_for(sample_files)
    s3 = _s3_with_objects(
        {
            "temp/batch-1/scan-manifest.json": json.dumps(manifest).encode(),
            "temp/batch-1/snapshot.tar.gz": _tar_bytes(sample_files),
        }
    )
    provider = CliSnapshotRepositoryAdapter(s3, "cli-bucket", "batch-1")

    files = provider.get_files_for_paths({"src/a.py"})
    assert len(files) == 1
    assert files[0].path == "src/a.py"
