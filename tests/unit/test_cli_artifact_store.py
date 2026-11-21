"""Unit tests for CLI latest layout and lock acquire/release."""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock

import boto3
import pytest
from moto import mock_aws

from rag_indexer.infra.adapters.s3_artifact_store_adapter import S3ArtifactStoreAdapter


@pytest.fixture
def aws_credentials(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


@pytest.fixture
def store(aws_credentials):
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="rag-bucket")
        cp = MagicMock()
        cp.get_value.side_effect = lambda key: (
            "rag-bucket" if key == "rag_index_bucket" else None
        )
        yield S3ArtifactStoreAdapter(s3_client=s3, configuration_provider=cp)


REPO = "https://github.com/acme/service"


def test_upload_and_download_cli_latest(store):
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp.write(b"sqlite-bytes")
        db_path = tmp.name
    try:
        manifest = {"version": 1, "files": [{"path": "a.py", "sha256": "a", "size": 1}]}
        meta = {"batch_id": "b1", "indexed_at": "2026-01-01T00:00:00+00:00"}
        store.upload_cli_latest(REPO, db_path, manifest, meta)

        out = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        out_path = out.name
        out.close()
        try:
            assert store.download_cli_latest_db(REPO, out_path) == out_path
            with open(out_path, "rb") as fh:
                assert fh.read() == b"sqlite-bytes"
            assert store.get_cli_latest_manifest(REPO) == manifest
            assert store.get_cli_latest_meta(REPO)["batch_id"] == "b1"
        finally:
            os.remove(out_path)
    finally:
        os.remove(db_path)

    # Keys under {repo}/cli/latest/ and not branches/
    s3 = store.s3_client
    keys = [
        obj["Key"]
        for obj in s3.list_objects_v2(Bucket="rag-bucket")["Contents"]
    ]
    assert "github.com/acme/service/cli/latest/index.db" in keys
    assert "github.com/acme/service/cli/latest/manifest.json" in keys
    assert "github.com/acme/service/cli/latest/meta.json" in keys
    assert all("/branches/" not in k for k in keys)


def test_cli_lock_acquire_release(store):
    assert store.acquire_cli_lock(REPO, owner="job-a", ttl_minutes=10, batch_id="b1")
    lock = store.get_cli_lock(REPO)
    assert lock is not None
    assert lock["owner"] == "job-a"
    assert lock["batch_id"] == "b1"

    # Second acquire fails while held
    assert store.acquire_cli_lock(REPO, owner="job-b", ttl_minutes=10) is False

    assert store.release_cli_lock(REPO, "job-a") is True
    assert store.get_cli_lock(REPO) is None

    # Now job-b can acquire
    assert store.acquire_cli_lock(REPO, owner="job-b", ttl_minutes=10)


def test_cli_lock_release_wrong_owner(store):
    assert store.acquire_cli_lock(REPO, owner="job-a", ttl_minutes=10)
    assert store.release_cli_lock(REPO, "job-b") is False
    assert store.get_cli_lock(REPO)["owner"] == "job-a"


def test_cli_lock_key_is_cli_lock_json(store):
    store.acquire_cli_lock(REPO, owner="job-a", ttl_minutes=5)
    body = store.s3_client.get_object(
        Bucket="rag-bucket",
        Key="github.com/acme/service/cli/lock.json",
    )["Body"].read()
    data = json.loads(body)
    assert data["owner"] == "job-a"
