"""Unit tests for CLI execute_cli: lock wait, merge skip/reembed/drop."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from rag_indexer.application.index_repository_use_case import IndexRepositoryUseCase
from rag_indexer.domain.dto.checkpoint_config_dto import CheckpointConfig
from rag_indexer.domain.dto.file_content_dto import FileContent


def _manifest(*pairs: tuple[str, str], commit: str = "abc"):
    return {
        "version": 1,
        "scope": "full",
        "commit": commit,
        "branch": "main",
        "focus_files": [],
        "files": [
            {"path": path, "sha256": sha, "size": 1} for path, sha in pairs
        ],
    }


def _cli_provider() -> MagicMock:
    provider = MagicMock()
    provider.scan_manifest = {}
    provider.get_files_for_paths = MagicMock(return_value=[])
    return provider


@pytest.fixture
def cli_use_case():
    provider = _cli_provider()
    artifact_store = MagicMock()
    artifact_store.acquire_cli_lock.return_value = True
    artifact_store.get_cli_lock.return_value = {"etag": '"etag1"'}
    artifact_store.release_cli_lock.return_value = True
    artifact_store.renew_cli_lock.return_value = True

    code_splitter = MagicMock()
    code_splitter.iter_chunks.side_effect = lambda f: iter([f"chunk:{f.path}"])
    embedding_provider = MagicMock()
    embedding_provider.embed_iter.side_effect = lambda it: (
        (text, [0.1] * 1536) for text in it
    )

    return IndexRepositoryUseCase(
        repository_provider=provider,
        code_splitter=code_splitter,
        embedding_provider=embedding_provider,
        artifact_store=artifact_store,
        checkpoint_config=CheckpointConfig(lock_ttl_minutes=60),
        job_id="job-cli-1",
    )


def test_execute_cli_acquires_and_releases_lock(cli_use_case):
    manifest = _manifest(("a.py", "aaa"))
    cli_use_case.repository_provider.scan_manifest = manifest
    cli_use_case.artifact_store.get_cli_latest_manifest.return_value = None
    cli_use_case.repository_provider.get_files_for_paths.return_value = [
        FileContent(path="a.py", content="a"),
    ]

    with (
        patch("os.path.exists", return_value=False),
        patch("os.remove"),
        patch(
            "rag_indexer.application.index_repository_use_case.SqliteVecStoreAdapter"
        ) as mock_vs_cls,
    ):
        mock_vs = MagicMock()
        mock_vs.is_file_indexed.return_value = False
        mock_vs_cls.return_value = mock_vs
        result = cli_use_case.execute_cli("https://github.com/acme/repo", "batch-1")

    assert result.files_processed == 1
    cli_use_case.artifact_store.acquire_cli_lock.assert_called()
    cli_use_case.artifact_store.release_cli_lock.assert_called_once_with(
        "https://github.com/acme/repo", "job-cli-1"
    )
    cli_use_case.artifact_store.upload_cli_latest.assert_called_once()


def test_execute_cli_noop_when_inventory_identical(cli_use_case):
    manifest = _manifest(("a.py", "aaa"), ("b.py", "bbb"))
    cli_use_case.repository_provider.scan_manifest = manifest
    cli_use_case.artifact_store.get_cli_latest_manifest.return_value = manifest

    result = cli_use_case.execute_cli("https://github.com/acme/repo", "batch-2")

    assert result.chunks_indexed == 0
    assert result.files_processed == 0
    assert result.files_skipped_resume == 2
    cli_use_case.artifact_store.upload_cli_latest.assert_not_called()
    cli_use_case.repository_provider.get_files_for_paths.assert_not_called()
    cli_use_case.artifact_store.release_cli_lock.assert_called_once()


def test_execute_cli_reembeds_changed_and_skips_unchanged(cli_use_case):
    prev = _manifest(("a.py", "aaa"), ("b.py", "bbb"))
    new = _manifest(("a.py", "aaa"), ("b.py", "BBB"))
    cli_use_case.repository_provider.scan_manifest = new
    cli_use_case.artifact_store.get_cli_latest_manifest.return_value = prev
    cli_use_case.artifact_store.download_cli_latest_db.return_value = (
        "/tmp/rag_index_cli.db"
    )
    cli_use_case.repository_provider.get_files_for_paths.return_value = [
        FileContent(path="b.py", content="b2"),
    ]

    with (
        patch("os.path.exists", return_value=False),
        patch("os.remove"),
        patch(
            "rag_indexer.application.index_repository_use_case.SqliteVecStoreAdapter"
        ) as mock_vs_cls,
    ):
        mock_vs = MagicMock()
        mock_vs.is_file_indexed.return_value = False
        mock_vs_cls.return_value = mock_vs
        result = cli_use_case.execute_cli("https://github.com/acme/repo", "batch-3")

    assert result.files_processed == 1
    assert result.files_skipped_resume == 1
    cli_use_case.repository_provider.get_files_for_paths.assert_called_once_with(
        {"b.py"}
    )
    mock_vs.delete_by_file_paths.assert_called_once_with(["b.py"])


def test_execute_cli_drops_absent_paths(cli_use_case):
    prev = _manifest(("a.py", "aaa"), ("gone.py", "ggg"))
    new = _manifest(("a.py", "aaa"))
    cli_use_case.repository_provider.scan_manifest = new
    cli_use_case.artifact_store.get_cli_latest_manifest.return_value = prev
    cli_use_case.artifact_store.download_cli_latest_db.return_value = (
        "/tmp/rag_index_cli.db"
    )
    cli_use_case.repository_provider.get_files_for_paths.return_value = []

    with (
        patch("os.path.exists", return_value=False),
        patch("os.remove"),
        patch(
            "rag_indexer.application.index_repository_use_case.SqliteVecStoreAdapter"
        ) as mock_vs_cls,
    ):
        mock_vs = MagicMock()
        mock_vs_cls.return_value = mock_vs
        result = cli_use_case.execute_cli("https://github.com/acme/repo", "batch-4")

    assert result.files_processed == 0
    mock_vs.delete_by_file_paths.assert_called_once_with(["gone.py"])
    cli_use_case.artifact_store.upload_cli_latest.assert_called_once()


def test_cli_lock_wait_then_acquire(cli_use_case):
    cli_use_case.artifact_store.acquire_cli_lock.side_effect = [False, True]
    expires = (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()
    cli_use_case.artifact_store.get_cli_lock.side_effect = [
        {"owner": "other", "expires_at": expires, "etag": '"e1"'},
        {"etag": '"e2"'},
    ]
    manifest = _manifest(("a.py", "aaa"))
    cli_use_case.repository_provider.scan_manifest = manifest
    cli_use_case.artifact_store.get_cli_latest_manifest.return_value = manifest

    with patch("time.sleep") as sleep:
        result = cli_use_case.execute_cli(
            "https://github.com/acme/repo",
            "batch-5",
            lock_wait_seconds=30,
            lock_poll_seconds=1,
        )

    assert result.chunks_indexed == 0
    sleep.assert_called()
    assert cli_use_case.artifact_store.acquire_cli_lock.call_count == 2


def test_cli_lock_stale_takeover(cli_use_case):
    cli_use_case.artifact_store.acquire_cli_lock.side_effect = [False, True]
    stale = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    cli_use_case.artifact_store.get_cli_lock.side_effect = [
        {"owner": "dead-job", "expires_at": stale, "etag": '"e1"'},
        {"etag": '"e2"'},
    ]
    manifest = _manifest(("a.py", "aaa"))
    cli_use_case.repository_provider.scan_manifest = manifest
    cli_use_case.artifact_store.get_cli_latest_manifest.return_value = manifest

    with patch("time.sleep") as sleep:
        cli_use_case.execute_cli("https://github.com/acme/repo", "batch-6")

    cli_use_case.artifact_store.release_cli_lock.assert_any_call(
        "https://github.com/acme/repo", "dead-job"
    )
    sleep.assert_not_called()


def test_cli_lock_wait_timeout(cli_use_case):
    cli_use_case.artifact_store.acquire_cli_lock.return_value = False
    expires = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    cli_use_case.artifact_store.get_cli_lock.return_value = {
        "owner": "other",
        "expires_at": expires,
        "etag": '"e1"',
    }

    with (
        patch("time.sleep"),
        patch("time.monotonic", side_effect=[0.0, 0.0, 301.0]),
        pytest.raises(RuntimeError, match="timed out"),
    ):
        cli_use_case.execute_cli(
            "https://github.com/acme/repo",
            "batch-7",
            lock_wait_seconds=300,
            lock_poll_seconds=5,
        )

    cli_use_case.artifact_store.release_cli_lock.assert_not_called()


def test_execute_cli_rejects_non_cli_provider():
    provider = MagicMock(spec=["get_files", "close"])
    use_case = IndexRepositoryUseCase(
        repository_provider=provider,
        code_splitter=MagicMock(),
        embedding_provider=MagicMock(),
        artifact_store=MagicMock(),
    )
    with pytest.raises(TypeError, match="CLI snapshot repository provider"):
        use_case.execute_cli("https://github.com/acme/repo", "batch")
