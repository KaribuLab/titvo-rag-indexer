import logging
import os
import shutil
import tempfile
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from rag_indexer.application.cli_manifest_merge import plan_cli_merge
from rag_indexer.domain.dto.checkpoint_config_dto import CheckpointConfig
from rag_indexer.domain.dto.file_content_dto import FileContent
from rag_indexer.domain.dto.index_result_dto import IndexResultDto
from rag_indexer.domain.ports.artifact_store_port import IArtifactStorePort
from rag_indexer.domain.ports.code_splitter_port import ICodeSplitter
from rag_indexer.domain.ports.embedding_provider import IEmbeddingProvider
from rag_indexer.domain.ports.repository_provider import IRepositoryProvider
from rag_indexer.infra.adapters.sqlite_vec_store_adapter import SqliteVecStoreAdapter

LOGGER = logging.getLogger(__name__)

_DEFAULT_DB_PATH = "/tmp/rag_index.db"
_DELTA_DB_PATH = "/tmp/rag_index_delta.db"
_CLI_DB_PATH = "/tmp/rag_index_cli.db"
# Wait up to 300s for another CLI job's lock; poll every 5s. Stale locks
# (expires_at in the past) are taken over. TTL comes from CheckpointConfig.
_DEFAULT_CLI_LOCK_WAIT_SECONDS = 300
_DEFAULT_CLI_LOCK_POLL_SECONDS = 5


class IndexRepositoryUseCase:
    def __init__(
        self,
        repository_provider: IRepositoryProvider,
        code_splitter: ICodeSplitter,
        embedding_provider: IEmbeddingProvider,
        artifact_store: IArtifactStorePort,
        checkpoint_config: Optional[CheckpointConfig] = None,
        job_id: Optional[str] = None,
        db_path: str = _DEFAULT_DB_PATH,
    ):
        self.repository_provider = repository_provider
        self.code_splitter = code_splitter
        self.embedding_provider = embedding_provider
        self.artifact_store = artifact_store
        self.checkpoint_config = checkpoint_config or CheckpointConfig()
        self.job_id = job_id or os.getenv("AWS_BATCH_JOB_ID") or uuid.uuid4().hex[:8]
        self.db_path = db_path
        # Lock state initialized lazily on execute(); _execute_full/_process_files
        # remain safe to call directly (lock renewal becomes a no-op).
        self._lock_acquired_at: Optional[datetime] = None
        self._lock_renew_interval: timedelta = timedelta(
            minutes=self.checkpoint_config.lock_renew_interval_minutes
        )
        self._current_lock_etag: Optional[str] = None

    def execute(
        self,
        repository_url: str,
        branch: str = "",
        commit_sha: Optional[str] = None,
    ) -> IndexResultDto:
        LOGGER.info("Starting indexing for repository: %s", repository_url)

        if not branch:
            raise ValueError("branch is required")

        # === Distributed lock acquisition ===
        # The lock is acquired BEFORE any expensive work (get_files, embed) so a
        # second job that loses the race exits with no OpenAI spent.
        lock_ttl = self.checkpoint_config.lock_ttl_minutes
        acquired = self.artifact_store.acquire_lock(
            repository_url=repository_url,
            branch=branch,
            owner=self.job_id,
            ttl_minutes=lock_ttl,
            commit_sha=commit_sha or "",
            aws_batch_job_id=self.job_id,
        )
        if not acquired:
            existing = self.artifact_store.get_lock(repository_url, branch)
            if existing:
                expires_at = existing.get("expires_at")
                try:
                    if expires_at and datetime.fromisoformat(
                        expires_at
                    ) <= datetime.now(timezone.utc):
                        LOGGER.warning(
                            "Stale lock found (expires_at=%s). Taking over.",
                            expires_at,
                        )
                        self.artifact_store.release_lock(
                            repository_url, branch, existing.get("owner", "")
                        )
                        acquired = self.artifact_store.acquire_lock(
                            repository_url=repository_url,
                            branch=branch,
                            owner=self.job_id,
                            ttl_minutes=lock_ttl,
                            commit_sha=commit_sha or "",
                            aws_batch_job_id=self.job_id,
                        )
                        if not acquired:
                            raise RuntimeError(
                                "Lock re-acquisition failed after stale-lock takeover"
                            )
                    else:
                        raise RuntimeError(
                            f"Lock held by {existing.get('owner')} until "
                            f"{existing.get('expires_at')}, "
                            f"cannot start index for {branch}"
                        )
                except ValueError:
                    raise RuntimeError(
                        f"Lock held by {existing.get('owner')} until "
                        f"{existing.get('expires_at')}, cannot start index for {branch}"
                    )
            else:
                raise RuntimeError(
                    "Lock acquire failed and no existing lock found (race)"
                )

        # Per-run timing for lock renewal
        self._lock_acquired_at = datetime.now(timezone.utc)
        self._lock_renew_interval = timedelta(
            minutes=self.checkpoint_config.lock_renew_interval_minutes
        )
        self._current_lock_etag = self._read_current_lock_etag(repository_url, branch)

        try:
            if commit_sha:
                return self._execute_delta(repository_url, branch, commit_sha)
            resolved = self.repository_provider.resolve_branch_sha(
                repository_url, branch
            )
            return self._execute_full(repository_url, branch, resolved)
        finally:
            try:
                self.artifact_store.release_lock(
                    repository_url=repository_url,
                    branch=branch,
                    owner=self.job_id,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Failed to release lock: %s", exc)

    # ---------- CLI (S3 snapshot, no Git) ----------

    def execute_cli(
        self,
        repository_url: str,
        batch_id: str,
        lock_wait_seconds: int = _DEFAULT_CLI_LOCK_WAIT_SECONDS,
        lock_poll_seconds: int = _DEFAULT_CLI_LOCK_POLL_SECONDS,
    ) -> IndexResultDto:
        """Index from CLI temp batch into ``{repo}/cli/latest/`` with merge.

        Acquires ``{repo}/cli/lock.json`` waiting up to ``lock_wait_seconds``
        (default 300). Stale locks (expires_at <= now) are released and taken
        over. Fails if the wait budget expires while another owner holds a
        fresh lock.
        """
        provider = self.repository_provider
        if not hasattr(provider, "scan_manifest") or not hasattr(
            provider, "get_files_for_paths"
        ):
            raise TypeError(
                "execute_cli requires a CLI snapshot repository provider "
                "(scan_manifest + get_files_for_paths)"
            )
        LOGGER.info(
            "Starting CLI indexing for repository=%s batch_id=%s",
            repository_url,
            batch_id,
        )

        lock_ttl = self.checkpoint_config.lock_ttl_minutes
        self._acquire_cli_lock_with_wait(
            repository_url=repository_url,
            batch_id=batch_id,
            lock_ttl=lock_ttl,
            wait_seconds=lock_wait_seconds,
            poll_seconds=lock_poll_seconds,
        )
        self._lock_acquired_at = datetime.now(timezone.utc)
        self._lock_renew_interval = timedelta(
            minutes=self.checkpoint_config.lock_renew_interval_minutes
        )
        cli_lock = self.artifact_store.get_cli_lock(repository_url)
        self._current_lock_etag = cli_lock.get("etag") if cli_lock else None

        try:
            return self._execute_cli_merge(repository_url, batch_id, provider)
        finally:
            try:
                self.artifact_store.release_cli_lock(repository_url, self.job_id)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Failed to release CLI lock: %s", exc)

    def _acquire_cli_lock_with_wait(
        self,
        repository_url: str,
        batch_id: str,
        lock_ttl: int,
        wait_seconds: int,
        poll_seconds: int,
    ) -> None:
        deadline = time.monotonic() + wait_seconds
        while True:
            acquired = self.artifact_store.acquire_cli_lock(
                repository_url=repository_url,
                owner=self.job_id,
                ttl_minutes=lock_ttl,
                batch_id=batch_id,
                aws_batch_job_id=self.job_id,
            )
            if acquired:
                return

            existing = self.artifact_store.get_cli_lock(repository_url)
            if existing:
                expires_at = existing.get("expires_at")
                try:
                    expired = bool(
                        expires_at
                        and datetime.fromisoformat(expires_at)
                        <= datetime.now(timezone.utc)
                    )
                except ValueError:
                    expired = False
                if expired:
                    LOGGER.warning(
                        "Stale CLI lock found (expires_at=%s). Taking over.",
                        expires_at,
                    )
                    self.artifact_store.release_cli_lock(
                        repository_url, existing.get("owner", "")
                    )
                    continue

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                holder = existing.get("owner") if existing else "unknown"
                raise RuntimeError(
                    f"CLI lock wait timed out after {wait_seconds}s "
                    f"(held by {holder})"
                )
            sleep_for = min(poll_seconds, remaining)
            LOGGER.info(
                "CLI lock held; waiting %.1fs (remaining budget %.1fs)",
                sleep_for,
                remaining,
            )
            time.sleep(sleep_for)

    def _execute_cli_merge(
        self,
        repository_url: str,
        batch_id: str,
        provider,
    ) -> IndexResultDto:
        manifest = provider.scan_manifest
        new_files = manifest.get("files") or []
        previous = self.artifact_store.get_cli_latest_manifest(repository_url)
        prev_files = (previous or {}).get("files") if previous else None
        plan = plan_cli_merge(prev_files, new_files)

        commit_ref = str(manifest.get("commit") or batch_id)

        if plan.inventory_identical and previous is not None:
            LOGGER.info(
                "CLI inventory identical (%d files); no-op merge for %s",
                len(new_files),
                repository_url,
            )
            return IndexResultDto(
                repository_url=repository_url,
                commit_sha=commit_ref,
                is_delta=True,
                chunks_indexed=0,
                files_processed=0,
                files_skipped_resume=len(plan.skip),
            )

        self.db_path = _CLI_DB_PATH
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

        had_previous = previous is not None
        prev_paths = {item["path"] for item in (prev_files or [])}
        if had_previous:
            downloaded = self.artifact_store.download_cli_latest_db(
                repository_url, self.db_path
            )
            if not downloaded:
                LOGGER.warning(
                    "CLI manifest exists but index.db missing; "
                    "rebuilding from inventory"
                )
                plan = plan_cli_merge(None, new_files)
                had_previous = False
                prev_paths = set()

        vector_store = self._open_db(self.db_path)

        # Drop absent paths; clear chunks for paths we will re-embed.
        to_clear = sorted(plan.drop | (plan.embed & prev_paths))
        if to_clear:
            vector_store.delete_by_file_paths(to_clear)

        files = provider.get_files_for_paths(set(plan.embed))
        LOGGER.info(
            "CLI merge: embed=%d skip=%d drop=%d",
            len(plan.embed),
            len(plan.skip),
            len(plan.drop),
        )

        chunks_indexed = self._process_files_cli(
            vector_store=vector_store,
            files=files,
            repository_url=repository_url,
        )

        meta = {
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            "batch_id": batch_id,
            "commit": manifest.get("commit"),
            "branch": manifest.get("branch"),
            "scope": manifest.get("scope"),
        }
        self.artifact_store.upload_cli_latest(
            repository_url, self.db_path, manifest, meta
        )

        try:
            os.remove(self.db_path)
        except OSError:
            pass

        return IndexResultDto(
            repository_url=repository_url,
            commit_sha=commit_ref,
            is_delta=had_previous,
            chunks_indexed=chunks_indexed,
            files_processed=len(files),
            files_skipped_resume=len(plan.skip),
        )

    def _process_files_cli(
        self,
        vector_store: SqliteVecStoreAdapter,
        files: list[FileContent],
        repository_url: str,
    ) -> int:
        """Embed files for CLI merge; renew CLI lock, no Git checkpoints."""
        processed_count = 0
        total_chunks = 0

        for file in files:
            self._maybe_renew_cli_lock(repository_url)

            if vector_store.is_file_indexed(file.path):
                LOGGER.info("Skipping file already indexed: path=%s", file.path)
                continue

            chunk_iter = self.code_splitter.iter_chunks(file)
            for chunk_text, embedding in self.embedding_provider.embed_iter(chunk_iter):
                doc_id = str(uuid.uuid4())
                vector_store.insert_one(
                    doc_id=doc_id,
                    file_path=file.path,
                    chunk_text=chunk_text,
                    embedding=embedding,
                )
                total_chunks += 1

            vector_store.mark_file_indexed(file.path)
            processed_count += 1

        return total_chunks

    def _maybe_renew_cli_lock(self, repository_url: str) -> None:
        if self._lock_acquired_at is None:
            return
        now = datetime.now(timezone.utc)
        if now - self._lock_acquired_at < self._lock_renew_interval:
            return
        new_expires = now + timedelta(minutes=self.checkpoint_config.lock_ttl_minutes)
        renewed = self.artifact_store.renew_cli_lock(
            repository_url=repository_url,
            owner=self.job_id,
            etag=self._current_lock_etag or "",
            new_expires_at=new_expires.isoformat(),
        )
        if not renewed:
            raise RuntimeError(
                "CLI lock lost during renewal; aborting run to avoid duplicate embeds"
            )
        LOGGER.info("CLI lock renewed new_expires_at=%s", new_expires.isoformat())
        self._lock_acquired_at = now
        cli_lock = self.artifact_store.get_cli_lock(repository_url)
        self._current_lock_etag = cli_lock.get("etag") if cli_lock else None

    # ---------- full ----------

    def _execute_full(
        self,
        repository_url: str,
        branch: str,
        commit_sha: str,
    ) -> IndexResultDto:
        LOGGER.info(
            "Executing full index for %s@%s (branch: %s)",
            repository_url,
            commit_sha[:7],
            branch,
        )

        # 1. Idempotency: already indexed this exact commit?
        existing_sha = self.artifact_store.get_latest_commit_sha(repository_url, branch)
        if existing_sha == commit_sha:
            LOGGER.info(
                "Commit %s already indexed for branch %s, skipping",
                commit_sha[:7],
                branch,
            )
            return IndexResultDto(
                repository_url=repository_url,
                commit_sha=commit_sha,
                is_delta=False,
                chunks_indexed=0,
                files_processed=0,
                files_skipped_resume=0,
            )

        # 2. Try to download checkpoint (resume mode)
        checkpoint_path = self.artifact_store.download_checkpoint(
            repository_url, branch, commit_sha, self.db_path
        )
        already_indexed: set[str] = set()
        if checkpoint_path:
            try:
                vector_store_probe = self._open_db(self.db_path)
                already_indexed = vector_store_probe.indexed_files_set()
                already_indexed_count = len(already_indexed)
                LOGGER.info(
                    "Resume mode: checkpoint_found=True files_already_indexed=%d",
                    already_indexed_count,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.error(
                    "Checkpoint at %s is corrupted (%s); discarding and starting fresh",
                    checkpoint_path,
                    exc,
                )
                try:
                    os.remove(self.db_path)
                except OSError:
                    pass
                checkpoint_path = None
                already_indexed = set()
        else:
            # No checkpoint: clean local DB
            if os.path.exists(self.db_path):
                os.remove(self.db_path)

        # 3. Optional: restore source snapshot (skip git fetch if present)
        snapshot_path = self.artifact_store.download_source_snapshot(
            repository_url, branch, commit_sha, target_dir=tempfile.gettempdir()
        )
        if snapshot_path:
            try:
                self.repository_provider.restore_from_snapshot(
                    snapshot_path, commit_sha
                )
                LOGGER.info("Source snapshot restored; skipping git fetch")
                files = self.repository_provider.get_files(
                    repository_url, commit_sha, exclude_paths=already_indexed
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "Source snapshot restore failed (%s); falling back to git fetch",
                    exc,
                )
                files = self.repository_provider.get_files(
                    repository_url, commit_sha, exclude_paths=already_indexed
                )
        else:
            LOGGER.info("No source snapshot; running git fetch")
            files = self.repository_provider.get_files(
                repository_url, commit_sha, exclude_paths=already_indexed
            )

        # First successful get_files: upload source snapshot for next resume
        if not snapshot_path and hasattr(self.repository_provider, "get_repo_dir"):
            try:
                repo_dir = str(self.repository_provider.get_repo_dir())
                self.artifact_store.upload_source_snapshot(
                    repository_url, branch, commit_sha, repo_dir
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Could not upload source snapshot: %s", exc)

        # 4. Filter out already-indexed and create vector store
        files_to_process = [f for f in files if f.path not in already_indexed]
        files_skipped_resume = len(files) - len(files_to_process)
        LOGGER.info(
            "Found %d files; %d already indexed; processing %d",
            len(files),
            files_skipped_resume,
            len(files_to_process),
        )

        vector_store = self._open_db(self.db_path)
        chunks_indexed = self._process_files(
            vector_store=vector_store,
            files=files_to_process,
            repository_url=repository_url,
            branch=branch,
            commit_sha=commit_sha,
        )

        # 5. Upload final DB
        self.artifact_store.upload_db(repository_url, branch, commit_sha, self.db_path)

        # 6. Cleanup checkpoints / snapshot
        self.artifact_store.delete_checkpoint(repository_url, branch, commit_sha)
        self.artifact_store.delete_source_snapshot(repository_url, branch, commit_sha)

        try:
            os.remove(self.db_path)
        except OSError:
            pass

        LOGGER.info(
            "Full index complete: %d chunks from %d files (%d skipped by resume)",
            chunks_indexed,
            len(files_to_process),
            files_skipped_resume,
        )
        return IndexResultDto(
            repository_url=repository_url,
            commit_sha=commit_sha,
            is_delta=False,
            chunks_indexed=chunks_indexed,
            files_processed=len(files_to_process),
            files_skipped_resume=files_skipped_resume,
        )

    # ---------- delta ----------

    def _execute_delta(
        self,
        repository_url: str,
        branch: str,
        commit_sha: str,
    ) -> IndexResultDto:
        LOGGER.info(
            "Executing delta index for %s@%s (branch: %s)",
            repository_url,
            commit_sha[:7],
            branch,
        )

        prev_sha = self.artifact_store.get_latest_commit_sha(repository_url, branch)
        if not prev_sha:
            raise ValueError(
                f"No previous index found for branch '{branch}'. Run full index first."
            )

        if prev_sha == commit_sha:
            LOGGER.info(
                "Commit %s already indexed as latest for branch %s, skipping",
                commit_sha[:7],
                branch,
            )
            return IndexResultDto(
                repository_url=repository_url,
                commit_sha=commit_sha,
                is_delta=True,
                chunks_indexed=0,
                files_processed=0,
                files_skipped_resume=0,
            )

        temp_db_path = self.artifact_store.download_latest_db(repository_url, branch)
        if not temp_db_path:
            raise ValueError(
                f"Could not download latest DB for branch '{branch}'. "
                "Index may be corrupted."
            )

        shutil.copy(temp_db_path, _DELTA_DB_PATH)
        os.remove(temp_db_path)
        self.db_path = _DELTA_DB_PATH

        vector_store = self._open_db(self.db_path)

        diff = self.repository_provider.get_changed_files(
            repository_url, prev_sha, commit_sha
        )

        if not diff.added and not diff.modified and not diff.deleted:
            LOGGER.info(
                "No changes detected between %s..%s",
                prev_sha[:7],
                commit_sha[:7],
            )
            os.remove(self.db_path)
            return IndexResultDto(
                repository_url=repository_url,
                commit_sha=commit_sha,
                is_delta=True,
                chunks_indexed=0,
                files_processed=0,
                files_skipped_resume=0,
            )

        LOGGER.info(
            "Delta: %d added, %d modified, %d deleted",
            len(diff.added),
            len(diff.modified),
            len(diff.deleted),
        )

        files_to_delete = diff.modified + diff.deleted
        if files_to_delete:
            vector_store.delete_by_file_paths(files_to_delete)

        all_files = self.repository_provider.get_files(repository_url, commit_sha)
        files_to_fetch = set(diff.added + diff.modified)
        files = [f for f in all_files if f.path in files_to_fetch]

        LOGGER.info("Processing %d files from delta", len(files))

        chunks_indexed = self._process_files(
            vector_store=vector_store,
            files=files,
            repository_url=repository_url,
            branch=branch,
            commit_sha=commit_sha,
        )

        self.artifact_store.upload_db(repository_url, branch, commit_sha, self.db_path)
        os.remove(self.db_path)

        LOGGER.info(
            "Delta index complete: %d chunks from %d files",
            chunks_indexed,
            len(files),
        )
        return IndexResultDto(
            repository_url=repository_url,
            commit_sha=commit_sha,
            is_delta=True,
            chunks_indexed=chunks_indexed,
            files_processed=len(files),
            files_skipped_resume=0,
        )

    # ---------- shared ----------

    def _open_db(self, db_path: str) -> SqliteVecStoreAdapter:
        return SqliteVecStoreAdapter(
            db_path=db_path,
            embedding_provider=self.embedding_provider,
            artifact_store=self.artifact_store,
            repository_url="",
        )

    def _read_current_lock_etag(
        self, repository_url: str, branch: str
    ) -> Optional[str]:
        try:
            lock = self.artifact_store.get_lock(repository_url, branch)
            if lock:
                return lock.get("etag")
        except Exception:  # noqa: BLE001
            return None
        return None

    def _maybe_renew_lock(
        self,
        repository_url: str,
        branch: str,
    ) -> None:
        """Renew the lock if its renewal interval has elapsed. Abort if lost."""
        if self._lock_acquired_at is None:
            # Lock was not acquired (e.g., direct call to _execute_full). Skip.
            return
        now = datetime.now(timezone.utc)
        if now - self._lock_acquired_at < self._lock_renew_interval:
            return
        new_expires = now + timedelta(minutes=self.checkpoint_config.lock_ttl_minutes)
        renewed = self.artifact_store.renew_lock(
            repository_url=repository_url,
            branch=branch,
            owner=self.job_id,
            etag=self._current_lock_etag or "",
            new_expires_at=new_expires.isoformat(),
        )
        if not renewed:
            raise RuntimeError(
                "Lock lost during renewal; aborting run to avoid duplicate embeds"
            )
        LOGGER.info("Lock renewed new_expires_at=%s", new_expires.isoformat())
        self._lock_acquired_at = now
        # refresh etag after renewal
        self._current_lock_etag = self._read_current_lock_etag(repository_url, branch)

    def _maybe_flush_checkpoint(
        self,
        repository_url: str,
        branch: str,
        commit_sha: str,
        processed_count: int,
    ) -> None:
        if processed_count <= 0:
            return
        if processed_count % self.checkpoint_config.every_n_files != 0:
            return
        t0 = time.time()
        try:
            self.artifact_store.upload_checkpoint(
                repository_url, branch, commit_sha, self.db_path
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Checkpoint upload failed (continuing): %s", exc)
            return
        size_mb = 0.0
        try:
            size_mb = round(os.path.getsize(self.db_path) / (1024 * 1024), 2)
        except OSError:
            pass
        LOGGER.info(
            "Checkpoint flushed: files_processed=%d db_size_mb=%s upload_ms=%.0f",
            processed_count,
            size_mb,
            (time.time() - t0) * 1000,
        )

    def _process_files(
        self,
        vector_store: SqliteVecStoreAdapter,
        files: list[FileContent],
        repository_url: str,
        branch: str,
        commit_sha: str,
    ) -> int:
        """Stream-process files: split → embed → insert → mark indexed → flush.

        Memory peak per chunk: ~10 KB (chunk text + one embedding).
        """
        processed_count = 0
        total_chunks = 0

        for file in files:
            self._maybe_renew_lock(repository_url, branch)

            # Defense in depth: if another job (perhaps with a stale lock) already
            # indexed this file, skip it to avoid duplicate embeddings.
            if vector_store.is_file_indexed(file.path):
                LOGGER.info("Skipping file already indexed: path=%s", file.path)
                continue

            # Stream chunks for this file. The provider consumes the chunk
            # iterator in batches and yields each chunk next to its vector, so
            # the pairs stay aligned without reading the iterator twice.
            chunk_iter = self.code_splitter.iter_chunks(file)

            for chunk_text, embedding in self.embedding_provider.embed_iter(chunk_iter):
                doc_id = str(uuid.uuid4())
                vector_store.insert_one(
                    doc_id=doc_id,
                    file_path=file.path,
                    chunk_text=chunk_text,
                    embedding=embedding,
                )
                total_chunks += 1

            # All chunks of this file are committed; mark it indexed.
            vector_store.mark_file_indexed(file.path)
            processed_count += 1
            self._maybe_flush_checkpoint(
                repository_url, branch, commit_sha, processed_count
            )

        return total_chunks
