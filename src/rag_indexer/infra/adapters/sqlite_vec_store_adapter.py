import logging
import os
import sqlite3
import uuid
from typing import Any, Optional, Set

import sqlite_vec

from rag_indexer.domain.ports.artifact_store_port import IArtifactStorePort
from rag_indexer.domain.ports.embedding_provider import IEmbeddingProvider
from rag_indexer.domain.ports.vector_store_port import IVectorStorePort

LOGGER = logging.getLogger(__name__)

_VECTOR_DIMENSION = 1536  # Default for text-embedding-3-small


class SqliteVecStoreAdapter(IVectorStorePort):
    def __init__(
        self,
        db_path: str,
        embedding_provider: IEmbeddingProvider,
        artifact_store: IArtifactStorePort,
        repository_url: str,
    ):
        self.db_path = db_path
        self.embedding_provider = embedding_provider
        self.artifact_store = artifact_store
        self.repository_url = repository_url
        self._ensure_db()

    def _ensure_db(self) -> None:
        """Ensure the database exists and has the required schema."""
        os.makedirs(
            os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".",
            exist_ok=True,
        )

        conn = sqlite3.connect(self.db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

        # Create the virtual table for vector search
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING vec0(
                id TEXT PRIMARY KEY,
                file_path TEXT,
                chunk_text TEXT,
                embedding FLOAT[{_VECTOR_DIMENSION}] distance_metric=cosine
            )
        """)

        # Tracking table for resume / defense-in-depth
        conn.execute("""
            CREATE TABLE IF NOT EXISTS indexed_files (
                file_path TEXT PRIMARY KEY,
                indexed_at TEXT NOT NULL
            )
        """)

        conn.commit()
        conn.close()
        LOGGER.debug("Initialized sqlite-vec database at %s", self.db_path)

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return conn

    def store(
        self, repository_url: str, commit_sha: str, documents: list[dict[str, Any]]
    ) -> None:
        """Store documents with their embeddings (bulk)."""
        if not documents:
            LOGGER.info("No documents to store")
            return

        texts = [doc["text"] for doc in documents]
        embeddings = self.embedding_provider.embed(texts)

        conn = self._get_connection()
        cursor = conn.cursor()

        for doc, embedding in zip(documents, embeddings):
            doc_id = str(uuid.uuid4())
            cursor.execute(
                """
                INSERT INTO chunks (id, file_path, chunk_text, embedding)
                VALUES (?, ?, ?, ?)
                """,
                (
                    doc_id,
                    doc["file_path"],
                    doc["text"],
                    sqlite_vec.serialize_float32(embedding),
                ),
            )

        conn.commit()
        conn.close()
        LOGGER.info("Stored %d chunks in database", len(documents))

    def insert_one(
        self,
        doc_id: str,
        file_path: str,
        chunk_text: str,
        embedding: list[float],
    ) -> None:
        """Insert a single chunk with its embedding. Autocommit per call.

        Raises ValueError if embedding has the wrong dimension."""
        if len(embedding) != _VECTOR_DIMENSION:
            raise ValueError(
                f"Embedding dimension mismatch: expected {_VECTOR_DIMENSION}, "
                f"got {len(embedding)}"
            )

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO chunks (id, file_path, chunk_text, embedding)
            VALUES (?, ?, ?, ?)
            """,
            (
                doc_id,
                file_path,
                chunk_text,
                sqlite_vec.serialize_float32(embedding),
            ),
        )
        conn.commit()
        conn.close()

    def is_file_indexed(self, file_path: str) -> bool:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM indexed_files WHERE file_path = ? LIMIT 1",
            (file_path,),
        )
        row = cursor.fetchone()
        conn.close()
        return row is not None

    def mark_file_indexed(self, file_path: str) -> None:
        from datetime import datetime, timezone

        now_iso = datetime.now(timezone.utc).isoformat()
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO indexed_files (file_path, indexed_at) VALUES (?, ?)",
            (file_path, now_iso),
        )
        conn.commit()
        conn.close()

    def count_indexed_files(self) -> int:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM indexed_files")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def indexed_files_set(self) -> Set[str]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT file_path FROM indexed_files")
        rows = cursor.fetchall()
        conn.close()
        return {row[0] for row in rows}

    def delete_by_file_paths(self, file_paths: list[str]) -> None:
        """Delete all chunks for the given file paths."""
        if not file_paths:
            return

        conn = self._get_connection()
        cursor = conn.cursor()

        placeholders = ",".join("?" for _ in file_paths)
        cursor.execute(
            f"DELETE FROM chunks WHERE file_path IN ({placeholders})", file_paths
        )

        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        LOGGER.info("Deleted %d chunks for %d files", deleted, len(file_paths))

    def search(self, query: str, k: int) -> list[dict[str, Any]]:
        """Search for k most similar chunks to the query."""
        query_embedding = self.embedding_provider.embed([query])[0]

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, file_path, chunk_text, distance
            FROM chunks
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
            """,
            (sqlite_vec.serialize_float32(query_embedding), k),
        )

        results = []
        for row in cursor.fetchall():
            results.append(
                {
                    "id": row[0],
                    "file_path": row[1],
                    "chunk_text": row[2],
                    "distance": row[3],
                }
            )

        conn.close()
        LOGGER.debug("Search returned %d results", len(results))
        return results

    def get_latest_indexed_commit(self, repository_url: str) -> Optional[str]:
        """Get the latest indexed commit SHA from artifact store."""
        return self.artifact_store.get_latest_commit_sha(repository_url)
