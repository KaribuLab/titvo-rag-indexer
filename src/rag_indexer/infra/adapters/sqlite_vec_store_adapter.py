import json
import logging
import os
import sqlite3
import uuid
from typing import Any, Optional

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
        # Create directory if needed
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".", exist_ok=True)

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

        conn.commit()
        conn.close()
        LOGGER.debug("Initialized sqlite-vec database at %s", self.db_path)

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return conn

    def store(self, repository_url: str, commit_sha: str, documents: list[dict[str, Any]]) -> None:
        """Store documents with their embeddings."""
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
                (doc_id, doc["file_path"], doc["text"], sqlite_vec.serialize_float32(embedding)),
            )

        conn.commit()
        conn.close()
        LOGGER.info("Stored %d chunks in database", len(documents))

    def delete_by_file_paths(self, file_paths: list[str]) -> None:
        """Delete all chunks for the given file paths."""
        if not file_paths:
            return

        conn = self._get_connection()
        cursor = conn.cursor()

        placeholders = ",".join("?" for _ in file_paths)
        cursor.execute(f"DELETE FROM chunks WHERE file_path IN ({placeholders})", file_paths)

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
            results.append({
                "id": row[0],
                "file_path": row[1],
                "chunk_text": row[2],
                "distance": row[3],
            })

        conn.close()
        LOGGER.debug("Search returned %d results", len(results))
        return results

    def get_latest_indexed_commit(self, repository_url: str) -> Optional[str]:
        """Get the latest indexed commit SHA from artifact store."""
        return self.artifact_store.get_latest_commit_sha(repository_url)
