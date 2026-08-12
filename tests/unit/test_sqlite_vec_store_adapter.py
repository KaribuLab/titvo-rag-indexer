"""Tests for SqliteVecStoreAdapter new methods."""

import os
import tempfile
from unittest.mock import MagicMock

import pytest

from rag_indexer.infra.adapters.sqlite_vec_store_adapter import SqliteVecStoreAdapter


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        yield SqliteVecStoreAdapter(
            db_path=db_path,
            embedding_provider=MagicMock(),
            artifact_store=MagicMock(),
            repository_url="",
        )


class TestIndexedFilesTable:
    def test_2_5_mark_file_indexed_idempotent(self, store):
        store.mark_file_indexed("src/a.py")
        store.mark_file_indexed("src/a.py")  # second call
        assert store.is_file_indexed("src/a.py")
        assert store.count_indexed_files() == 1

    def test_is_file_indexed_false_for_missing(self, store):
        assert not store.is_file_indexed("src/missing.py")

    def test_indexed_files_set_returns_all(self, store):
        store.mark_file_indexed("src/a.py")
        store.mark_file_indexed("src/b.py")
        assert store.indexed_files_set() == {"src/a.py", "src/b.py"}


class TestInsertOne:
    def test_5D_2_insert_one_persists_chunk(self, store):
        """insert_one with a valid vector persists and is searchable."""

        # Make the embedding provider return a real vector for search
        def fake_embed(texts):
            return [[0.1] * 1536 for _ in texts]

        store.embedding_provider.embed.side_effect = fake_embed

        doc_id = "doc-123"
        embedding = [0.1] * 1536
        store.insert_one(doc_id, "src/a.py", "chunk text", embedding)

        assert store.is_file_indexed("src/a.py") is False  # insert_one doesn't mark
        # Verify the chunk is in the DB by searching
        results = store.search("chunk text", k=1)
        assert len(results) == 1
        assert results[0]["file_path"] == "src/a.py"

    def test_5D_3_insert_one_rejects_wrong_dimension(self, store):
        """insert_one with wrong dimension raises ValueError."""
        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            store.insert_one("doc-1", "src/a.py", "text", [0.1] * 100)
