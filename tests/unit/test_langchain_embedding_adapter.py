"""Tests for LangChainEmbeddingAdapter batching + embed_iter."""

from unittest.mock import MagicMock

import pytest

from rag_indexer.infra.adapters.langchain_embedding_adapter import (
    LangChainEmbeddingAdapter,
)


class FakeEmbeddings:
    """Test double that mimics LangChain's Embeddings interface."""

    def __init__(self):
        self.call_count = 0
        self.failures_remaining = 0
        self.batches_received: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        self.batches_received.append(list(texts))
        if self.failures_remaining > 0:
            self.failures_remaining -= 1
            raise RuntimeError(f"simulated failure ({self.failures_remaining} left)")
        return [[0.1] * 1536 for _ in texts]


class TestExplicitBatching:
    @pytest.fixture
    def adapter(self):
        configuration_provider = MagicMock()
        configuration_provider.get_value.side_effect = lambda key: {
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
        }[key]
        configuration_provider.get_secret.return_value = "test-key"
        return LangChainEmbeddingAdapter(
            configuration_provider=configuration_provider, batch_size=3
        )

    def test_5B_5_embed_partitions_correctly(self, adapter):
        """embed with 3500 chunks and batch_size=1000 makes 4 calls (but here 3)."""
        # Use batch_size=3 with 10 texts → 4 calls (3,3,3,1)
        fake = FakeEmbeddings()
        adapter._embeddings = fake

        result = adapter.embed(["t"] * 10)
        assert fake.call_count == 4
        assert len(result) == 10

    def test_5B_6_empty_texts_no_call(self, adapter):
        fake = FakeEmbeddings()
        adapter._embeddings = fake

        result = adapter.embed([])
        assert fake.call_count == 0
        assert result == []

    def test_5B_7_retry_then_succeed(self, adapter):
        """Batch that fails 2 times then succeeds: retry recovers."""
        fake = FakeEmbeddings()
        fake.failures_remaining = 2
        adapter._embeddings = fake

        result = adapter.embed(["a", "b", "c"])
        assert fake.call_count == 3  # 2 retries + 1 success
        assert len(result) == 3

    def test_5B_8_retry_exhausted_raises(self, adapter):
        """Batch that fails 4+ times eventually raises RuntimeError."""
        fake = FakeEmbeddings()
        fake.failures_remaining = 10  # always fails
        adapter._embeddings = fake

        with pytest.raises(RuntimeError, match="Batch .* failed after 3 attempts"):
            adapter.embed(["a"])


class TestEmbedIter:
    @pytest.fixture
    def adapter(self):
        configuration_provider = MagicMock()
        configuration_provider.get_value.side_effect = lambda key: {
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
        }[key]
        configuration_provider.get_secret.return_value = "test-key"
        return LangChainEmbeddingAdapter(
            configuration_provider=configuration_provider, batch_size=3
        )

    def test_5B_10_embed_iter_emits_one_at_a_time(self, adapter):
        """embed_iter consumes a 5-item iterator with batch_size=3: 2 calls."""
        fake = FakeEmbeddings()
        adapter._embeddings = fake

        result = list(adapter.embed_iter(iter(["t"] * 5)))
        assert len(result) == 5
        assert fake.call_count == 2  # 3 + 2

    def test_5B_11_embed_iter_empty_no_call(self, adapter):
        fake = FakeEmbeddings()
        adapter._embeddings = fake

        result = list(adapter.embed_iter(iter([])))
        assert fake.call_count == 0
        assert result == []
