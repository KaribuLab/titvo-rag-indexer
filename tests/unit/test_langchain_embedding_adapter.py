"""Tests for LangChainEmbeddingAdapter batching + embed_iter."""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest
from langchain_openai import OpenAIEmbeddings

from rag_indexer.infra.adapters.langchain_embedding_adapter import (
    LangChainEmbeddingAdapter,
)

CLIENT_PATH = "rag_indexer.infra.adapters.langchain_embedding_adapter.OpenAIEmbeddings"


@pytest.fixture
def configuration():
    """Provide mutable configuration with DynamoDB's missing-value semantics."""
    values = {"embedding_provider": "openai", "embedding_model": "external-model"}
    provider = MagicMock()
    provider.get_value.side_effect = values.get
    provider.get_secret.return_value = "test-key"
    return provider, values


@pytest.mark.parametrize("url", [None, "", "   ", " https://example.com/custom/v1 "])
def test_endpoint_configuration(configuration, url):
    """Only configured endpoints override the SDK defaults and tokenization."""
    provider, values = configuration
    if url is not None:
        values["embedding_base_url"] = url
    with patch(CLIENT_PATH) as constructor:
        LangChainEmbeddingAdapter(provider)._get_embeddings_client()
    expected = {"model": "external-model", "api_key": "test-key"}
    if url and url.strip():
        expected.update(base_url=url.strip(), check_embedding_ctx_length=False)
    constructor.assert_called_once_with(**expected)
    provider.get_value.assert_any_call("embedding_base_url")
    provider.get_secret.assert_called_once_with("embedding_api_key")


def test_configuration_failure_propagates(configuration):
    """Configuration failures must not silently select another endpoint."""
    provider, values = configuration

    def read_value(key):
        """Simulate failure specifically while reading the endpoint."""
        if key == "embedding_base_url":
            raise RuntimeError("DynamoDB unavailable")
        return values.get(key)

    provider.get_value.side_effect = read_value
    with patch(CLIENT_PATH) as constructor:
        with pytest.raises(RuntimeError, match="DynamoDB unavailable"):
            LangChainEmbeddingAdapter(provider)._get_embeddings_client()
    constructor.assert_not_called()


def test_endpoint_is_cached_per_instance(configuration):
    """Existing instances retain configuration while new ones reload it."""
    provider, values = configuration
    values["embedding_base_url"] = "https://first.example/v1"
    adapter = LangChainEmbeddingAdapter(provider)
    with patch(CLIENT_PATH) as constructor:
        client = adapter._get_embeddings_client()
        provider.reset_mock()
        values["embedding_base_url"] = "https://second.example/v1"
        assert adapter._get_embeddings_client() is client
        provider.get_value.assert_not_called()
        assert constructor.call_count == 1
        LangChainEmbeddingAdapter(provider)._get_embeddings_client()
        assert constructor.call_count == 2
        assert constructor.call_args.kwargs["base_url"] == values["embedding_base_url"]


@pytest.mark.parametrize("streaming", [False, True])
def test_custom_endpoint_http_payload(configuration, monkeypatch, streaming):
    """Exercise the real SDK against a local transport for both embedding paths."""
    provider, values = configuration
    values["embedding_base_url"] = "https://embeddings.example.com/custom/v1"
    monkeypatch.setenv("OPENAI_API_BASE", "https://ignored.example/v1")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://also-ignored.example/v1")
    requests = []

    def respond(request):
        """Validate a compatible request and return distinct ordered vectors."""
        requests.append(request)
        assert str(request.url) == "https://embeddings.example.com/custom/v1/embeddings"
        assert request.headers["authorization"] == "Bearer test-key"
        body = json.loads(request.content)
        assert body["model"] == "external-model"
        assert all(isinstance(text, str) for text in body["input"])
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": i, "embedding": [float(text), 0.5], "object": "embedding"}
                    for i, text in enumerate(body["input"])
                ],
                "model": "external-model",
                "object": "list",
                "usage": {"prompt_tokens": 3, "total_tokens": 3},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:

        def create_client(**kwargs):
            """Inject only the test transport, preserving production SDK options."""
            return OpenAIEmbeddings(http_client=http_client, **kwargs)

        with patch(CLIENT_PATH, side_effect=create_client):
            adapter = LangChainEmbeddingAdapter(provider, batch_size=2)
            texts = ["1", "2", "3"]
            result = (
                list(adapter.embed_iter(iter(texts)))
                if streaming
                else adapter.embed(texts)
            )
    assert result == [[1.0, 0.5], [2.0, 0.5], [3.0, 0.5]]
    assert [json.loads(r.content)["input"] for r in requests] == [["1", "2"], ["3"]]


def test_absent_endpoint_preserves_sdk_environment(configuration, monkeypatch):
    """Omitting the new option retains the SDK's existing endpoint resolution."""
    provider, _ = configuration
    monkeypatch.setenv("OPENAI_API_BASE", "https://existing.example/v1")
    client = LangChainEmbeddingAdapter(provider)._get_embeddings_client()
    assert client.openai_api_base == "https://existing.example/v1"
    assert client.check_embedding_ctx_length is True


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
