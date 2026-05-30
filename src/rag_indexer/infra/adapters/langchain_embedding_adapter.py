import logging
from typing import Any

from langchain.embeddings.base import Embeddings
from langchain_openai import OpenAIEmbeddings

from rag_indexer.domain.ports.embedding_provider import IEmbeddingProvider
from shared.domain.ports.configuration_provider import IConfigurationProvider

LOGGER = logging.getLogger(__name__)

_SUPPORTED_PROVIDERS = {"openai"}


class LangChainEmbeddingAdapter(IEmbeddingProvider):
    def __init__(self, configuration_provider: IConfigurationProvider):
        self.configuration_provider = configuration_provider
        self._embeddings: Embeddings | None = None

    def _get_embeddings_client(self) -> Embeddings:
        if self._embeddings is not None:
            return self._embeddings

        provider = self.configuration_provider.get_value("embedding_provider")
        model = self.configuration_provider.get_value("embedding_model")
        api_key = self.configuration_provider.get_secret("embedding_api_key")

        if not provider:
            raise ValueError("embedding_provider not found in configuration")
        if not model:
            raise ValueError("embedding_model not found in configuration")
        if not api_key:
            raise ValueError("embedding_api_key not found in secrets")

        provider_lower = provider.lower()

        if provider_lower not in _SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported embedding provider: {provider}. Supported: {_SUPPORTED_PROVIDERS}"
            )

        if provider_lower == "openai":
            LOGGER.debug("Using OpenAI embeddings with model %s", model)
            self._embeddings = OpenAIEmbeddings(
                model=model,
                api_key=api_key,
            )
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

        return self._embeddings

    def embed(self, texts: list[str]) -> list[list[float]]:
        client = self._get_embeddings_client()
        embeddings = client.embed_documents(texts)
        return embeddings
