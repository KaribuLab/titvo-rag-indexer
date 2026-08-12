import logging
import time
from typing import Iterable, Iterator, List

from langchain.embeddings.base import Embeddings
from langchain_openai import OpenAIEmbeddings

from rag_indexer.domain.ports.embedding_provider import IEmbeddingProvider
from shared.domain.ports.configuration_provider import IConfigurationProvider

LOGGER = logging.getLogger(__name__)

_SUPPORTED_PROVIDERS = {"openai"}

_DEFAULT_BATCH_SIZE = 1000
_MAX_RETRIES = 3
_RETRY_BACKOFFS = (1, 2, 4)


class LangChainEmbeddingAdapter(IEmbeddingProvider):
    def __init__(
        self,
        configuration_provider: IConfigurationProvider,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ):
        self.configuration_provider = configuration_provider
        self.batch_size = batch_size
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
                f"Unsupported embedding provider: {provider}. "
                f"Supported: {_SUPPORTED_PROVIDERS}"
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

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Batch the list and call the SDK per batch. Returns the concatenated list."""
        all_embeddings: List[List[float]] = []
        client = self._get_embeddings_client()
        total_batches = (
            (len(texts) + self.batch_size - 1) // self.batch_size if texts else 0
        )

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_index = i // self.batch_size + 1
            result = self._embed_with_retry(client, batch, batch_index, total_batches)
            all_embeddings.extend(result)

        return all_embeddings

    def embed_iter(self, texts_iter: Iterable[str]) -> Iterator[List[float]]:
        """Yield embeddings one at a time, batching the input iterator on-the-fly.

        This is the streaming-friendly entry point. Memory: peak ~batch_size chunks
        of floats, not the whole list."""
        client = self._get_embeddings_client()
        batch: List[str] = []
        # We need to know total batches for logging, but with an iterator we cannot
        # count up front. Log per-batch index without total.
        for text in texts_iter:
            batch.append(text)
            if len(batch) >= self.batch_size:
                yield from self._embed_with_retry(client, batch, 0, 0)
                batch = []
        if batch:
            yield from self._embed_with_retry(client, batch, 0, 0)

    def _embed_with_retry(
        self,
        client: Embeddings,
        batch: List[str],
        batch_index: int,
        total_batches: int,
    ) -> List[List[float]]:
        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                t0 = time.time()
                result = client.embed_documents(batch)
                duration_ms = (time.time() - t0) * 1000
                if total_batches > 0:
                    LOGGER.info(
                        "Embedded batch %d/%d chunks=%d duration_ms=%.0f",
                        batch_index,
                        total_batches,
                        len(batch),
                        duration_ms,
                    )
                else:
                    LOGGER.info(
                        "Embedded batch chunks=%d duration_ms=%.0f",
                        len(batch),
                        duration_ms,
                    )
                return result
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                backoff = (
                    _RETRY_BACKOFFS[attempt] if attempt < len(_RETRY_BACKOFFS) else 4
                )
                LOGGER.warning(
                    "Retry batch %s attempt=%d error=%s backoff_s=%d",
                    batch_index if total_batches > 0 else "?",
                    attempt + 1,
                    exc,
                    backoff,
                )
                time.sleep(backoff)
        raise RuntimeError(
            f"Batch {batch_index} failed after {_MAX_RETRIES} attempts: {last_error}"
        )
