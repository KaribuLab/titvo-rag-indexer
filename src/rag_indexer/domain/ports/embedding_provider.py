import abc
from typing import Iterable, Iterator


class IEmbeddingProvider(abc.ABC):
    @abc.abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts and return their embeddings."""
        pass

    @abc.abstractmethod
    def embed_iter(self, texts: Iterable[str]) -> Iterator[tuple[str, list[float]]]:
        """Stream ``(text, embedding)`` pairs for an iterable of texts.

        Yielding the text next to its vector keeps both aligned by construction:
        the caller must not read the source iterable again, because the
        provider consumes it in batches ahead of the pairs it yields.
        """
        pass
