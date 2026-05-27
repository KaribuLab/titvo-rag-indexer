import abc


class IEmbeddingProvider(abc.ABC):
    @abc.abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts and return their embeddings."""
        pass
