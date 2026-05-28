import abc


class ICodeSplitter(abc.ABC):
    @abc.abstractmethod
    def split(self, file_path: str, content: str) -> list[str]:
        """Split file content into chunks."""
        pass
