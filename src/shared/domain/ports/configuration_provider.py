import abc


class IConfigurationProvider(abc.ABC):
    @abc.abstractmethod
    def get_value(self, name: str) -> str:
        pass

    @abc.abstractmethod
    def get_secret(self, name: str) -> str | None:
        pass
