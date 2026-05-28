import abc


class IEncryptionService(abc.ABC):
    @abc.abstractmethod
    def encrypt(self, value: str) -> str:
        pass

    @abc.abstractmethod
    def decrypt(self, value: str) -> str:
        pass
