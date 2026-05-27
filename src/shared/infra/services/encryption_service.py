from base64 import b64decode, b64encode

from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

from shared.domain.ports.secrets_provider import ISecretsProvider
from shared.domain.services.iencryption_service import IEncryptionService


class EncryptionService(IEncryptionService):
    def __init__(self, secrets_provider: ISecretsProvider):
        self.secrets_provider = secrets_provider

    def encrypt(self, value: str) -> str:
        secret = self.secrets_provider.get_secret()
        if secret is None:
            raise ValueError("Encryption key not found")
        key = b64decode(secret)
        cipher = AES.new(key, AES.MODE_ECB)
        encrypted_data = cipher.encrypt(pad(value.encode("utf-8"), AES.block_size))
        return b64encode(encrypted_data).decode("utf-8")

    def decrypt(self, value: str) -> str:
        secret = self.secrets_provider.get_secret()
        if secret is None:
            raise ValueError("Encryption key not found")
        key = b64decode(secret)
        cipher = AES.new(key, AES.MODE_ECB)
        decrypted_data = unpad(cipher.decrypt(b64decode(value)), AES.block_size)
        return decrypted_data.decode("utf-8")
