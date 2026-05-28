import logging
from typing import Any

from shared.domain.ports.secrets_provider import ISecretsProvider

LOGGER = logging.getLogger(__name__)


class AwsSecretsAdapter(ISecretsProvider):
    def __init__(self, client: Any, key_name: str):
        self.client = client
        self.key_name = key_name

    def get_secret(self) -> str | None:
        response = self.client.get_secret_value(SecretId=self.key_name)
        LOGGER.debug("Response from AWS Secrets Manager: %s", response)
        if response["SecretString"] is None:
            LOGGER.warning("Secret string is None")
            return None
        return response["SecretString"]
