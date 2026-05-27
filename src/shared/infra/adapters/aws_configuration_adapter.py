from typing import Any

from shared.domain.ports.configuration_provider import IConfigurationProvider
from shared.domain.services.iencryption_service import IEncryptionService


class AwsConfigurationAdapter(IConfigurationProvider):
    def __init__(
        self,
        dynamodb_client: Any,
        table_name: str,
        encryption_service: IEncryptionService,
    ):
        self.encryption_service = encryption_service
        self.dynamodb_client = dynamodb_client
        self.table_name = table_name

    def get_value(self, parameter_id: str) -> str:
        response = self.dynamodb_client.get_item(
            TableName=self.table_name, Key={"parameter_id": {"S": parameter_id}}
        )
        if "Item" not in response or response["Item"] is None:
            return None
        return response["Item"]["value"]["S"]

    def get_secret(self, parameter_id: str) -> str:
        encrypted_value = self.get_value(parameter_id)
        if encrypted_value is None:
            return None
        return self.encryption_service.decrypt(encrypted_value)
