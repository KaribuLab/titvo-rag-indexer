import logging
import os
from logging.config import dictConfig
from typing import Any

import boto3

from logging_config import config
from rag_indexer.application.index_repository_use_case import IndexRepositoryUseCase
from rag_indexer.infra.adapters.langchain_code_splitter import LangChainCodeSplitter
from rag_indexer.infra.adapters.langchain_embedding_adapter import (
    LangChainEmbeddingAdapter,
)
from rag_indexer.infra.adapters.repository_provider_factory import (
    create_repository_provider,
)
from rag_indexer.infra.adapters.s3_artifact_store_adapter import (
    S3ArtifactStoreAdapter,
)
from shared.infra.adapters.aws_configuration_adapter import AwsConfigurationAdapter
from shared.infra.adapters.aws_secrets_adapter import AwsSecretsAdapter
from shared.infra.services.encryption_service import EncryptionService

dictConfig(config)

LOGGER = logging.getLogger(__name__)


def create_boto3_client(service_name: str) -> Any:
    aws_endpoint = os.getenv("AWS_ENDPOINT")
    if aws_endpoint is not None:
        return boto3.client(service_name, endpoint_url=aws_endpoint)
    return boto3.client(service_name)


def main():
    repo_url = os.getenv("TITVO_REPO_URL")
    commit_sha = os.getenv("TITVO_COMMIT_SHA")
    branch = os.getenv("TITVO_BRANCH")

    LOGGER.debug(
        "Starting rag-indexer with repo=%s, commit=%s, branch=%s",
        repo_url,
        commit_sha,
        branch,
    )

    if repo_url is None:
        raise ValueError("TITVO_REPO_URL is not set")
    if branch is None:
        raise ValueError(
            "TITVO_BRANCH is required. Use TITVO_BRANCH for full index or "
            "TITVO_BRANCH + TITVO_COMMIT_SHA for delta index.",
        )

    config_table_name = os.getenv("TITVO_DYNAMO_CONFIGURATION_TABLE_NAME")
    LOGGER.debug("Config table name: %s", config_table_name)
    if config_table_name is None:
        raise ValueError("TITVO_DYNAMO_CONFIGURATION_TABLE_NAME is not set")

    encryption_key_name = os.getenv("TITVO_ENCRYPTION_KEY_NAME")
    LOGGER.debug("Encryption key name: %s", encryption_key_name)
    if encryption_key_name is None:
        raise ValueError("TITVO_ENCRYPTION_KEY_NAME is not set")

    configuration_provider = AwsConfigurationAdapter(
        dynamodb_client=create_boto3_client("dynamodb"),
        table_name=config_table_name,
        encryption_service=EncryptionService(
            secrets_provider=AwsSecretsAdapter(
                client=create_boto3_client("secretsmanager"),
                key_name=encryption_key_name,
            ),
        ),
    )

    repository_provider = create_repository_provider(
        url=repo_url,
        configuration_provider=configuration_provider,
    )
    try:
        chunk_size = int(os.getenv("TITVO_CHUNK_SIZE", "1000"))
        chunk_overlap = int(os.getenv("TITVO_CHUNK_OVERLAP", "200"))
        code_splitter = LangChainCodeSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        embedding_provider = LangChainEmbeddingAdapter(configuration_provider)
        artifact_store = S3ArtifactStoreAdapter(
            s3_client=create_boto3_client("s3"),
            configuration_provider=configuration_provider,
        )
        use_case = IndexRepositoryUseCase(
            repository_provider=repository_provider,
            code_splitter=code_splitter,
            embedding_provider=embedding_provider,
            artifact_store=artifact_store,
        )
        result = use_case.execute(
            repository_url=repo_url,
            branch=branch,
            commit_sha=commit_sha,
        )

        LOGGER.info(
            "Indexing complete: repo=%s, commit=%s, is_delta=%s, "
            "chunks=%d, files=%d",
            result.repository_url,
            result.commit_sha[:7],
            result.is_delta,
            result.chunks_indexed,
            result.files_processed,
        )
    finally:
        repository_provider.close()


if __name__ == "__main__":
    main()
