import logging
from typing import Any

from rag_indexer.domain.ports.repository_provider import IRepositoryProvider
from rag_indexer.infra.adapters.cli_snapshot_repository_adapter import (
    CliSnapshotRepositoryAdapter,
)
from rag_indexer.infra.adapters.ssh_git_repository_adapter import (
    GitRunner,
    SshGitRepositoryAdapter,
    parse_repository_url,
)
from shared.domain.ports.configuration_provider import IConfigurationProvider

LOGGER = logging.getLogger(__name__)


def create_repository_provider(
    url: str,
    configuration_provider: IConfigurationProvider,
    runner: GitRunner | None = None,
) -> IRepositoryProvider:
    """Create an SSH repository provider and load only its host credential."""
    location = parse_repository_url(url)
    private_key = configuration_provider.get_secret(location.secret_name)
    if not private_key:
        raise ValueError(
            f"SSH private key not found in configuration: {location.secret_name}"
        )

    LOGGER.debug("Using Git SSH adapter for %s", location.host)
    return SshGitRepositoryAdapter(
        clone_url=location.clone_url,
        private_key=private_key,
        runner=runner,
    )


def resolve_cli_files_bucket(
    configuration_provider: IConfigurationProvider,
    *,
    env_bucket: str | None = None,
) -> str:
    """Resolve CLI files bucket: env override, then config ``cli_files_bucket_name``."""
    if env_bucket:
        return env_bucket
    bucket = configuration_provider.get_value("cli_files_bucket_name")
    if not bucket:
        raise ValueError(
            "CLI files bucket not set; use TITVO_CLI_FILES_BUCKET or "
            "config key cli_files_bucket_name"
        )
    return bucket


def create_cli_repository_provider(
    s3_client: Any,
    bucket: str,
    batch_id: str,
) -> CliSnapshotRepositoryAdapter:
    """Create a CLI snapshot provider (S3 tar + scan-manifest; no Git)."""
    LOGGER.debug(
        "Using CLI snapshot adapter for batch_id=%s bucket=%s",
        batch_id,
        bucket,
    )
    return CliSnapshotRepositoryAdapter(
        s3_client=s3_client,
        bucket=bucket,
        batch_id=batch_id,
    )
