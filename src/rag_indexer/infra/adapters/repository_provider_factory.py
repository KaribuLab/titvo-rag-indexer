import logging

from rag_indexer.domain.ports.repository_provider import IRepositoryProvider
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
