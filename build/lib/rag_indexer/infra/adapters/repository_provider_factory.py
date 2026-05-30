import logging

from rag_indexer.domain.ports.repository_provider import IRepositoryProvider
from rag_indexer.infra.adapters.bitbucket_api_adapter import BitbucketApiAdapter
from rag_indexer.infra.adapters.github_api_adapter import GitHubApiAdapter

LOGGER = logging.getLogger(__name__)

_GITHUB_HOST = "github.com"
_BITBUCKET_HOST = "bitbucket.org"


def create_repository_provider(
    url: str, github_token: str, bitbucket_token: str
) -> IRepositoryProvider:
    """Create the appropriate repository provider based on URL."""
    if _GITHUB_HOST in url:
        LOGGER.debug("Using GitHub API adapter for %s", url)
        return GitHubApiAdapter(token=github_token)
    elif _BITBUCKET_HOST in url:
        LOGGER.debug("Using Bitbucket API adapter for %s", url)
        return BitbucketApiAdapter(token=bitbucket_token)
    else:
        raise ValueError(
            f"Unsupported repository provider for URL: {url}. Only GitHub and Bitbucket are supported."
        )
