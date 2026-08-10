from unittest.mock import MagicMock

import pytest

from rag_indexer.infra.adapters.repository_provider_factory import (
    create_repository_provider,
)
from rag_indexer.infra.adapters.ssh_git_repository_adapter import (
    SshGitRepositoryAdapter,
    parse_repository_url,
)


@pytest.mark.parametrize(
    ("url", "host", "clone_url", "secret_name"),
    [
        (
            "https://github.com/acme/service",
            "github.com",
            "git@github.com:acme/service.git",
            "github_ssh_private_key",
        ),
        (
            "git@github.com:acme/service.git",
            "github.com",
            "git@github.com:acme/service.git",
            "github_ssh_private_key",
        ),
        (
            "https://bitbucket.org/acme/service.git",
            "bitbucket.org",
            "git@bitbucket.org:acme/service.git",
            "bitbucket_ssh_private_key",
        ),
        (
            "git@bitbucket.org:acme/service",
            "bitbucket.org",
            "git@bitbucket.org:acme/service.git",
            "bitbucket_ssh_private_key",
        ),
    ],
)
def test_parse_repository_url(url, host, clone_url, secret_name):
    location = parse_repository_url(url)

    assert location.host == host
    assert location.clone_url == clone_url
    assert location.secret_name == secret_name


@pytest.mark.parametrize(
    "url",
    [
        "https://evilgithub.com/acme/service",
        "https://github.com.evil.test/acme/service",
        "http://github.com/acme/service",
        "https://user@github.com/acme/service",
        "https://github.com/acme",
        "git@github.com:acme",
        "git@unknown.test:acme/service.git",
        "not-a-url",
    ],
)
def test_parse_repository_url_rejects_invalid_or_unsupported_urls(url):
    with pytest.raises(ValueError):
        parse_repository_url(url)


def test_factory_loads_only_the_selected_provider_key():
    configuration_provider = MagicMock()
    configuration_provider.get_secret.return_value = "private-key"
    runner = MagicMock()

    provider = create_repository_provider(
        "https://github.com/acme/service",
        configuration_provider,
        runner=runner,
    )

    assert isinstance(provider, SshGitRepositoryAdapter)
    configuration_provider.get_secret.assert_called_once_with("github_ssh_private_key")
    provider.close()


def test_factory_fails_when_selected_key_is_missing():
    configuration_provider = MagicMock()
    configuration_provider.get_secret.return_value = None

    with pytest.raises(ValueError, match="bitbucket_ssh_private_key"):
        create_repository_provider(
            "https://bitbucket.org/acme/service",
            configuration_provider,
        )
