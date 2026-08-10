from unittest.mock import MagicMock, patch

import pytest

import main as main_module


@pytest.fixture
def required_environment(monkeypatch):
    monkeypatch.setenv("TITVO_REPO_URL", "https://github.com/acme/service")
    monkeypatch.setenv("TITVO_BRANCH", "main")
    monkeypatch.setenv("TITVO_DYNAMO_CONFIGURATION_TABLE_NAME", "config-table")
    monkeypatch.setenv("TITVO_ENCRYPTION_KEY_NAME", "encryption-key")
    monkeypatch.delenv("TITVO_COMMIT_SHA", raising=False)


def test_main_closes_repository_provider_after_success(required_environment):
    provider = MagicMock()
    result = MagicMock(
        repository_url="https://github.com/acme/service",
        commit_sha="a" * 40,
        is_delta=False,
        chunks_indexed=1,
        files_processed=1,
    )

    with (
        patch.object(main_module, "create_boto3_client"),
        patch.object(main_module, "AwsConfigurationAdapter"),
        patch.object(main_module, "create_repository_provider", return_value=provider),
        patch.object(main_module, "IndexRepositoryUseCase") as use_case,
    ):
        use_case.return_value.execute.return_value = result
        main_module.main()

    provider.close.assert_called_once_with()


def test_main_preserves_indexing_error_and_closes_provider(required_environment):
    provider = MagicMock()

    with (
        patch.object(main_module, "create_boto3_client"),
        patch.object(main_module, "AwsConfigurationAdapter"),
        patch.object(main_module, "create_repository_provider", return_value=provider),
        patch.object(main_module, "IndexRepositoryUseCase") as use_case,
    ):
        use_case.return_value.execute.side_effect = RuntimeError("index failed")
        with pytest.raises(RuntimeError, match="index failed"):
            main_module.main()

    provider.close.assert_called_once_with()
