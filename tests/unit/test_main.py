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
    monkeypatch.delenv("TITVO_SOURCE_TYPE", raising=False)
    monkeypatch.delenv("TITVO_SOURCE_BATCH_ID", raising=False)


@pytest.fixture
def cli_environment(monkeypatch):
    monkeypatch.setenv("TITVO_REPO_URL", "https://github.com/acme/service")
    monkeypatch.setenv("TITVO_SOURCE_TYPE", "cli")
    monkeypatch.setenv("TITVO_SOURCE_BATCH_ID", "batch-xyz")
    monkeypatch.setenv("TITVO_CLI_FILES_BUCKET", "cli-files-bucket")
    monkeypatch.setenv("TITVO_DYNAMO_CONFIGURATION_TABLE_NAME", "config-table")
    monkeypatch.setenv("TITVO_ENCRYPTION_KEY_NAME", "encryption-key")
    monkeypatch.delenv("TITVO_BRANCH", raising=False)
    monkeypatch.delenv("TITVO_COMMIT_SHA", raising=False)


def test_main_closes_repository_provider_after_success(required_environment):
    provider = MagicMock()
    result = MagicMock(
        repository_url="https://github.com/acme/service",
        commit_sha="a" * 40,
        is_delta=False,
        chunks_indexed=1,
        files_processed=1,
        files_skipped_resume=0,
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
    use_case.return_value.execute.assert_called_once()
    use_case.return_value.execute_cli.assert_not_called()


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


def test_main_cli_uses_cli_provider_and_execute_cli(cli_environment):
    provider = MagicMock()
    result = MagicMock(
        repository_url="https://github.com/acme/service",
        commit_sha="abc1234",
        is_delta=False,
        chunks_indexed=2,
        files_processed=1,
        files_skipped_resume=0,
    )

    with (
        patch.object(main_module, "create_boto3_client"),
        patch.object(main_module, "AwsConfigurationAdapter"),
        patch.object(
            main_module, "create_cli_repository_provider", return_value=provider
        ) as create_cli,
        patch.object(main_module, "create_repository_provider") as create_git,
        patch.object(main_module, "IndexRepositoryUseCase") as use_case,
    ):
        use_case.return_value.execute_cli.return_value = result
        main_module.main()

    create_cli.assert_called_once()
    create_git.assert_not_called()
    use_case.return_value.execute_cli.assert_called_once()
    use_case.return_value.execute.assert_not_called()
    kwargs = use_case.return_value.execute_cli.call_args.kwargs
    assert kwargs["repository_url"] == "https://github.com/acme/service"
    assert kwargs["batch_id"] == "batch-xyz"
    assert kwargs["lock_wait_seconds"] == 300
    provider.close.assert_called_once_with()


def test_main_cli_requires_batch_id(monkeypatch):
    monkeypatch.setenv("TITVO_REPO_URL", "https://github.com/acme/service")
    monkeypatch.setenv("TITVO_SOURCE_TYPE", "cli")
    monkeypatch.setenv("TITVO_DYNAMO_CONFIGURATION_TABLE_NAME", "config-table")
    monkeypatch.setenv("TITVO_ENCRYPTION_KEY_NAME", "encryption-key")
    monkeypatch.delenv("TITVO_SOURCE_BATCH_ID", raising=False)

    with pytest.raises(ValueError, match="TITVO_SOURCE_BATCH_ID"):
        main_module.main()
