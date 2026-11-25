import pytest

from rag_indexer.application.cli_manifest_merge import is_root_policy_path


@pytest.mark.parametrize(
    "path,expected",
    [
        (".titvoignore.yml", True),
        ("./.titvoignore.yml", True),
        ("config/.titvoignore.yml", False),
        ("config/./.titvoignore.yml", False),
        (".titvoignore.yml.backup", False),
        ("", False),
    ],
)
def test_only_normalized_root_policy_is_control_input(path, expected):
    assert is_root_policy_path(path) is expected
