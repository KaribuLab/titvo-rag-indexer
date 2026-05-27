import base64
from unittest.mock import MagicMock, patch

import pytest

from rag_indexer.infra.adapters.github_api_adapter import GitHubApiAdapter


class TestGitHubApiAdapter:
    def test_parse_url_valid_https(self):
        adapter = GitHubApiAdapter(token="test-token")

        owner, repo = adapter._parse_url("https://github.com/owner/repo")
        assert owner == "owner"
        assert repo == "repo"

        owner, repo = adapter._parse_url("https://github.com/owner/repo.git")
        assert owner == "owner"
        assert repo == "repo"

    def test_parse_url_valid_ssh(self):
        adapter = GitHubApiAdapter(token="test-token")

        owner, repo = adapter._parse_url("git@github.com:owner/repo")
        assert owner == "owner"
        assert repo == "repo"

        owner, repo = adapter._parse_url("git@github.com:owner/repo.git")
        assert owner == "owner"
        assert repo == "repo"

    def test_parse_url_invalid(self):
        adapter = GitHubApiAdapter(token="test-token")

        with pytest.raises(ValueError):
            adapter._parse_url("https://github.com/")

        with pytest.raises(ValueError):
            adapter._parse_url("invalid-url")

        with pytest.raises(ValueError):
            adapter._parse_url("git@github.com:owner")

    def test_resolve_branch_sha(self):
        adapter = GitHubApiAdapter(token="test-token")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "object": {"sha": "abc123def456"}
        }
        adapter.client.get = MagicMock(return_value=mock_response)

        sha = adapter.resolve_branch_sha("https://github.com/owner/repo", "main")

        assert sha == "abc123def456"
        adapter.client.get.assert_called_once_with("/repos/owner/repo/git/ref/heads/main")

    def test_is_excluded(self):
        adapter = GitHubApiAdapter(token="test-token")

        assert adapter._is_excluded("node_modules/lodash/index.js") is True
        assert adapter._is_excluded(".git/config") is True
        assert adapter._is_excluded("src/main.py") is False
        assert adapter._is_excluded("image.png") is True

    def test_get_changed_files(self):
        adapter = GitHubApiAdapter(token="test-token")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "files": [
                {"filename": "src/new.py", "status": "added"},
                {"filename": "src/changed.py", "status": "modified"},
                {"filename": "src/deleted.py", "status": "removed"},
                {"filename": "src/old.py", "previous_filename": "src/renamed.py", "status": "renamed"},
            ]
        }
        adapter.client.get = MagicMock(return_value=mock_response)

        result = adapter.get_changed_files(
            "https://github.com/owner/repo",
            "abc123",
            "def456"
        )

        assert "src/new.py" in result.added
        assert "src/changed.py" in result.modified
        assert "src/deleted.py" in result.deleted
        assert "src/old.py" in result.added
        assert "src/renamed.py" in result.deleted

    def test_get_changed_files_excludes_ignored_files(self):
        adapter = GitHubApiAdapter(token="test-token")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "files": [
                {"filename": "src/main.py", "status": "modified"},
                {"filename": "node_modules/lib/index.js", "status": "modified"},
                {"filename": "image.png", "status": "added"},
            ]
        }
        adapter.client.get = MagicMock(return_value=mock_response)

        result = adapter.get_changed_files(
            "https://github.com/owner/repo",
            "abc123",
            "def456"
        )

        assert "src/main.py" in result.modified
        assert "node_modules/lib/index.js" not in result.modified
        assert "image.png" not in result.added

    def test_fetch_file_content_base64(self):
        adapter = GitHubApiAdapter(token="test-token")

        encoded_content = base64.b64encode(b"Hello, World!").decode()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content": encoded_content,
            "encoding": "base64"
        }
        adapter.client.get = MagicMock(return_value=mock_response)

        content = adapter._fetch_file_content("owner", "repo", "abc123", "src/file.py")

        assert content == "Hello, World!"

    def test_fetch_file_content_plain(self):
        adapter = GitHubApiAdapter(token="test-token")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content": "plain text content",
            "encoding": None
        }
        adapter.client.get = MagicMock(return_value=mock_response)

        content = adapter._fetch_file_content("owner", "repo", "abc123", "src/file.py")

        assert content == "plain text content"
