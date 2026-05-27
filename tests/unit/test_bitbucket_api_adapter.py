from unittest.mock import MagicMock

import pytest

from rag_indexer.infra.adapters.bitbucket_api_adapter import BitbucketApiAdapter


class TestBitbucketApiAdapter:
    def test_parse_url_valid_https(self):
        adapter = BitbucketApiAdapter(token="test-token")

        workspace, repo = adapter._parse_url("https://bitbucket.org/workspace/repo")
        assert workspace == "workspace"
        assert repo == "repo"

        workspace, repo = adapter._parse_url("https://bitbucket.org/workspace/repo.git")
        assert workspace == "workspace"
        assert repo == "repo"

    def test_parse_url_valid_ssh(self):
        adapter = BitbucketApiAdapter(token="test-token")

        workspace, repo = adapter._parse_url("git@bitbucket.org:workspace/repo")
        assert workspace == "workspace"
        assert repo == "repo"

        workspace, repo = adapter._parse_url("git@bitbucket.org:workspace/repo.git")
        assert workspace == "workspace"
        assert repo == "repo"

    def test_parse_url_invalid(self):
        adapter = BitbucketApiAdapter(token="test-token")

        with pytest.raises(ValueError):
            adapter._parse_url("https://bitbucket.org/")

        with pytest.raises(ValueError):
            adapter._parse_url("invalid-url")

        with pytest.raises(ValueError):
            adapter._parse_url("git@bitbucket.org:workspace")

    def test_resolve_branch_sha(self):
        adapter = BitbucketApiAdapter(token="test-token")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "target": {"hash": "abc123def456"}
        }
        adapter.client.get = MagicMock(return_value=mock_response)

        sha = adapter.resolve_branch_sha("https://bitbucket.org/workspace/repo", "main")

        assert sha == "abc123def456"
        adapter.client.get.assert_called_once_with(
            "/repositories/workspace/repo/refs/branches/main"
        )

    def test_is_excluded(self):
        adapter = BitbucketApiAdapter(token="test-token")

        assert adapter._is_excluded("node_modules/lodash/index.js") is True
        assert adapter._is_excluded(".git/config") is True
        assert adapter._is_excluded("src/main.py") is False
        assert adapter._is_excluded("image.png") is True

    def test_get_changed_files_pagination(self):
        adapter = BitbucketApiAdapter(token="test-token")

        # First page
        mock_response1 = MagicMock()
        mock_response1.json.return_value = {
            "values": [
                {"status": "added", "new": {"path": "src/new.py"}},
                {"status": "modified", "new": {"path": "src/changed.py"}},
            ],
            "next": "https://api.bitbucket.org/next-page"
        }

        # Second page
        mock_response2 = MagicMock()
        mock_response2.json.return_value = {
            "values": [
                {"status": "removed", "old": {"path": "src/deleted.py"}},
            ],
            "next": None
        }

        adapter.client.get = MagicMock(side_effect=[mock_response1, mock_response2])

        result = adapter.get_changed_files(
            "https://bitbucket.org/workspace/repo",
            "abc123",
            "def456"
        )

        assert "src/new.py" in result.added
        assert "src/changed.py" in result.modified
        assert "src/deleted.py" in result.deleted
        assert adapter.client.get.call_count == 2

    def test_get_changed_files_excludes_ignored_files(self):
        adapter = BitbucketApiAdapter(token="test-token")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "values": [
                {"status": "modified", "new": {"path": "src/main.py"}},
                {"status": "modified", "new": {"path": "node_modules/lib/index.js"}},
                {"status": "added", "new": {"path": "image.png"}},
            ],
            "next": None
        }
        adapter.client.get = MagicMock(return_value=mock_response)

        result = adapter.get_changed_files(
            "https://bitbucket.org/workspace/repo",
            "abc123",
            "def456"
        )

        assert "src/main.py" in result.modified
        assert "node_modules/lib/index.js" not in result.modified
        assert "image.png" not in result.added

    def test_fetch_file_content(self):
        adapter = BitbucketApiAdapter(token="test-token")

        mock_response = MagicMock()
        mock_response.text = "file content here"
        adapter.client.get = MagicMock(return_value=mock_response)

        content = adapter._fetch_file_content("ws", "repo", "abc123", "src/file.py")

        assert content == "file content here"
        adapter.client.get.assert_called_once_with(
            "/repositories/ws/repo/src/abc123/src/file.py"
        )
