import base64
import logging
from typing import Any

import httpx

from rag_indexer.domain.dto.diff_result_dto import DiffResult
from rag_indexer.domain.dto.file_content_dto import FileContent
from rag_indexer.domain.ports.repository_provider import IRepositoryProvider

LOGGER = logging.getLogger(__name__)

_GITHUB_API_BASE = "https://api.github.com"

_EXCLUDED_EXTENSIONS = {
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".bin",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".svg",
    ".ico",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".7z",
    ".rar",
    ".db",
    ".sqlite",
    ".sqlite3",
}

_EXCLUDED_DIRS = {
    "node_modules/",
    ".git/",
    "__pycache__/",
    ".venv/",
    "venv/",
    ".tox/",
    ".pytest_cache/",
}


class GitHubApiAdapter(IRepositoryProvider):
    def __init__(self, token: str):
        self.token = token
        self.client = httpx.Client(
            base_url=_GITHUB_API_BASE,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )

    def _parse_url(self, url: str) -> tuple[str, str]:
        """Parse https://github.com/owner/repo or git@github.com:owner/repo to (owner, repo)."""
        url = url.rstrip("/").replace(".git", "")

        # Handle SSH format: git@github.com:owner/repo
        if url.startswith("git@github.com:"):
            path = url.replace("git@github.com:", "")
            parts = path.split("/")
            if len(parts) < 2:
                raise ValueError(f"Invalid GitHub SSH URL: {url}")
            return parts[0], parts[1]

        # Handle HTTPS format: https://github.com/owner/repo
        parts = url.replace("https://", "").replace("http://", "").split("/")
        if len(parts) < 3:
            raise ValueError(f"Invalid GitHub URL: {url}")
        return parts[1], parts[2]

    def resolve_branch_sha(self, url: str, branch: str) -> str:
        owner, repo = self._parse_url(url)
        response = self.client.get(f"/repos/{owner}/{repo}/git/ref/heads/{branch}")
        response.raise_for_status()
        data = response.json()
        sha = data.get("object", {}).get("sha")
        if not sha:
            raise ValueError(f"Could not resolve branch {branch} for {url}")
        LOGGER.debug(
            "Resolved branch %s to SHA %s for %s/%s", branch, sha[:7], owner, repo
        )
        return sha

    def _is_excluded(self, path: str) -> bool:
        for excluded in _EXCLUDED_DIRS:
            if excluded in path:
                return True
        for ext in _EXCLUDED_EXTENSIONS:
            if path.endswith(ext):
                return True
        return False

    def get_files(self, url: str, commit_sha: str) -> list[FileContent]:
        owner, repo = self._parse_url(url)
        LOGGER.info("Fetching file tree for %s/%s@%s", owner, repo, commit_sha[:7])

        response = self.client.get(
            f"/repos/{owner}/{repo}/git/trees/{commit_sha}",
            params={"recursive": "1"},
        )
        response.raise_for_status()
        tree_data = response.json()

        files: list[FileContent] = []
        for item in tree_data.get("tree", []):
            if item.get("type") != "blob":
                continue
            path = item.get("path", "")
            if self._is_excluded(path):
                continue

            try:
                content = self._fetch_file_content(owner, repo, commit_sha, path)
                files.append(FileContent(path=path, content=content))
            except Exception as e:
                LOGGER.warning("Failed to fetch %s: %s", path, e)

        LOGGER.info("Fetched %d files from %s/%s", len(files), owner, repo)
        return files

    def _fetch_file_content(self, owner: str, repo: str, ref: str, path: str) -> str:
        response = self.client.get(
            f"/repos/{owner}/{repo}/contents/{path}", params={"ref": ref}
        )
        response.raise_for_status()
        data = response.json()

        encoding = data.get("encoding")
        content = data.get("content", "")

        if encoding == "base64":
            return base64.b64decode(content).decode("utf-8")
        return content

    def get_changed_files(self, url: str, from_sha: str, to_sha: str) -> DiffResult:
        owner, repo = self._parse_url(url)
        LOGGER.info(
            "Fetching diff between %s..%s for %s/%s",
            from_sha[:7],
            to_sha[:7],
            owner,
            repo,
        )

        response = self.client.get(
            f"/repos/{owner}/{repo}/compare/{from_sha}...{to_sha}"
        )
        response.raise_for_status()
        data = response.json()

        added: list[str] = []
        modified: list[str] = []
        deleted: list[str] = []

        for file in data.get("files", []):
            status = file.get("status")
            filename = file.get("filename", "")

            if self._is_excluded(filename):
                continue

            if status == "added":
                added.append(filename)
            elif status == "modified":
                modified.append(filename)
            elif status == "removed":
                deleted.append(filename)
            elif status == "renamed":
                old_filename = file.get("previous_filename", "")
                if old_filename and not self._is_excluded(old_filename):
                    deleted.append(old_filename)
                if not self._is_excluded(filename):
                    added.append(filename)

        LOGGER.info(
            "Diff: %d added, %d modified, %d deleted",
            len(added),
            len(modified),
            len(deleted),
        )
        return DiffResult(added=added, modified=modified, deleted=deleted)
