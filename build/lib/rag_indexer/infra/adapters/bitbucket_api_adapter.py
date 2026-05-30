import logging
from typing import Any

import httpx

from rag_indexer.domain.dto.diff_result_dto import DiffResult
from rag_indexer.domain.dto.file_content_dto import FileContent
from rag_indexer.domain.ports.repository_provider import IRepositoryProvider

LOGGER = logging.getLogger(__name__)

_BITBUCKET_API_BASE = "https://api.bitbucket.org/2.0"

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


class BitbucketApiAdapter(IRepositoryProvider):
    def __init__(self, token: str):
        self.token = token
        self.client = httpx.Client(
            base_url=_BITBUCKET_API_BASE,
            headers={
                "Authorization": f"Basic {token}",
                "Accept": "application/json",
            },
        )

    def _parse_url(self, url: str) -> tuple[str, str]:
        """Parse https://bitbucket.org/workspace/repo or git@bitbucket.org:workspace/repo to (workspace, repo)."""
        url = url.rstrip("/").replace(".git", "")

        # Handle SSH format: git@bitbucket.org:workspace/repo
        if url.startswith("git@bitbucket.org:"):
            path = url.replace("git@bitbucket.org:", "")
            parts = path.split("/")
            if len(parts) < 2:
                raise ValueError(f"Invalid Bitbucket SSH URL: {url}")
            return parts[0], parts[1]

        # Handle HTTPS format: https://bitbucket.org/workspace/repo
        parts = url.replace("https://", "").replace("http://", "").split("/")
        if len(parts) < 3:
            raise ValueError(f"Invalid Bitbucket URL: {url}")
        return parts[1], parts[2]

    def resolve_branch_sha(self, url: str, branch: str) -> str:
        workspace, repo = self._parse_url(url)
        response = self.client.get(
            f"/repositories/{workspace}/{repo}/refs/branches/{branch}"
        )
        response.raise_for_status()
        data = response.json()
        target = data.get("target", {})
        sha = target.get("hash")
        if not sha:
            raise ValueError(f"Could not resolve branch {branch} for {url}")
        LOGGER.debug(
            "Resolved branch %s to SHA %s for %s/%s", branch, sha[:7], workspace, repo
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
        workspace, repo = self._parse_url(url)
        LOGGER.info("Fetching file tree for %s/%s@%s", workspace, repo, commit_sha[:7])

        files: list[FileContent] = []
        next_url: str | None = f"/repositories/{workspace}/{repo}/src/{commit_sha}/"

        while next_url:
            response = self.client.get(next_url)
            response.raise_for_status()
            data = response.json()

            for item in data.get("values", []):
                item_type = item.get("type")
                path = item.get("path", "")

                if item_type != "commit_file":
                    continue
                if self._is_excluded(path):
                    continue

                try:
                    content = self._fetch_file_content(
                        workspace, repo, commit_sha, path
                    )
                    files.append(FileContent(path=path, content=content))
                except Exception as e:
                    LOGGER.warning("Failed to fetch %s: %s", path, e)

            next_url = data.get("next")

        LOGGER.info("Fetched %d files from %s/%s", len(files), workspace, repo)
        return files

    def _fetch_file_content(
        self, workspace: str, repo: str, commit: str, path: str
    ) -> str:
        response = self.client.get(
            f"/repositories/{workspace}/{repo}/src/{commit}/{path}"
        )
        response.raise_for_status()
        return response.text

    def get_changed_files(self, url: str, from_sha: str, to_sha: str) -> DiffResult:
        workspace, repo = self._parse_url(url)
        LOGGER.info(
            "Fetching diff between %s..%s for %s/%s",
            from_sha[:7],
            to_sha[:7],
            workspace,
            repo,
        )

        added: list[str] = []
        modified: list[str] = []
        deleted: list[str] = []

        next_url: str | None = (
            f"/repositories/{workspace}/{repo}/diffstat/{from_sha}..{to_sha}"
        )

        while next_url:
            response = self.client.get(next_url)
            response.raise_for_status()
            data = response.json()

            for stat in data.get("values", []):
                old_file = stat.get("old", {})
                new_file = stat.get("new", {})
                status = stat.get("status")

                old_path = old_file.get("path", "") if old_file else ""
                new_path = new_file.get("path", "") if new_file else ""

                if status == "added":
                    if new_path and not self._is_excluded(new_path):
                        added.append(new_path)
                elif status == "modified":
                    if new_path and not self._is_excluded(new_path):
                        modified.append(new_path)
                elif status == "removed":
                    if old_path and not self._is_excluded(old_path):
                        deleted.append(old_path)
                elif status == "renamed":
                    if old_path and not self._is_excluded(old_path):
                        deleted.append(old_path)
                    if new_path and not self._is_excluded(new_path):
                        added.append(new_path)

            next_url = data.get("next")

        LOGGER.info(
            "Diff: %d added, %d modified, %d deleted",
            len(added),
            len(modified),
            len(deleted),
        )
        return DiffResult(added=added, modified=modified, deleted=deleted)
