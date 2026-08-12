import dataclasses
import logging
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Optional, Protocol, Set
from urllib.parse import urlsplit

from rag_indexer.domain.dto.diff_result_dto import DiffResult
from rag_indexer.domain.dto.file_content_dto import FileContent
from rag_indexer.domain.ports.repository_provider import IRepositoryProvider

LOGGER = logging.getLogger(__name__)

_COMMAND_TIMEOUT_SECONDS = 300
_SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")
_PATH_SEGMENT_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
_SSH_URL_PATTERN = re.compile(r"^git@(?P<host>[^:]+):(?P<path>[^?#]+)$")

_PROVIDER_SECRET_NAMES = {
    "github.com": "github_ssh_private_key",
    "bitbucket.org": "bitbucket_ssh_private_key",
}

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
    "node_modules",
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    ".tox",
    ".pytest_cache",
}


@dataclasses.dataclass(frozen=True)
class RepositoryLocation:
    host: str
    owner: str
    repository: str
    clone_url: str
    secret_name: str


class GitRunner(Protocol):
    def run(self, args: list[str], cwd: str | None = None) -> bytes:
        """Execute Git and return stdout bytes."""
        ...


def parse_repository_url(url: str) -> RepositoryLocation:
    """Validate a supported repository URL and normalize it to SSH."""
    host: str
    raw_path: str

    ssh_match = _SSH_URL_PATTERN.fullmatch(url)
    if ssh_match:
        host = ssh_match.group("host").lower()
        raw_path = ssh_match.group("path")
    else:
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.port
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(f"Invalid repository URL: {url}")
        host = parsed.hostname.lower()
        raw_path = parsed.path.lstrip("/")

    secret_name = _PROVIDER_SECRET_NAMES.get(host)
    if secret_name is None:
        raise ValueError(
            f"Unsupported repository provider for URL: {url}. "
            "Only github.com and bitbucket.org are supported."
        )

    clean_path = raw_path.rstrip("/")
    if clean_path.endswith(".git"):
        clean_path = clean_path[:-4]
    parts = clean_path.split("/")
    if len(parts) != 2 or not all(_PATH_SEGMENT_PATTERN.fullmatch(p) for p in parts):
        raise ValueError(f"Invalid repository path for {host}: {raw_path}")

    owner, repository = parts
    return RepositoryLocation(
        host=host,
        owner=owner,
        repository=repository,
        clone_url=f"git@{host}:{owner}/{repository}.git",
        secret_name=secret_name,
    )


class SubprocessGitRunner:
    def __init__(
        self,
        ssh_key_path: Path,
        known_hosts_path: Path,
        timeout_seconds: int = _COMMAND_TIMEOUT_SECONDS,
    ):
        ssh_command = shlex.join(
            [
                "ssh",
                "-i",
                str(ssh_key_path),
                "-o",
                "BatchMode=yes",
                "-o",
                "IdentitiesOnly=yes",
                "-o",
                "StrictHostKeyChecking=yes",
                "-o",
                f"UserKnownHostsFile={known_hosts_path}",
            ]
        )
        self.environment = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "HOME": str(ssh_key_path.parent),
            "LC_ALL": "C",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_SSH_COMMAND": ssh_command,
        }
        self.timeout_seconds = timeout_seconds

    def run(self, args: list[str], cwd: str | None = None) -> bytes:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=cwd,
                env=self.environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                shell=False,
                timeout=self.timeout_seconds,
            )
            return result.stdout
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"Git command timed out: git {args[0]}") from error
        except subprocess.CalledProcessError as error:
            stderr = error.stderr.decode("utf-8", errors="replace").strip()
            detail = stderr.splitlines()[-1] if stderr else "unknown error"
            raise RuntimeError(
                f"Git command failed: git {args[0]}: {detail}"
            ) from error


class SshGitRepositoryAdapter(IRepositoryProvider):
    def __init__(
        self,
        clone_url: str,
        private_key: str,
        runner: GitRunner | None = None,
        known_hosts_path: Path | None = None,
    ):
        self.clone_url = clone_url
        self._key_dir = Path(tempfile.mkdtemp(prefix="rag-ssh-key-"))
        self._key_path = self._key_dir / "id_ed25519"
        self._repo_dir: Path | None = None
        self._fetched_commits: set[str] = set()

        descriptor = os.open(
            self._key_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as key_file:
            key_file.write(private_key)

        trusted_hosts = known_hosts_path or Path(__file__).with_name("known_hosts")
        self._runner = runner or SubprocessGitRunner(
            ssh_key_path=self._key_path,
            known_hosts_path=trusted_hosts,
        )

    def resolve_branch_sha(self, url: str, branch: str) -> str:
        del url
        ref = f"refs/heads/{branch}"
        output = self._runner.run(["ls-remote", "--heads", self.clone_url, ref])
        for line in output.decode("ascii").splitlines():
            sha, separator, remote_ref = line.partition("\t")
            if separator and remote_ref == ref and _SHA_PATTERN.fullmatch(sha):
                return sha.lower()
        raise ValueError(f"Could not resolve branch {branch} for {self.clone_url}")

    def get_files(
        self,
        url: str,
        commit_sha: str,
        exclude_paths: Optional[Set[str]] = None,
    ) -> list[FileContent]:
        del url
        self._fetch_commit(commit_sha)
        assert self._repo_dir is not None

        tree = self._runner.run(
            ["ls-tree", "-r", "-z", "--full-tree", commit_sha],
            cwd=str(self._repo_dir),
        )
        files: list[FileContent] = []
        skipped_excluded = 0
        for entry in tree.split(b"\0"):
            if not entry:
                continue
            metadata, separator, encoded_path = entry.partition(b"\t")
            if not separator:
                continue
            fields = metadata.split()
            if len(fields) != 3 or fields[1] != b"blob":
                continue

            path = encoded_path.decode("utf-8", errors="surrogateescape")
            if self._is_excluded(path):
                continue
            if exclude_paths and path in exclude_paths:
                skipped_excluded += 1
                continue

            object_id = fields[2].decode("ascii")
            try:
                content = self._runner.run(
                    ["cat-file", "blob", object_id],
                    cwd=str(self._repo_dir),
                ).decode("utf-8")
            except UnicodeDecodeError:
                LOGGER.warning("Skipping non-UTF-8 blob: %s", path)
                continue
            except RuntimeError as error:
                LOGGER.warning("Failed to read %s: %s", path, error)
                continue
            files.append(FileContent(path=path, content=content))

        if exclude_paths:
            LOGGER.info(
                "Fetched %d files from %s (skipped %d via exclude_paths)",
                len(files),
                self.clone_url,
                skipped_excluded,
            )
        else:
            LOGGER.info("Fetched %d files from %s", len(files), self.clone_url)
        return files

    def get_changed_files(self, url: str, from_sha: str, to_sha: str) -> DiffResult:
        del url
        self._fetch_commit(from_sha)
        self._fetch_commit(to_sha)
        assert self._repo_dir is not None

        output = self._runner.run(
            ["diff", "--name-status", "-z", "-M", from_sha, to_sha, "--"],
            cwd=str(self._repo_dir),
        )
        fields = [field for field in output.split(b"\0") if field]
        added: list[str] = []
        modified: list[str] = []
        deleted: list[str] = []
        index = 0

        while index < len(fields):
            status = fields[index].decode("ascii")
            index += 1
            if status.startswith(("R", "C")):
                if index + 1 >= len(fields):
                    raise RuntimeError("Malformed Git rename/copy diff output")
                old_path = fields[index].decode("utf-8", errors="surrogateescape")
                new_path = fields[index + 1].decode("utf-8", errors="surrogateescape")
                index += 2
                if status.startswith("R") and not self._is_excluded(old_path):
                    deleted.append(old_path)
                if not self._is_excluded(new_path):
                    added.append(new_path)
                continue

            if index >= len(fields):
                raise RuntimeError("Malformed Git diff output")
            path = fields[index].decode("utf-8", errors="surrogateescape")
            index += 1
            if self._is_excluded(path):
                continue
            if status == "A":
                added.append(path)
            elif status == "D":
                deleted.append(path)
            else:
                modified.append(path)

        return DiffResult(added=added, modified=modified, deleted=deleted)

    def restore_from_snapshot(
        self,
        snapshot_path: str,
        commit_sha: str,
    ) -> None:
        """Move the extracted .git snapshot into a fresh mkdtemp repo dir.

        The snapshot_path is expected to be a directory containing a .git subdir
        (as returned by S3ArtifactStoreAdapter.download_source_snapshot)."""
        snapshot_path_obj = Path(snapshot_path)
        git_dir = (
            snapshot_path_obj / ".git"
            if (snapshot_path_obj / ".git").is_dir()
            else snapshot_path_obj
        )
        if not git_dir.is_dir():
            raise ValueError(
                f"Snapshot path does not contain a .git directory: {snapshot_path}"
            )

        # Wipe any prior local repo
        if self._repo_dir is not None:
            shutil.rmtree(self._repo_dir, ignore_errors=True)
            self._repo_dir = None

        new_dir = Path(tempfile.mkdtemp(prefix="rag-git-repo-"))
        shutil.move(str(git_dir), str(new_dir / ".git"))
        self._repo_dir = new_dir
        # Mark the commit as already fetched so get_files() does not call git fetch.
        normalized = commit_sha.lower()
        if _SHA_PATTERN.fullmatch(normalized):
            self._fetched_commits.add(normalized)
        LOGGER.info("Restored local repo from %s at %s", snapshot_path, self._repo_dir)

    def get_repo_dir(self) -> Path:
        if self._repo_dir is None:
            return self._ensure_repository()
        return self._repo_dir

    def close(self) -> None:
        if self._repo_dir is not None:
            shutil.rmtree(self._repo_dir, ignore_errors=True)
        shutil.rmtree(self._key_dir, ignore_errors=True)
        self._repo_dir = None
        self._fetched_commits.clear()

    def _ensure_repository(self) -> Path:
        if self._repo_dir is None:
            self._repo_dir = Path(tempfile.mkdtemp(prefix="rag-git-repo-"))
            self._runner.run(["init", "--quiet", str(self._repo_dir)])
        return self._repo_dir

    def _fetch_commit(self, commit_sha: str) -> None:
        if not _SHA_PATTERN.fullmatch(commit_sha):
            raise ValueError(f"Invalid commit SHA: {commit_sha}")
        normalized_sha = commit_sha.lower()
        if normalized_sha in self._fetched_commits:
            return
        repo_dir = self._ensure_repository()
        self._runner.run(
            [
                "fetch",
                "--quiet",
                "--depth=1",
                "--no-tags",
                self.clone_url,
                normalized_sha,
            ],
            cwd=str(repo_dir),
        )
        self._fetched_commits.add(normalized_sha)

    @staticmethod
    def _is_excluded(path: str) -> bool:
        pure_path = PurePosixPath(path)
        if any(part in _EXCLUDED_DIRS for part in pure_path.parts):
            return True
        return pure_path.suffix.lower() in _EXCLUDED_EXTENSIONS
