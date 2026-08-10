import logging
import os

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from rag_indexer.domain.ports.code_splitter_port import ICodeSplitter

LOGGER = logging.getLogger(__name__)


def _get_extension_to_language() -> dict[str, Language]:
    """Build extension mapping only for available languages."""
    mapping: dict[str, Language] = {}

    # Map extensions to Language enum values that exist
    lang_map = {
        ".py": "PYTHON",
        ".js": "JS",
        ".ts": "TS",
        ".tsx": "TS",
        ".jsx": "JS",
        ".go": "GO",
        ".rs": "RUST",
        ".java": "JAVA",
        ".kt": "KOTLIN",
        ".cpp": "CPP",
        ".c": "C",
        ".h": "C",
        ".hpp": "CPP",
        ".cs": "CSHARP",
        ".rb": "RUBY",
        ".php": "PHP",
        ".swift": "SWIFT",
        ".scala": "SCALA",
        ".r": "R",
        ".html": "HTML",
        ".css": "CSS",
        ".md": "MARKDOWN",
        ".json": "JSON",
        ".yaml": "YAML",
        ".yml": "YAML",
        ".xml": "XML",
        ".sh": "BASH",
        ".ps1": "POWERSHELL",
    }

    for ext, lang_name in lang_map.items():
        if hasattr(Language, lang_name):
            mapping[ext] = getattr(Language, lang_name)

    return mapping


_EXTENSION_TO_LANGUAGE = _get_extension_to_language()

_EXCLUDED_EXTENSIONS = {
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".bin",
    ".o",
    ".a",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".svg",
    ".ico",
    ".webp",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".mp3",
    ".wav",
    ".flac",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".7z",
    ".rar",
    ".xz",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".mdb",
    ".accdb",
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
}

_EXCLUDED_DIRS = {
    "node_modules",
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    ".tox",
    ".pytest_cache",
    ".mypy_cache",
    "dist",
    "build",
}

_DEFAULT_CHUNK_SIZE = 1000
_DEFAULT_CHUNK_OVERLAP = 200


class LangChainCodeSplitter(ICodeSplitter):
    def __init__(
        self,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _get_language(self, file_path: str) -> Language | None:
        _, ext = os.path.splitext(file_path)
        ext_lower = ext.lower()
        return _EXTENSION_TO_LANGUAGE.get(ext_lower)

    def _is_excluded(self, file_path: str) -> bool:
        # Check excluded directories
        parts = file_path.split(os.sep)
        for part in parts:
            if part in _EXCLUDED_DIRS:
                return True

        # Check excluded extensions
        _, ext = os.path.splitext(file_path)
        if ext.lower() in _EXCLUDED_EXTENSIONS:
            return True

        return False

    def split(self, file_path: str, content: str) -> list[str]:
        if self._is_excluded(file_path):
            LOGGER.debug("Skipping excluded file: %s", file_path)
            return []

        language = self._get_language(file_path)

        if language:
            try:
                splitter = RecursiveCharacterTextSplitter.from_language(
                    language=language,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )
            except Exception as e:
                LOGGER.warning(
                    "Failed to create language splitter for %s: %s", file_path, e
                )
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

        chunks = splitter.split_text(content)
        LOGGER.debug("Split %s into %d chunks", file_path, len(chunks))
        return chunks
