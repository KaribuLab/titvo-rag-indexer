import dataclasses


@dataclasses.dataclass
class IndexResultDto:
    repository_url: str
    commit_sha: str
    is_delta: bool
    chunks_indexed: int
    files_processed: int
    files_skipped_resume: int = 0
