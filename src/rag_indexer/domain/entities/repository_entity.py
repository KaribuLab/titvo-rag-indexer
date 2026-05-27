import dataclasses


@dataclasses.dataclass
class RepositoryEntity:
    url: str
    provider: str
    commit_sha: str
