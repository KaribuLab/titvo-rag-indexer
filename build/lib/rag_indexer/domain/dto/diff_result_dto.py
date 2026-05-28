import dataclasses


@dataclasses.dataclass
class DiffResult:
    added: list[str]
    modified: list[str]
    deleted: list[str]
