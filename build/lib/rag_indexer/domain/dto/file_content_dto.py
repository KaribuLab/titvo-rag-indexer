import dataclasses


@dataclasses.dataclass
class FileContent:
    path: str
    content: str
