import dataclasses
from typing import Optional


@dataclasses.dataclass
class LockInfo:
    owner: str
    aws_batch_job_id: Optional[str]
    acquired_at: str
    expires_at: str
    commit_sha: str
    etag: str

    def is_expired(self, now_iso: str) -> bool:
        """Return True if now_iso >= expires_at. Both inputs are ISO 8601 strings."""
        from datetime import datetime

        now = datetime.fromisoformat(now_iso)
        expires = datetime.fromisoformat(self.expires_at)
        return now >= expires
