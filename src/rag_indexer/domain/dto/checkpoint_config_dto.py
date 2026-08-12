import dataclasses


@dataclasses.dataclass
class CheckpointConfig:
    every_n_files: int = 100
    s3_key_template: str = (
        "{repo_host}/{owner}/{repo}/branches/{branch}/checkpoints/{commit_sha}/index.db"
    )
    max_snapshot_mb: int = 200
    lock_ttl_minutes: int = 360
    embedding_batch_size: int = 1000
    lock_renew_interval_minutes: int = 30
