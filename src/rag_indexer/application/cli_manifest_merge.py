"""Diff scan-manifest files[] by path+sha256 for CLI index merge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class CliMergePlan:
    """Paths to skip (unchanged), embed (new/changed), and drop (absent)."""

    skip: frozenset[str]
    embed: frozenset[str]
    drop: frozenset[str]

    @property
    def inventory_identical(self) -> bool:
        return not self.embed and not self.drop


def _path_sha_map(
    files: Sequence[Mapping[str, Any]] | None,
) -> dict[str, str]:
    if not files:
        return {}
    result: dict[str, str] = {}
    for item in files:
        path = item["path"]
        result[path] = item["sha256"]
    return result


def plan_cli_merge(
    previous_files: Sequence[Mapping[str, Any]] | None,
    new_files: Sequence[Mapping[str, Any]],
) -> CliMergePlan:
    """Compare previous cli/latest manifest files[] with the new inventory.

    - same path+sha256 → skip embed
    - new or changed sha → embed
    - path absent from new files[] → drop from index
    """
    prev = _path_sha_map(previous_files)
    new = _path_sha_map(new_files)

    skip: set[str] = set()
    embed: set[str] = set()
    for path, sha in new.items():
        if prev.get(path) == sha:
            skip.add(path)
        else:
            embed.add(path)
    drop = frozenset(prev.keys() - new.keys())
    return CliMergePlan(
        skip=frozenset(skip),
        embed=frozenset(embed),
        drop=drop,
    )
