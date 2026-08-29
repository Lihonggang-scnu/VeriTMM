"""Version registry for long-lived VeriTMM scientific archive records."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

ARCHIVE_SCHEMA_VERSION = 2
SCHEMA_HISTORY: dict[int, str] = {
    1: "Pre-registry records without archive_schema_version",
    2: "Versioned records with provenance, evidence, and immutable export support",
}


class SchemaTooNewError(ValueError):
    """Raised when a record requires an archive schema newer than this build."""

    def __init__(self, version: int) -> None:
        self.version = int(version)
        super().__init__(
            f"archive schema version {self.version} is newer than supported version "
            f"{ARCHIVE_SCHEMA_VERSION}; refusing to guess"
        )


def detect_schema_version(record: Mapping[str, Any]) -> int:
    """Return a declared archive schema version, treating absent as historical v1."""

    if not isinstance(record, Mapping):
        raise TypeError("archive record must be a mapping")
    raw = record.get("archive_schema_version", 1)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise ValueError("archive_schema_version must be a positive integer")
    return raw


__all__ = [
    "ARCHIVE_SCHEMA_VERSION",
    "SCHEMA_HISTORY",
    "SchemaTooNewError",
    "detect_schema_version",
]
