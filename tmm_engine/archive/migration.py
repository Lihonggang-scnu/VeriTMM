"""Read-time migrations for scientific archive records."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from .._version import __version__
from ..protocol.evidence import from_certificate
from .schema_registry import ARCHIVE_SCHEMA_VERSION, SchemaTooNewError, detect_schema_version


def _certificate_mapping(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for key in ("certificate", "physics_certificate"):
        candidate = record.get(key)
        if isinstance(candidate, Mapping):
            return candidate
    if "accepted" in record:
        return record
    return None


def migrate_record(record: dict[str, Any]) -> dict[str, Any]:
    """Upgrade one record in memory to the current archive schema.

    Migration is deliberately read-time only: this function never writes a
    migrated mapping back over the original scientific artifact.
    """

    if not isinstance(record, Mapping):
        raise TypeError("archive record must be a mapping")
    # Copy before inspecting or adding fields so callers retain the exact
    # historical input object and can decide whether to persist a new artifact.
    migrated = copy.deepcopy(dict(record))
    version = detect_schema_version(migrated)
    if version > ARCHIVE_SCHEMA_VERSION:
        raise SchemaTooNewError(version)
    if version == ARCHIVE_SCHEMA_VERSION:
        return migrated
    if version != 1:  # pragma: no cover - guarded by the registry history
        raise ValueError(f"unsupported archive schema version: {version}")

    migrated["archive_schema_version"] = ARCHIVE_SCHEMA_VERSION
    recorded_version = migrated.get("veritmm_version")
    migrated["version_identity_status"] = (
        "verified"
        if recorded_version == __version__
        else "legacy_inconsistent"
    )

    certificate = _certificate_mapping(migrated)
    if certificate is not None and "evidence_coverage" not in migrated:
        migrated["evidence_coverage"] = from_certificate(certificate).model_dump(
            mode="json"
        )
    migrated["migrated_from_schema"] = 1
    return migrated


__all__ = ["migrate_record"]
