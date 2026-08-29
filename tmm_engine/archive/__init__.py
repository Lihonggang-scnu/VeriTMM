"""Scientific archive schema, migration, and export helpers."""

from typing import Any


def __getattr__(name: str) -> Any:
    if name in {
        "ARCHIVE_SCHEMA_VERSION",
        "SCHEMA_HISTORY",
        "SchemaTooNewError",
        "detect_schema_version",
    }:
        from . import schema_registry

        return getattr(schema_registry, name)
    if name == "migrate_record":
        from .migration import migrate_record

        return migrate_record
    if name in {
        "ArchiveArtifactCorruptError",
        "ArchiveArtifactMissingError",
        "ArchiveExportError",
        "export_run",
    }:
        from . import export

        return getattr(export, name)
    raise AttributeError(name)

__all__ = [
    "ARCHIVE_SCHEMA_VERSION",
    "SCHEMA_HISTORY",
    "SchemaTooNewError",
    "detect_schema_version",
    "migrate_record",
    "ArchiveArtifactCorruptError",
    "ArchiveArtifactMissingError",
    "ArchiveExportError",
    "export_run",
]
