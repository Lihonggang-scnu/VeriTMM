"""Immutable, byte-preserving export profile for scientific run artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .._version import __version__
from ..protocol.responses import validate_artifact_references
from .schema_registry import ARCHIVE_SCHEMA_VERSION


class ArchiveExportError(ValueError):
    """Base error for a refused scientific archive export."""


class ArchiveArtifactMissingError(ArchiveExportError):
    """Raised when a run references an artifact that is not present."""


class ArchiveArtifactCorruptError(ArchiveExportError):
    """Raised when an artifact differs from its recorded size or SHA-256."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArchiveExportError(f"{label} is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise ArchiveExportError(f"{label} must contain a JSON object")
    return value


def _safe_source_path(root: Path, raw_path: Any) -> Path:
    if not isinstance(raw_path, str) or not raw_path or "\\" in raw_path:
        raise ArchiveExportError(f"invalid artifact path: {raw_path!r}")
    relative = PurePosixPath(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ArchiveExportError(f"artifact path escapes run root: {raw_path!r}")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ArchiveExportError(f"artifact path escapes run root: {raw_path!r}") from exc
    return candidate


def _validate_referenced_artifacts(root: Path, run_result: Mapping[str, Any]) -> None:
    references = run_result.get("artifacts", [])
    if not isinstance(references, list):
        raise ArchiveExportError("RUN_RESULT.artifacts must be a list")
    try:
        validate_artifact_references(references)
    except ValueError as exc:
        raise ArchiveExportError(f"invalid RUN_RESULT artifact references: {exc}") from exc
    for reference in references:
        candidate = _safe_source_path(root, reference.get("path"))
        if not candidate.is_file():
            raise ArchiveArtifactMissingError(
                f"referenced artifact is missing: {reference.get('path')}"
            )
        actual_size = candidate.stat().st_size
        actual_hash = _sha256(candidate)
        if actual_size != int(reference.get("size_bytes", -1)):
            raise ArchiveArtifactCorruptError(
                f"referenced artifact size is stale: {reference.get('path')}"
            )
        if actual_hash != reference.get("sha256"):
            raise ArchiveArtifactCorruptError(
                f"referenced artifact hash is stale: {reference.get('path')}"
            )


def export_run(
    run_dir: str | Path,
    dest: str | Path,
    profile: str = "veritmm-archive-v1",
) -> dict[str, Any]:
    """Copy a run byte-for-byte and write an immutable export manifest."""

    if not isinstance(profile, str) or not profile.strip():
        raise ValueError("export profile must be a non-empty string")
    source = Path(run_dir).resolve()
    destination = Path(dest).resolve()
    if not source.is_dir():
        raise ArchiveExportError(f"run directory is missing: {source}")
    if destination == source or source in destination.parents:
        raise ArchiveExportError("export destination must not be inside the source run")
    if destination.exists() and any(destination.iterdir()):
        raise ArchiveExportError("export destination must be empty")

    run_result_path = source / "RUN_RESULT.json"
    if not run_result_path.is_file():
        raise ArchiveArtifactMissingError("RUN_RESULT.json is missing")
    run_result = _read_json(run_result_path, "RUN_RESULT.json")
    _validate_referenced_artifacts(source, run_result)
    certificate_path = source / "PHYSICS_ACCEPTANCE_CERTIFICATE.json"
    certificate = (
        _read_json(certificate_path, "PHYSICS_ACCEPTANCE_CERTIFICATE.json")
        if certificate_path.is_file()
        else {}
    )

    destination.mkdir(parents=True, exist_ok=True)
    artifact_entries: list[dict[str, Any]] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "EXPORT_MANIFEST.json":
            continue
        if path.is_symlink():
            raise ArchiveExportError(f"symlink artifact is not exportable: {path}")
        relative = path.relative_to(source).as_posix()
        target = destination / PurePosixPath(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        source_hash = _sha256(path)
        target_hash = _sha256(target)
        if source_hash != target_hash or path.stat().st_size != target.stat().st_size:
            raise ArchiveArtifactCorruptError(f"byte-preserving copy failed: {relative}")
        artifact_entries.append(
            {
                "path": relative,
                "sha256": source_hash,
                "bytes": int(path.stat().st_size),
            }
        )

    manifest = {
        "profile": profile,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "veritmm_version": __version__,
        "archive_schema_version": ARCHIVE_SCHEMA_VERSION,
        "certificate_id": run_result.get("certificate_id") or certificate.get("certificate_id"),
        "task_hash": run_result.get("task_sha256") or certificate.get("task_sha256"),
        "task_sha256": run_result.get("task_sha256") or certificate.get("task_sha256"),
        "run_id": run_result.get("run_id") or certificate.get("run_id"),
        "artifacts": artifact_entries,
    }
    (destination / "EXPORT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "ArchiveArtifactCorruptError",
    "ArchiveArtifactMissingError",
    "ArchiveExportError",
    "export_run",
]
