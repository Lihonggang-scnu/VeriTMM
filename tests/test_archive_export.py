"""Tests for byte-preserving scientific archive exports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tmm_engine.archive.export import (
    ArchiveArtifactCorruptError,
    ArchiveArtifactMissingError,
    export_run,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_run(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    artifact = run_dir / "payload.bin"
    artifact.write_bytes(b"immutable scientific bytes\x00\x01")
    reference = {
        "kind": "simulation_result",
        "path": "payload.bin",
        "schema_version": "veritmm-simulation-result-v1",
        "sha256": _sha256(artifact),
        "size_bytes": artifact.stat().st_size,
    }
    certificate = {
        "accepted": True,
        "certificate_id": "c" * 64,
        "task_sha256": "a" * 64,
    }
    (run_dir / "PHYSICS_ACCEPTANCE_CERTIFICATE.json").write_text(
        json.dumps(certificate, sort_keys=True), encoding="utf-8"
    )
    run_result = {
        "schema_version": "veritmm-run-result-v1",
        "run_id": "run_archive",
        "task_sha256": "a" * 64,
        "certificate_id": "c" * 64,
        "ok": True,
        "artifacts": [reference],
    }
    (run_dir / "RUN_RESULT.json").write_text(
        json.dumps(run_result, sort_keys=True), encoding="utf-8"
    )
    return run_dir, artifact


def test_export_round_trip_and_hash_stability(tmp_path: Path) -> None:
    run_dir, source_artifact = _make_run(tmp_path)
    destination = tmp_path / "export"

    manifest = export_run(run_dir, destination)

    assert manifest["profile"] == "veritmm-archive-v1"
    assert manifest["archive_schema_version"] == 2
    assert manifest["run_id"] == "run_archive"
    assert manifest["task_hash"] == "a" * 64
    assert manifest["certificate_id"] == "c" * 64
    on_disk_manifest = json.loads(
        (destination / "EXPORT_MANIFEST.json").read_text(encoding="utf-8")
    )
    assert on_disk_manifest == manifest
    for item in manifest["artifacts"]:
        exported = destination / Path(item["path"])
        assert exported.is_file()
        assert exported.stat().st_size == item["bytes"]
        assert _sha256(exported) == item["sha256"]
    assert _sha256(destination / "payload.bin") == _sha256(source_artifact)


def test_corrupted_referenced_artifact_raises_typed_failure(tmp_path: Path) -> None:
    run_dir, artifact = _make_run(tmp_path)
    artifact.write_bytes(b"corrupted")

    with pytest.raises(ArchiveArtifactCorruptError):
        export_run(run_dir, tmp_path / "export")


def test_missing_referenced_artifact_raises_typed_failure(tmp_path: Path) -> None:
    run_dir, artifact = _make_run(tmp_path)
    artifact.unlink()

    with pytest.raises(ArchiveArtifactMissingError):
        export_run(run_dir, tmp_path / "export")
