"""Tests for archive schema detection and read-time migration."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from tmm_engine import __version__
from tmm_engine.archive.migration import migrate_record
from tmm_engine.archive.schema_registry import (
    ARCHIVE_SCHEMA_VERSION,
    SchemaTooNewError,
    detect_schema_version,
)
from tmm_engine.experiment_store import ExperimentStore


def _record_root(tmp_path: Path, run_id: str = "run_archive") -> Path:
    root = tmp_path / run_id
    root.mkdir()
    return root


def test_v1_record_without_schema_field_is_detected_as_version_one() -> None:
    assert detect_schema_version({"run_id": "run_old"}) == 1


def test_v1_migration_sets_v2_legacy_status_and_evidence() -> None:
    record = {
        "accepted": True,
        "task_sha256": "a" * 64,
        "veritmm_version": "0.6.0",
        "certificate": {"accepted": True, "task_sha256": "a" * 64},
    }

    migrated = migrate_record(record)

    assert migrated["archive_schema_version"] == ARCHIVE_SCHEMA_VERSION == 2
    assert migrated["version_identity_status"] == "legacy_inconsistent"
    assert migrated["migrated_from_schema"] == 1
    assert migrated["evidence_coverage"]["capability_domain"] == "verified"


def test_migration_is_idempotent() -> None:
    record = {"veritmm_version": "0.6.0", "accepted": False}
    once = migrate_record(record)
    twice = migrate_record(once)
    assert twice == once


def test_schema_too_new_is_refused_instead_of_guessed() -> None:
    with pytest.raises(SchemaTooNewError):
        migrate_record({"archive_schema_version": 99})


def test_migration_does_not_mutate_input() -> None:
    record = {
        "accepted": True,
        "veritmm_version": "0.6.0",
        "certificate": {"accepted": True},
    }
    frozen_copy = copy.deepcopy(record)

    migrate_record(record)

    assert record == frozen_copy
    assert "archive_schema_version" not in record
    assert "evidence_coverage" not in record


def test_new_experiment_store_records_carry_archive_schema_v2(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / ".veritmm")
    root = _record_root(tmp_path)
    record = store.record_run(
        run_id="run_archive",
        experiment_id="exp_archive",
        task_sha256="a" * 64,
        execution_identity_sha256="b" * 64,
        operation="simulate",
        status="completed",
        protocol_version="veritmm-agent-v1",
        package_version=__version__,
        artifact_root=root,
    )

    assert record.archive_schema_version == 2
    assert record.to_dict()["archive_schema_version"] == 2
    assert store.get_run("run_archive").archive_schema_version == 2  # type: ignore[union-attr]
