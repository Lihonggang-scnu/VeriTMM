from __future__ import annotations

import json
from pathlib import Path

import pytest

from tmm_engine.capabilities import failure_from_exception
from tmm_engine.experiment_store import ExperimentStore, RunLedgerConflictError
from tmm_engine.run_artifacts import write_json


def _record(
    store: ExperimentStore,
    root: Path,
    run_id: str,
    experiment_id: str,
    *,
    parent_run_id: str | None = None,
    created_at: str,
) -> None:
    artifact_root = root / run_id
    artifact_root.mkdir(parents=True)
    write_json(artifact_root / "RUN_RESULT.json", {"run_id": run_id, "ok": True})
    store.record_run(
        run_id=run_id,
        experiment_id=experiment_id,
        parent_run_id=parent_run_id,
        task_sha256="a" * 64,
        execution_identity_sha256="b" * 64,
        operation="simulate",
        status="physically_valid",
        protocol_version="veritmm-agent-v1",
        package_version="0.3.0",
        certificate_id="c" * 64,
        artifact_root=artifact_root,
        tags=("z", "a", "a"),
        hypothesis="test hypothesis",
        change_reason="test change",
        user_metadata={"source": "owned-test", "ordinal": 1},
        created_at=created_at,
        completed_at=created_at,
    )


def test_store_persists_sqlite_index_metadata_and_canonical_artifacts(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / ".veritmm")
    source = tmp_path / "invocation"
    source.mkdir()
    write_json(source / "RUN_RESULT.json", {"run_id": "run_archived", "ok": True})
    write_json(source / "NORMALIZED_TASK.json", {"mode": "simulate"})

    archived = store.archive_artifacts(source, "run_archived")
    assert archived == (tmp_path / ".veritmm" / "runs" / "run_archived").resolve()
    assert (archived / "RUN_RESULT.json").is_file()
    assert (archived / "NORMALIZED_TASK.json").is_file()
    assert store.db_path.is_file()

    record = store.record_run(
        run_id="run_archived",
        experiment_id="exp_store",
        task_sha256="1" * 64,
        execution_identity_sha256="2" * 64,
        operation="simulate",
        status="physically_valid",
        protocol_version="veritmm-agent-v1",
        package_version="0.3.0",
        certificate_id="3" * 64,
        artifact_root=archived,
        cache_hit=False,
        tags=("research", "research", "baseline"),
        hypothesis="Does metadata stay outside the certificate?",
        change_reason="baseline",
        user_metadata={"owner": "verification", "nested": {"value": 7}},
        created_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T00:00:01+00:00",
    )

    restored = store.get_run("run_archived")
    assert restored == record
    assert restored.version_identity_status == "legacy_inconsistent"
    assert restored.to_dict()["version_identity_status"] == "legacy_inconsistent"
    assert restored is not None
    assert restored.tags == ("baseline", "research")
    assert restored.user_metadata == {
        "owner": "verification",
        "nested": {"value": 7},
    }
    assert restored.artifact_root == str(archived)
    assert json.loads((archived / "RUN_RESULT.json").read_text(encoding="utf-8"))[
        "run_id"
    ] == "run_archived"


def test_store_rejects_unknown_lineage_and_cache_references(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / ".veritmm")
    artifact_root = tmp_path / "artifact"
    artifact_root.mkdir()

    common = {
        "run_id": "run_child",
        "experiment_id": "exp_store",
        "task_sha256": "a" * 64,
        "execution_identity_sha256": "b" * 64,
        "operation": "simulate",
        "status": "physically_valid",
        "protocol_version": "veritmm-agent-v1",
        "package_version": "0.3.0",
        "artifact_root": artifact_root,
    }
    try:
        store.record_run(**common, parent_run_id="run_missing")
    except KeyError as exc:
        assert "run_missing" in str(exc)
    else:  # pragma: no cover - assertion branch documents the contract
        raise AssertionError("unknown parent must be rejected")

    try:
        store.record_run(**common, source_run_id="run_missing")
    except KeyError as exc:
        assert "run_missing" in str(exc)
    else:  # pragma: no cover - assertion branch documents the contract
        raise AssertionError("unknown cache source must be rejected")


def test_store_lists_runs_by_experiment_with_deterministic_order(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / ".veritmm")
    artifacts = tmp_path / "artifacts"
    _record(
        store,
        artifacts,
        "run_old",
        "exp_a",
        created_at="2026-01-01T00:00:00+00:00",
    )
    _record(
        store,
        artifacts,
        "run_new",
        "exp_a",
        created_at="2026-01-02T00:00:00+00:00",
    )
    _record(
        store,
        artifacts,
        "run_other",
        "exp_b",
        created_at="2026-01-03T00:00:00+00:00",
    )

    assert [item.run_id for item in store.list_runs(experiment_id="exp_a")] == [
        "run_new",
        "run_old",
    ]
    assert [item.run_id for item in store.list_runs(experiment_id="exp_b")] == [
        "run_other"
    ]
    assert {item.experiment_id for item in store.list_runs()} == {"exp_a", "exp_b"}


def test_duplicate_run_id_cannot_replace_row_or_artifacts(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / ".veritmm")
    source = tmp_path / "source"
    source.mkdir()
    write_json(source / "RUN_RESULT.json", {"run_id": "run_immutable", "value": "original"})
    archived = store.archive_artifacts(source, "run_immutable")
    _record_kwargs = {
        "run_id": "run_immutable",
        "experiment_id": "exp_original",
        "task_sha256": "a" * 64,
        "execution_identity_sha256": "b" * 64,
        "operation": "simulate",
        "status": "completed",
        "protocol_version": "veritmm-agent-v1",
        "package_version": "0.5.1",
        "artifact_root": archived,
        "hypothesis": "original hypothesis",
        "created_at": "2026-08-20T00:00:00+00:00",
        "completed_at": "2026-08-20T00:01:00+00:00",
    }
    original = store.record_run(**_record_kwargs)

    with pytest.raises(RunLedgerConflictError) as captured:
        store.record_run(
            **{
                **_record_kwargs,
                "experiment_id": "exp_replacement",
                "task_sha256": "f" * 64,
                "hypothesis": "replacement attempt",
            }
        )

    assert store.get_run("run_immutable") == original
    assert json.loads((archived / "RUN_RESULT.json").read_text(encoding="utf-8")) == {
        "run_id": "run_immutable",
        "value": "original",
    }
    failure = failure_from_exception(captured.value)
    assert failure.code.value == "provenance_conflict"
    assert failure.actions[0].action_id == "use_fresh_run_identity"


def test_status_update_changes_only_lifecycle_fields(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / ".veritmm")
    artifacts = tmp_path / "artifacts"
    _record(
        store,
        artifacts,
        "run_lifecycle",
        "exp_lifecycle",
        created_at="2026-08-20T00:00:00+00:00",
    )
    before = store.get_run("run_lifecycle")
    assert before is not None
    updated = store.update_run_status(
        "run_lifecycle",
        status="completed",
        completed_at="2026-08-20T01:00:00+00:00",
        certificate_id="d" * 64,
    )

    assert updated.status == "completed"
    assert updated.completed_at == "2026-08-20T01:00:00+00:00"
    assert updated.certificate_id == "d" * 64
    for field in (
        "task_sha256",
        "execution_identity_sha256",
        "parent_run_id",
        "experiment_id",
        "hypothesis",
        "change_reason",
        "artifact_root",
    ):
        assert getattr(updated, field) == getattr(before, field)


def test_archive_and_cache_materialization_refuse_nonempty_destinations(
    tmp_path: Path,
) -> None:
    store = ExperimentStore(tmp_path / ".veritmm")
    source = tmp_path / "source"
    source.mkdir()
    write_json(source / "RUN_RESULT.json", {"run_id": "run_source"})
    occupied = store.artifact_dir("run_occupied")
    occupied.mkdir(parents=True)
    write_json(occupied / "sentinel.json", {"keep": True})

    with pytest.raises(RunLedgerConflictError):
        store.archive_artifacts(source, "run_occupied")
    assert json.loads((occupied / "sentinel.json").read_text(encoding="utf-8")) == {
        "keep": True
    }

    archived_source = store.archive_artifacts(source, "run_source")
    source_record = store.record_run(
        run_id="run_source",
        experiment_id="exp_cache",
        task_sha256="a" * 64,
        execution_identity_sha256="b" * 64,
        operation="simulate",
        status="completed",
        protocol_version="veritmm-agent-v1",
        package_version="0.5.1",
        artifact_root=archived_source,
    )
    cache_destination = tmp_path / "occupied_cache_destination"
    cache_destination.mkdir()
    write_json(cache_destination / "sentinel.json", {"keep": "cache"})
    with pytest.raises(RunLedgerConflictError):
        store.materialize_cache_hit(
            source_record,
            cache_destination,
            new_run_id="run_cache_replay",
        )
    assert json.loads(
        (cache_destination / "sentinel.json").read_text(encoding="utf-8")
    ) == {"keep": "cache"}


def test_cache_materialization_rewrites_every_top_level_run_identity(
    tmp_path: Path,
) -> None:
    store = ExperimentStore(tmp_path / ".veritmm")
    source = tmp_path / "source_all_artifacts"
    source.mkdir()
    write_json(source / "RUN_RESULT.json", {"run_id": "run_source", "ok": True})
    names = (
        "RESULT_SUMMARY.json",
        "RUN_MANIFEST.json",
        "SENSITIVITY_RESULT.json",
        "TOLERANCE_RESULT.json",
        "ROBUSTNESS_REPORT.json",
    )
    for name in names:
        payload = {"run_id": "run_source", "payload": name}
        write_json(source / name, payload)
    archived = store.archive_artifacts(source, "run_source")
    source_record = store.record_run(
        run_id="run_source",
        experiment_id="exp_cache_rewrite",
        task_sha256="a" * 64,
        execution_identity_sha256="b" * 64,
        operation="tolerance",
        status="completed",
        protocol_version="veritmm-agent-v1",
        package_version="0.5.1",
        artifact_root=archived,
    )

    destination = tmp_path / "cache_replay"
    result = store.materialize_cache_hit(
        source_record,
        destination,
        new_run_id="run_replay",
    )

    assert result["run_id"] == "run_replay"
    assert result["source_run_id"] == "run_source"
    for name in names:
        payload = json.loads((destination / name).read_text(encoding="utf-8"))
        assert payload["run_id"] == "run_replay"
        assert payload["cache_hit"] is True
        assert payload["source_run_id"] == "run_source"
        assert payload["artifact_provenance"]["mode"] == "cache_copy"


def test_record_run_batch_rolls_back_parent_when_child_conflicts(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / ".veritmm")
    existing_root = tmp_path / "existing"
    existing_root.mkdir()
    store.record_run(
        run_id="run_existing_child",
        experiment_id="exp_atomic",
        task_sha256="a" * 64,
        execution_identity_sha256="b" * 64,
        operation="simulate",
        status="completed",
        protocol_version="veritmm-agent-v1",
        package_version="0.5.1",
        artifact_root=existing_root,
    )
    parent_root = tmp_path / "new_parent"
    child_root = tmp_path / "new_child"
    parent_root.mkdir()
    child_root.mkdir()
    common = {
        "experiment_id": "exp_atomic",
        "task_sha256": "c" * 64,
        "execution_identity_sha256": "d" * 64,
        "operation": "sweep",
        "status": "completed",
        "protocol_version": "veritmm-agent-v1",
        "package_version": "0.5.1",
    }

    with pytest.raises(RunLedgerConflictError):
        store.record_run_batch(
            (
                {**common, "run_id": "run_new_parent", "artifact_root": parent_root},
                {
                    **common,
                    "run_id": "run_existing_child",
                    "parent_run_id": "run_new_parent",
                    "artifact_root": child_root,
                },
            )
        )

    assert store.get_run("run_new_parent") is None
    assert store.get_run("run_existing_child") is not None


def test_cached_sweep_rejects_child_path_outside_children_before_rewrite(
    tmp_path: Path,
) -> None:
    store = ExperimentStore(tmp_path / ".veritmm")
    source = tmp_path / "unsafe_sweep_source"
    source.mkdir()
    write_json(source / "RUN_RESULT.json", {"run_id": "run_source", "ok": True})
    write_json(
        source / "SWEEP_RESULT.json",
        {
            "run_id": "run_source",
            "children": [
                {
                    "child_run_id": "run_source_child",
                    "artifact_root": ".",
                }
            ],
        },
    )
    archived = store.archive_artifacts(source, "run_source")
    source_record = store.record_run(
        run_id="run_source",
        experiment_id="exp_unsafe_cache",
        task_sha256="a" * 64,
        execution_identity_sha256="b" * 64,
        operation="sweep",
        status="completed",
        protocol_version="veritmm-agent-v1",
        package_version="0.5.1",
        artifact_root=archived,
    )
    store.record_run(
        run_id="run_source_child",
        experiment_id="exp_unsafe_cache",
        parent_run_id="run_source",
        task_sha256="c" * 64,
        execution_identity_sha256="d" * 64,
        operation="simulate",
        status="completed",
        protocol_version="veritmm-agent-v1",
        package_version="0.5.1",
        artifact_root=archived,
    )
    destination = tmp_path / "unsafe_replay"

    with pytest.raises(ValueError, match="under children"):
        store.materialize_cache_hit(
            source_record,
            destination,
            new_run_id="run_replay",
        )

    assert not destination.exists()


def test_direct_cache_materialization_rejects_failed_source_run(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / ".veritmm")
    source = tmp_path / "failed_source"
    source.mkdir()
    write_json(source / "RUN_RESULT.json", {"run_id": "run_failed", "ok": False})
    archived = store.archive_artifacts(source, "run_failed")
    record = store.record_run(
        run_id="run_failed",
        experiment_id="exp_failed_cache",
        task_sha256="a" * 64,
        execution_identity_sha256="b" * 64,
        operation="simulate",
        status="completed",
        protocol_version="veritmm-agent-v1",
        package_version="0.5.1",
        artifact_root=archived,
    )

    with pytest.raises(ValueError, match="successful source run"):
        store.materialize_cache_hit(
            record,
            tmp_path / "failed_replay",
            new_run_id="run_failed_replay",
        )

    assert not (tmp_path / "failed_replay").exists()


def test_successful_sweep_cache_requires_sweep_result(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / ".veritmm")
    source = tmp_path / "incomplete_sweep"
    source.mkdir()
    write_json(source / "RUN_RESULT.json", {"run_id": "run_sweep", "ok": True})
    archived = store.archive_artifacts(source, "run_sweep")
    record = store.record_run(
        run_id="run_sweep",
        experiment_id="exp_incomplete_sweep",
        task_sha256="a" * 64,
        execution_identity_sha256="b" * 64,
        operation="sweep",
        status="completed",
        protocol_version="veritmm-agent-v1",
        package_version="0.5.1",
        artifact_root=archived,
    )

    with pytest.raises(ValueError, match="must contain SWEEP_RESULT"):
        store.materialize_cache_hit(
            record,
            tmp_path / "incomplete_replay",
            new_run_id="run_incomplete_replay",
        )

    assert not (tmp_path / "incomplete_replay").exists()


@pytest.mark.parametrize(
    ("child_result", "error_match"),
    [
        ({"ok": True}, "requires run_id"),
        ({"run_id": "run_source_child", "ok": False}, "successful run"),
    ],
)
def test_successful_sweep_cache_rejects_incomplete_child_result(
    tmp_path: Path,
    child_result: dict[str, object],
    error_match: str,
) -> None:
    store = ExperimentStore(tmp_path / ".veritmm")
    source = tmp_path / "incomplete_child_sweep"
    child_root = source / "children" / "child-000"
    child_root.mkdir(parents=True)
    write_json(source / "RUN_RESULT.json", {"run_id": "run_source", "ok": True})
    write_json(child_root / "RUN_RESULT.json", child_result)
    write_json(
        source / "SWEEP_RESULT.json",
        {
            "run_id": "run_source",
            "children": [
                {
                    "child_run_id": "run_source_child",
                    "artifact_root": "children/child-000",
                }
            ],
        },
    )
    (source / "SWEEP_TABLE.csv").write_text(
        "child_run_id,artifact_root\n"
        "run_source_child,children/child-000\n",
        encoding="utf-8",
    )
    archived = store.archive_artifacts(source, "run_source")
    source_record = store.record_run(
        run_id="run_source",
        experiment_id="exp_incomplete_child",
        task_sha256="a" * 64,
        execution_identity_sha256="b" * 64,
        operation="sweep",
        status="completed",
        protocol_version="veritmm-agent-v1",
        package_version="0.5.1",
        artifact_root=archived,
    )
    store.record_run(
        run_id="run_source_child",
        experiment_id="exp_incomplete_child",
        parent_run_id="run_source",
        task_sha256="c" * 64,
        execution_identity_sha256="d" * 64,
        operation="simulate",
        status="completed",
        protocol_version="veritmm-agent-v1",
        package_version="0.5.1",
        artifact_root=archived / "children" / "child-000",
    )

    destination = tmp_path / "incomplete_child_replay"
    with pytest.raises(ValueError, match=error_match):
        store.materialize_cache_hit(
            source_record,
            destination,
            new_run_id="run_replay",
        )

    assert not destination.exists()
