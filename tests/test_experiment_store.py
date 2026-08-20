from __future__ import annotations

import json
from pathlib import Path

from tmm_engine.experiment_store import ExperimentStore
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

