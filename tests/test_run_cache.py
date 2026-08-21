from __future__ import annotations

import csv
import json
from pathlib import Path

from tmm_engine import (
    ExecutionSettings,
    IlluminationSpec,
    LayerSpec,
    MediumSpec,
    SimulationTask,
    SpectralGrid,
    StackSpec,
)
from tmm_engine.experiment_store import ExperimentStore
from tmm_engine.managed_execution import execute_managed_task
from tmm_engine.protocol import RunResultEnvelope, SweepTaskContract


def _task() -> SimulationTask:
    return SimulationTask(
        stack=StackSpec(
            layers=(LayerSpec(None, 100.0, constant_n=2.0),),
            incident=MediumSpec.air(),
            exit=MediumSpec(constant_n=1.5),
        ),
        spectrum=SpectralGrid(start_nm=500.0, stop_nm=600.0, points=11),
        illumination=IlluminationSpec((0.0,), ("unpolarized",)),
    )


def _settings(**overrides: object) -> ExecutionSettings:
    values: dict[str, object] = {
        "write_plot": False,
        "convergence_max_refinements": 2,
    }
    values.update(overrides)
    return ExecutionSettings(**values)


def _sweep_task():
    return SweepTaskContract.model_validate(
        {
            "schema_version": "sweep-task-v1",
            "mode": "sweep",
            "sweep": {
                "base_simulation": {
                    "stack": {
                        "layers": [{"constant_n": 2.0, "thickness_nm": 100.0}],
                        "incident": {"constant_n": 1.0},
                        "exit": {"constant_n": 1.5},
                    },
                    "spectrum": {"start_nm": 500.0, "stop_nm": 600.0, "points": 7},
                    "illumination": {
                        "angles_deg": [0.0],
                        "polarizations": ["unpolarized"],
                    },
                },
                "parameters": [
                    {
                        "path": "/stack/layers/0/thickness_nm",
                        "values": [90.0, 110.0],
                    }
                ],
                "metrics": [
                    {"name": "mean_R", "observable": "R", "aggregation": "mean"}
                ],
            },
        }
    ).sweep


def test_cache_hit_creates_new_run_and_preserves_source_provenance(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / ".veritmm")
    first = execute_managed_task(
        "simulate",
        _task(),
        tmp_path / "first",
        store=store,
        experiment_id="exp_cache",
        execution_settings=_settings(),
        cache=False,
    )
    second = execute_managed_task(
        "simulate",
        _task(),
        tmp_path / "second",
        store=store,
        experiment_id="exp_cache",
        execution_settings=_settings(),
        cache=True,
    )

    assert first["ok"] is True
    assert second["ok"] is True
    assert second["cache_hit"] is True
    assert second["source_run_id"] == first["run_id"]
    assert second["run_id"] != first["run_id"]
    assert second["task_sha256"] == first["task_sha256"]
    assert second["artifact_provenance"] == {
        "mode": "cache_copy",
        "source_run_id": first["run_id"],
    }
    RunResultEnvelope.model_validate(second)

    source = store.get_run(first["run_id"])
    cached = store.get_run(second["run_id"])
    assert source is not None and cached is not None
    assert cached.cache_hit is True
    assert cached.source_run_id == first["run_id"]
    assert Path(cached.artifact_root, "RUN_RESULT.json").is_file()
    assert Path(cached.artifact_root, "NORMALIZED_TASK.json").is_file()
    assert Path(cached.artifact_root, "PHYSICS_ACCEPTANCE_CERTIFICATE.json").is_file()
    cached_summary = json.loads(
        Path(cached.artifact_root, "RESULT_SUMMARY.json").read_text(encoding="utf-8")
    )
    assert cached_summary["run_id"] == second["run_id"]
    assert cached_summary["source_run_id"] == first["run_id"]


def test_execution_setting_change_invalidates_cache(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / ".veritmm")
    first = execute_managed_task(
        "simulate",
        _task(),
        tmp_path / "first",
        store=store,
        experiment_id="exp_cache_settings",
        execution_settings=_settings(convergence_pointwise_tolerance=5e-3),
        cache=False,
    )
    second = execute_managed_task(
        "simulate",
        _task(),
        tmp_path / "second",
        store=store,
        experiment_id="exp_cache_settings",
        execution_settings=_settings(convergence_pointwise_tolerance=1e-4),
        cache=True,
    )

    assert first["run_id"] != second["run_id"]
    assert second["cache_hit"] is False
    assert second["source_run_id"] is None
    assert store.get_run(second["run_id"]).cache_hit is False  # type: ignore[union-attr]


def test_cached_sweep_rebases_child_run_ids_and_records_lineage(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / ".veritmm")
    sweep = _sweep_task()
    first = execute_managed_task(
        "sweep",
        sweep,
        tmp_path / "first_sweep",
        store=store,
        experiment_id="exp_sweep_cache",
        execution_settings=_settings(),
        cache=False,
    )
    second = execute_managed_task(
        "sweep",
        sweep,
        tmp_path / "second_sweep",
        store=store,
        experiment_id="exp_sweep_cache",
        execution_settings=_settings(),
        cache=True,
    )

    first_sweep = json.loads(
        (tmp_path / "first_sweep" / "SWEEP_RESULT.json").read_text(encoding="utf-8")
    )
    second_sweep = json.loads(
        (tmp_path / "second_sweep" / "SWEEP_RESULT.json").read_text(encoding="utf-8")
    )
    source_ids = [child["child_run_id"] for child in first_sweep["children"]]
    replay_ids = [child["child_run_id"] for child in second_sweep["children"]]

    assert second["cache_hit"] is True
    assert set(source_ids).isdisjoint(replay_ids)
    assert [child["source_child_run_id"] for child in second_sweep["children"]] == source_ids
    replay_records = store.list_children(second["run_id"])
    assert {record.run_id for record in replay_records} == set(replay_ids)
    assert {record.source_run_id for record in replay_records} == set(source_ids)
    assert all(record.cache_hit for record in replay_records)
    with (tmp_path / "second_sweep" / "SWEEP_TABLE.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        table_ids = {row["child_run_id"] for row in csv.DictReader(handle)}
    assert table_ids == set(replay_ids)
    for child, source_id in zip(second_sweep["children"], source_ids, strict=True):
        child_result = json.loads(
            (
                tmp_path
                / "second_sweep"
                / child["artifact_root"]
                / "RUN_RESULT.json"
            ).read_text(encoding="utf-8")
        )
        assert child_result["run_id"] == child["child_run_id"]
        assert child_result["source_run_id"] == source_id
        assert child_result["artifact_provenance"]["source_parent_run_id"] == first["run_id"]


def test_material_catalog_identity_change_invalidates_cache(tmp_path: Path, monkeypatch) -> None:
    import tmm_engine.managed_execution as managed_execution

    store = ExperimentStore(tmp_path / ".veritmm")
    monkeypatch.setattr(managed_execution, "material_catalog_identity", lambda: "catalog-A")
    first = execute_managed_task(
        "simulate",
        _task(),
        tmp_path / "first",
        store=store,
        experiment_id="exp_cache_material",
        execution_settings=_settings(),
        cache=False,
    )
    monkeypatch.setattr(managed_execution, "material_catalog_identity", lambda: "catalog-B")
    second = execute_managed_task(
        "simulate",
        _task(),
        tmp_path / "second",
        store=store,
        experiment_id="exp_cache_material",
        execution_settings=_settings(),
        cache=True,
    )

    assert first["task_sha256"] == second["task_sha256"]
    assert first["run_id"] != second["run_id"]
    assert second["cache_hit"] is False
    assert second["source_run_id"] is None
