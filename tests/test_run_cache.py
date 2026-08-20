from __future__ import annotations

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
from tmm_engine.protocol import RunResultEnvelope


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

