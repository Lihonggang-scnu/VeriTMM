from __future__ import annotations

import json
from pathlib import Path

import pytest

import tmm_engine.sweep as sweep_module
from tmm_engine import ExecutionSettings
from tmm_engine.protocol import SweepTaskContract
from tmm_engine.sweep import SweepExecutionSettings, execute_sweep, expand_sweep


def _sweep(values: list[float]):
    document = {
        "schema_version": "sweep-task-v1",
        "mode": "sweep",
        "sweep": {
            "base_simulation": {
                "stack": {
                    "layers": [{"constant_n": 2.0, "thickness_nm": 100.0}],
                    "incident": {"constant_n": 1.0},
                    "exit": {"constant_n": 1.5},
                },
                "spectrum": {"start_nm": 500.0, "stop_nm": 600.0, "points": 11},
                "illumination": {
                    "angles_deg": [0.0],
                    "polarizations": ["unpolarized"],
                },
            },
            "parameters": [
                {"path": "/stack/layers/0/thickness_nm", "values": values}
            ],
            "metrics": [{"name": "mean_R", "observable": "R", "aggregation": "mean"}],
        },
    }
    return SweepTaskContract.model_validate(document).sweep


def _settings(*, resume: bool = False, stop_after_children: int | None = None):
    return SweepExecutionSettings(
        child_execution=ExecutionSettings(
            write_plot=False,
            convergence_max_refinements=1,
        ),
        resume=resume,
        stop_after_children=stop_after_children,
    )


def test_resume_reuses_completed_children_and_keeps_hashes_and_parent_id(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sweep = _sweep([90.0, 100.0, 110.0])
    output = tmp_path / "resumable"
    first = execute_sweep(sweep, output, settings=_settings(stop_after_children=1))
    first_result = json.loads((output / "SWEEP_RESULT.json").read_text(encoding="utf-8"))
    first_completed = first_result["children"][0]
    first_run_id = first["run_id"]
    first_child_root = output / first_completed["artifact_root"]
    assert first_result["status"] == "interrupted"
    assert first_result["executed_child_count"] == 1
    assert first_result["pending_child_count"] == 2

    original_execute_task = sweep_module.execute_task
    calls: list[Path] = []

    def counted_execute_task(mode, task, output_dir, **kwargs):
        calls.append(Path(output_dir))
        return original_execute_task(mode, task, output_dir, **kwargs)

    monkeypatch.setattr(sweep_module, "execute_task", counted_execute_task)
    resumed = execute_sweep(sweep, output, settings=_settings(resume=True))
    resumed_result = json.loads((output / "SWEEP_RESULT.json").read_text(encoding="utf-8"))

    assert resumed["run_id"] == first_run_id
    assert resumed_result["status"] == "completed"
    assert resumed_result["ok"] is True
    assert resumed_result["resumed_child_count"] == 1
    assert resumed_result["executed_child_count"] == 2
    assert resumed_result["pending_child_count"] == 0
    assert len(calls) == 2
    assert first_completed["child_run_id"] == resumed_result["children"][0]["child_run_id"]
    assert first_completed["child_task_sha256"] == resumed_result["children"][0]["child_task_sha256"]
    assert first_child_root.is_dir()
    assert all(child["status"] == "completed" for child in resumed_result["children"])
    assert [child["child_task_sha256"] for child in resumed_result["children"]] == [
        row["child_task_sha256"] for row in expand_sweep(sweep)
    ]
    assert (output / "RUN_RESULT.json").is_file()
    assert (output / "SWEEP_TABLE.csv").is_file()
    assert (output / "NORMALIZED_TASK.json").is_file()
    for child in resumed_result["children"]:
        child_root = output / child["artifact_root"]
        assert (child_root / "RUN_RESULT.json").is_file()
        assert (child_root / "PHYSICS_ACCEPTANCE_CERTIFICATE.json").is_file()


def test_resume_rejects_a_changed_study_task_hash(tmp_path: Path) -> None:
    output = tmp_path / "changed"
    execute_sweep(_sweep([90.0, 100.0]), output, settings=_settings(stop_after_children=1))
    with pytest.raises(ValueError, match="task hash differs"):
        execute_sweep(_sweep([90.0, 101.0]), output, settings=_settings(resume=True))

