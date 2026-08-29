from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from tmm_engine.execution import ExecutionSettings, execute_task
from tmm_engine.optimization import OptimizationResult
from tmm_engine.protocol import PreflightReport, RunResultEnvelope
from tmm_engine.run_artifacts import (
    file_sha256,
    prepare_output_directory,
    stable_payload_sha256,
    write_json,
    write_run_result,
)
from tmm_engine.schemas import dataclass_to_dict
from tmm_engine.task_io import load_task

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "tmm_tasks" / "periodic_dbr_simulation.json"


def test_successful_run_has_first_read_envelope_summary_and_correct_hashes(tmp_path: Path) -> None:
    mode, task = load_task(EXAMPLE)
    payload = execute_task(
        mode,
        task,
        tmp_path,
        input_path=EXAMPLE,
        settings=ExecutionSettings(write_plot=False),
    )
    assert payload["ok"] is True
    assert payload["archive_schema_version"] == 2
    RunResultEnvelope.model_validate(payload)
    assert payload["status"] == "completed"
    expected_task_hash = stable_payload_sha256(
        {"mode": mode, "simulation": dataclass_to_dict(task)}
    )
    assert payload["task_sha256"] == expected_task_hash
    kinds = {item["kind"] for item in payload["artifacts"]}
    assert {"normalized_task", "result_summary", "physics_certificate", "spectrum_table"} <= kinds
    assert "RUN_RESULT.json" not in {item["path"] for item in payload["artifacts"]}
    for item in payload["artifacts"]:
        assert item["sha256"] == file_sha256(tmp_path / item["path"])
    summary = json.loads((tmp_path / "RESULT_SUMMARY.json").read_text(encoding="utf-8"))
    assert summary["physics"]["accepted"] is True
    assert summary["archive_schema_version"] == 2
    assert summary["evidence_coverage"]["independent_solver"] == "verified"
    assert summary["physics"]["evidence_coverage"] == summary["evidence_coverage"]
    assert summary["spectral_features"]
    preflight = json.loads((tmp_path / "PREFLIGHT_REPORT.json").read_text(encoding="utf-8"))
    PreflightReport.model_validate(preflight)


def test_preflight_rejection_still_writes_machine_artifacts(tmp_path: Path) -> None:
    invalid = ROOT / "tests" / "fixtures" / "agent_invalid" / "unsupported_geometry.json"
    mode, task = load_task(invalid)
    payload = execute_task(mode, task, tmp_path, input_path=invalid)
    assert payload["ok"] is False
    RunResultEnvelope.model_validate(payload)
    assert payload["status"] == "preflight_rejected"
    assert (tmp_path / "RUN_RESULT.json").is_file()
    assert (tmp_path / "RESULT_SUMMARY.json").is_file()
    assert payload["next_machine_actions"]


def test_reusing_output_directory_does_not_index_stale_protocol_artifacts(
    tmp_path: Path,
) -> None:
    (tmp_path / "OPTIMIZATION_RESULT.json").write_text(
        '{"stale": true}', encoding="utf-8"
    )
    (tmp_path / "user_notes.txt").write_text("preserve me", encoding="utf-8")
    mode, task = load_task(EXAMPLE)
    payload = execute_task(
        mode,
        task,
        tmp_path,
        settings=ExecutionSettings(write_plot=False),
    )
    assert not (tmp_path / "OPTIMIZATION_RESULT.json").exists()
    assert (tmp_path / "user_notes.txt").read_text(encoding="utf-8") == "preserve me"
    assert "optimization_result" not in {item["kind"] for item in payload["artifacts"]}


def test_candidate_artifacts_are_indexed_and_owned_candidate_tree_is_cleaned(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidates" / "candidate_01" / "SIMULATION_RESULT.json"
    write_json(candidate, {"candidate": 1})
    (tmp_path / "user_subdir").mkdir()
    (tmp_path / "user_subdir" / "notes.txt").write_text("keep", encoding="utf-8")
    payload = write_run_result(
        tmp_path,
        operation="optimize",
        task_sha256="a" * 64,
        status="completed",
        ok=True,
    )
    entry = next(
        item for item in payload["artifacts"] if item["path"].startswith("candidates/")
    )
    assert entry["kind"] == "candidate_simulation_result"
    assert entry["sha256"] == file_sha256(candidate)

    prepare_output_directory(tmp_path)
    assert not (tmp_path / "candidates").exists()
    assert (tmp_path / "user_subdir" / "notes.txt").read_text(encoding="utf-8") == "keep"


def test_portfolio_requires_independent_validation_and_preserves_zero_loss(
    tmp_path: Path, monkeypatch
) -> None:
    from tmm_engine import execution
    from tmm_engine.acceptance import AcceptanceSettings

    mode, task = load_task(
        ROOT / "examples" / "tmm_tasks" / "antireflection_optimization.json"
    )
    assert mode == "optimize"
    thickness_count = len(task.simulation.stack.layers)
    zero = [100.0] * thickness_count
    worse = [110.0] * thickness_count
    unvalidated = [120.0] * thickness_count
    optimization = OptimizationResult(
        status="completed",
        initial_thicknesses_nm=zero,
        optimized_thicknesses_nm=zero,
        quantized_thicknesses_nm=None,
        initial_loss=1.0,
        optimized_loss=0.0,
        quantized_loss=None,
        steps_executed=1,
        best_start_index=0,
        stop_reason="test",
        wall_seconds=0.0,
        candidate_designs=[
            {"candidate_id": "zero", "source": "optimized_best", "thicknesses_nm": zero, "objective_loss": 0.0},
            {"candidate_id": "worse", "source": "initial", "thicknesses_nm": worse, "objective_loss": 1.0},
            {"candidate_id": "invalid", "source": "quantized_best", "thicknesses_nm": unvalidated, "objective_loss": -1.0},
        ],
    )

    class FakeOptimizer:
        def validate_result(self, _task, candidate, _workbench):
            status = (
                "failed"
                if candidate.optimized_thicknesses_nm == unvalidated
                else "passed"
            )
            return _task.simulation, None, {"status": status, "target_attainment": {}}

        def evaluate(self, _task, values):
            return float(sum(values)) / 1e6, {}

    monkeypatch.setattr(
        execution,
        "certify_simulation",
        lambda *_args, **_kwargs: SimpleNamespace(
            result=None,
            certificate={
                "accepted": True,
                "status": "physically_valid",
                "certificate_id": "cert",
            },
        ),
    )
    payload = execution._write_optimization_portfolio(
        tmp_path,
        task,
        optimization,
        FakeOptimizer(),
        SimpleNamespace(),
        AcceptanceSettings(),
        3,
    )
    assert payload["selected_roles"]["best_performance"] == "zero"
    assert payload["selected_roles"]["most_robust"] is None
    assert payload["selected_roles"]["best_heuristic_robustness"] in {"zero", "worse"}
    invalid = next(item for item in payload["candidates"] if item["candidate_id"] == "invalid")
    assert invalid["independent_validation_status"] == "failed"
    assert payload["selected_roles"]["best_performance"] != "invalid"
