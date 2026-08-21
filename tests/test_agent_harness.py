from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tmm_engine.agent_bench import BenchmarkCase
from tmm_engine.agent_harness import (
    AgentTrajectory,
    build_exposure,
    load_trajectories,
    run_agent_ab,
    score_trajectories,
)


def _case() -> BenchmarkCase:
    return BenchmarkCase.model_validate(
        {
            "case_id": "agent_case_001",
            "category": "valid_basic",
            "natural_language_task": "Simulate a planar film.",
            "task": {
                "mode": "simulate",
                "simulation": {
                    "stack": {
                        "incident": {"constant_n": 1.0},
                        "layers": [{"constant_n": 1.5, "thickness_nm": 100.0}],
                        "exit": {"constant_n": 1.0},
                    },
                    "spectrum": {"start_nm": 500.0, "stop_nm": 600.0, "points": 11},
                },
            },
            "expected_mode": "simulate",
            "expected_capability": "supported",
        }
    )


def test_agent_native_exposure_contains_protocol_but_traditional_does_not() -> None:
    traditional = build_exposure(_case(), "traditional")
    native = build_exposure(_case(), "agent_native")
    assert traditional["protocol_tools"] == []
    assert "preflight" in native["protocol_tools"]
    assert native["capabilities"]["geometry"] == ["layered_planar"]
    assert native["task_schema"]["$schema"].endswith("2020-12/schema")


def test_score_trajectories_preserves_unavailable_metrics_as_null() -> None:
    result = score_trajectories(
        [
            AgentTrajectory(
                benchmark_case="a",
                exposure="traditional",
                prompt="task",
                success=False,
                correction_turns=2,
            ),
            AgentTrajectory(
                benchmark_case="a",
                exposure="agent_native",
                prompt="task",
                success=True,
                first_task_valid=True,
                correction_turns=0,
                certificate_id="cert_1",
                physics_certificate_expected=True,
                unsupported_false_accept=False,
                reproducible=True,
            ),
        ]
    )
    assert result["response"]["profile"] == "compact"
    assert "trajectories" not in result
    assert result["arms"]["traditional"]["mean_input_tokens"] is None
    assert result["arms"]["agent_native"]["final_success_rate"] == 1.0
    assert result["arms"]["agent_native"]["certified_success_rate"] == 1.0
    assert result["arms"]["agent_native"]["unsupported_false_accept_rate"] == 0.0


def test_full_scorer_profile_is_richer_but_externalizes_trajectory_arrays() -> None:
    trajectory = AgentTrajectory(
        benchmark_case="a",
        exposure="agent_native",
        prompt="task",
        success=True,
        steps=[{"index": index, "action": "observe"} for index in range(40)],
    )
    standard = score_trajectories([trajectory], detail="standard")
    full = score_trajectories([trajectory], detail="full")

    assert standard["response"]["profile"] == "standard"
    assert full["response"]["profile"] == "full"
    assert "trajectories" not in standard
    assert "trajectories" not in full
    assert full["arms"] == standard["arms"]
    assert full["independent_audit_count"] == 1


def test_framework_neutral_runner_executes_both_arms() -> None:
    calls: list[str] = []

    def runner(case, exposure):
        calls.append(exposure["exposure"])
        native = exposure["exposure"] == "agent_native"
        return {
            "benchmark_case": case.case_id,
            "exposure": exposure["exposure"],
            "prompt": case.natural_language_task,
            "success": native,
            "first_task_valid": native,
            "correction_turns": 0 if native else 2,
            "tool_calls": [],
            "steps": [],
        }

    result = run_agent_ab([_case()], runner)
    assert result["response"]["profile"] == "compact"
    assert "trajectories" not in result
    assert calls == ["traditional", "agent_native"]
    assert result["arms"]["traditional"]["final_success_rate"] == 0.0
    assert result["arms"]["agent_native"]["final_success_rate"] == 1.0


def test_case_aware_scoring_overrides_false_self_report_from_task_evidence(
    tmp_path: Path,
) -> None:
    case = _case()
    trajectory = AgentTrajectory(
        benchmark_case=case.case_id,
        exposure="agent_native",
        prompt="task",
        success=False,
        first_task_valid=False,
        correction_turns=99,
        task_attempts=[case.task],
    )
    artifact = tmp_path / "agent_result.json"
    result = score_trajectories(
        [trajectory], cases={case.case_id: case}, output_path=artifact
    )
    assert result["response"]["profile"] == "compact"
    audited = json.loads(artifact.read_text(encoding="utf-8"))["trajectories"][0]
    assert audited["success"] is True
    assert audited["first_task_valid"] is True
    assert audited["correction_turns"] == 0
    provenance = json.loads(artifact.read_text(encoding="utf-8"))["independent_audit"][0][
        "metric_provenance"
    ]
    assert provenance["success"] == "recomputed_from_case_and_evidence"


def test_out_of_scope_case_counts_correct_rejection_as_success_not_certificate(
    tmp_path: Path,
) -> None:
    task = _case().task
    task["simulation"]["physics"] = {"geometry_class": "lateral_periodic"}
    case = BenchmarkCase.model_validate(
        {
            "case_id": "unsupported_grating",
            "category": "invalid_out_of_scope",
            "natural_language_task": "Compute diffraction orders from a grating.",
            "task": task,
            "expected_mode": "simulate",
            "expected_capability": "unsupported",
            "expected_failure_codes": ["unsupported_geometry"],
            "difficulty": "adversarial",
        }
    )
    trajectory = AgentTrajectory(
        benchmark_case=case.case_id,
        exposure="agent_native",
        prompt=case.natural_language_task,
        success=False,
        unsupported_false_accept=True,
        task_attempts=[task],
    )
    artifact = tmp_path / "agent_result.json"
    result = score_trajectories(
        [trajectory], cases={case.case_id: case}, output_path=artifact
    )
    audited = json.loads(artifact.read_text(encoding="utf-8"))["trajectories"][0]
    assert audited["success"] is True
    assert audited["unsupported_false_accept"] is False
    assert audited["physics_certificate_expected"] is False
    assert result["arms"]["agent_native"]["certified_success_rate"] is None


def test_agent_benchmark_cli_independently_audits_task_attempts(tmp_path: Path) -> None:
    case = _case()
    cases = tmp_path / "cases"
    cases.mkdir()
    (cases / "case.json").write_text(case.model_dump_json(indent=2), encoding="utf-8")
    trajectory = AgentTrajectory(
        benchmark_case=case.case_id,
        exposure="agent_native",
        prompt=case.natural_language_task,
        success=False,
        task_attempts=[case.task],
    )
    trajectory_path = tmp_path / "trajectories.json"
    trajectory_path.write_text(json.dumps([trajectory.model_dump(mode="json")]), encoding="utf-8")
    result_path = tmp_path / "agent_result.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tmm_engine.cli",
            "agent-benchmark",
            "--trajectories",
            str(trajectory_path),
            "--cases-dir",
            str(cases),
            "--json",
            "--output",
            str(result_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["response"]["profile"] == "compact"
    assert payload["trajectories_count"] == 1
    raw = json.loads(result_path.read_text(encoding="utf-8"))
    assert raw["trajectories"][0]["success"] is True
    assert raw["independent_audit"][0]["task_attempt_count"] == 1


def test_loader_accepts_one_documented_trajectory_object(tmp_path: Path) -> None:
    trajectory = AgentTrajectory(
        benchmark_case="case",
        exposure="agent_native",
        prompt="task",
        success=True,
    )
    path = tmp_path / "one.json"
    path.write_text(trajectory.model_dump_json(indent=2), encoding="utf-8")
    loaded = load_trajectories(path)
    assert len(loaded) == 1
    assert loaded[0].benchmark_case == "case"


@pytest.mark.parametrize("payload", [{}, {"trajectories": []}, []])
def test_loader_rejects_empty_or_ambiguous_input(tmp_path: Path, payload) -> None:
    path = tmp_path / "empty.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="trajector"):
        load_trajectories(path)


def test_scorer_rejects_empty_evidence() -> None:
    with pytest.raises(ValueError, match="at least one"):
        score_trajectories([])


def test_run_case_rejects_fabricated_or_mismatched_run_envelope(tmp_path: Path) -> None:
    case_payload = _case().model_dump(mode="json")
    case_payload["execution"] = "run"
    case_payload["expected_artifacts"] = ["physics_certificate"]
    case = BenchmarkCase.model_validate(case_payload)
    fabricated = {
        "schema_version": "veritmm-run-result-v1",
        "protocol_version": "veritmm-agent-v1",
        "ok": True,
        "run_id": "invented",
        "task_sha256": "0" * 64,
        "task_hash_scope": "normalized_operation_wrapper",
        "input_sha256": None,
        "operation": "simulate",
        "status": "completed",
        "summary": {
            "physics": {
                "accepted": True,
                "certificate_id": "invented-certificate",
            }
        },
        "warnings": [],
        "failures": [],
        "certificate_id": "invented-certificate",
        "artifacts": [],
        "next_machine_actions": [],
    }
    trajectory = AgentTrajectory(
        benchmark_case=case.case_id,
        exposure="agent_native",
        prompt=case.natural_language_task,
        success=True,
        task_attempts=[case.task],
        final_run_result=fabricated,
    )
    artifact = tmp_path / "agent_result.json"
    score_trajectories([trajectory], cases={case.case_id: case}, output_path=artifact)
    raw = json.loads(artifact.read_text(encoding="utf-8"))
    assert raw["trajectories"][0]["success"] is False
    evidence = raw["independent_audit"][0]["run_evidence"]
    assert evidence["schema_valid"] is True
    assert evidence["task_hash_match"] is False
    assert evidence["valid"] is False
