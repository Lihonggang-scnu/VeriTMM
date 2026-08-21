from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tmm_engine.agent_bench import (
    BenchmarkCase,
    default_benchmark_cases_dir,
    load_benchmark_cases,
    run_offline_benchmark,
)


def _simulation(*, geometry: str = "layered_planar") -> dict:
    return {
        "mode": "simulate",
        "simulation": {
            "stack": {
                "incident": {"constant_n": 1.0},
                "layers": [{"constant_n": 1.45, "thickness_nm": 100.0}],
                "exit": {"constant_n": 1.5},
            },
            "spectrum": {"start_nm": 500.0, "stop_nm": 600.0, "points": 21},
            "illumination": {"angles_deg": [0.0], "polarizations": ["unpolarized"]},
            "requested_outputs": ["R", "T", "A"],
            "physics": {"geometry_class": geometry},
        },
    }


def _valid_case() -> dict:
    return {
        "case_id": "valid_single_film",
        "category": "valid_basic",
        "natural_language_task": "Simulate one isotropic film at normal incidence.",
        "task": _simulation(),
        "expected_mode": "simulate",
        "expected_capability": "supported",
        "expected_failure_codes": [],
        "expected_artifacts": ["physics_certificate", "result_summary", "spectrum_table"],
        "physics_assertions": [
            {
                "source": "summary",
                "path": "physics.accepted",
                "operator": "eq",
                "expected": True,
            }
        ],
        "difficulty": "basic",
        "tags": ["single_film"],
        "execution": "run",
        "reproducibility_runs": 2,
    }


def _invalid_case() -> dict:
    return {
        "case_id": "invalid_grating",
        "category": "invalid_out_of_scope",
        "natural_language_task": "Simulate diffraction orders from a lateral grating.",
        "task": _simulation(geometry="lateral_periodic"),
        "expected_mode": "simulate",
        "expected_capability": "unsupported",
        "expected_failure_codes": ["unsupported_geometry"],
        "expected_artifacts": [],
        "physics_assertions": [],
        "difficulty": "adversarial",
        "tags": ["unsupported", "grating"],
        "execution": "preflight_only",
        "reproducibility_runs": 2,
    }


def _write_cases(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "valid.json").write_text(json.dumps(_valid_case()), encoding="utf-8")
    (root / "invalid.json").write_text(json.dumps(_invalid_case()), encoding="utf-8")


def test_case_contract_rejects_supported_case_with_failure_codes() -> None:
    payload = _valid_case()
    payload["expected_failure_codes"] = ["invalid_task"]
    with pytest.raises(ValueError):
        BenchmarkCase.model_validate(payload)


def test_loader_orders_cases_and_rejects_duplicate_ids(tmp_path: Path) -> None:
    _write_cases(tmp_path)
    cases = load_benchmark_cases(tmp_path)
    assert [case.case_id for case in cases] == ["invalid_grating", "valid_single_film"]
    (tmp_path / "duplicate.json").write_text(json.dumps(_valid_case()), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_benchmark_cases(tmp_path)


def test_offline_benchmark_is_reproducible_and_false_accepts_zero(tmp_path: Path) -> None:
    cases_dir = tmp_path / "cases"
    _write_cases(cases_dir)
    output = tmp_path / "BENCHMARK_RESULT.json"
    result = run_offline_benchmark(
        load_benchmark_cases(cases_dir),
        output_path=output,
        work_dir=tmp_path / "work",
        minimum_case_count=2,
    )
    assert result["release_gate_passed"] is True
    assert result["case_count"] == 2
    assert result["metrics"]["unsupported_false_accept_rate"] == 0.0
    assert result["metrics"]["reproducibility_rate"]["rate"] == 1.0
    assert result["metrics"]["artifact_completeness_rate"]["rate"] == 1.0
    assert len(result["case_catalog_sha256"]) == 64
    assert len(result["result_content_sha256"]) == 64
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "passed"


def test_cli_benchmark_stdout_is_one_json_object(tmp_path: Path) -> None:
    cases_dir = tmp_path / "cases"
    _write_cases(cases_dir)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tmm_engine.cli",
            "benchmark",
            "--offline",
            "--json",
            "--cases-dir",
            str(cases_dir),
            "--output",
            str(tmp_path / "result.json"),
            "--work-dir",
            str(tmp_path / "work"),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert completed.returncode == 3, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["release_gate_passed"] is False
    assert payload["case_count_requirement_passed"] is False
    assert completed.stdout.count("\n") == 1


def test_packaged_catalogue_is_release_scale_and_diverse() -> None:
    cases = load_benchmark_cases(default_benchmark_cases_dir())
    assert len(cases) >= 80
    assert len({case.case_id for case in cases}) == len(cases)
    assert len({case.natural_language_task for case in cases}) == len(cases)
    assert {case.expected_mode for case in cases} == {
        "simulate",
        "optimize",
        "sweep",
        "sensitivity",
        "tolerance",
    }
    assert sum(case.expected_capability == "unsupported" for case in cases) >= 15
    assert sum(case.execution == "run" for case in cases) >= 15
    assert {case.scenario for case in cases} >= {
        "standard",
        "cache_replay",
        "sweep_resume",
    }


@pytest.mark.parametrize("case_id", ["sweep_cache_replay", "sweep_resume"])
def test_runtime_scenarios_prove_cache_and_resume(case_id: str, tmp_path: Path) -> None:
    selected = [
        case
        for case in load_benchmark_cases(default_benchmark_cases_dir())
        if case.case_id == case_id
    ]
    assert len(selected) == 1
    artifact_path = tmp_path / f"{case_id}_BENCHMARK_RESULT.json"
    result = run_offline_benchmark(
        selected,
        output_path=artifact_path,
        work_dir=tmp_path / case_id,
        minimum_case_count=1,
    )
    assert result["failed_case_count"] == 0
    assert "cases" not in result
    detailed = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert detailed["cases"][0]["passed"] is True
    assert all(item["passed"] for item in detailed["cases"][0]["assertions"])
