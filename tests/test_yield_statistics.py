from __future__ import annotations

import json
from pathlib import Path

import pytest

from tmm_engine import scientific_analysis
from tmm_engine.capabilities import PhysicsEngineError
from tmm_engine.protocol import ToleranceTaskContract
from tmm_engine.scientific_analysis import (
    _evaluate_tolerance_sample,
    execute_tolerance,
    wilson_interval,
)
from tmm_engine.uncertainty import yield_accounting


def _request(sample_count: int = 12):
    metric = {
        "name": "mean_T",
        "observable": "T",
        "aggregation": "mean",
    }
    document = {
        "schema_version": "tolerance-task-v1",
        "mode": "tolerance",
        "tolerance": {
            "simulation": {
                "stack": {
                    "layers": [{"constant_n": 2.0, "thickness_nm": 100.0, "optimizable": False}],
                    "incident": {"constant_n": 1.0},
                    "exit": {"constant_n": 1.5},
                },
                "spectrum": {"start_nm": 500.0, "stop_nm": 600.0, "points": 11},
                "illumination": {
                    "angles_deg": [0.0],
                    "polarizations": ["unpolarized"],
                },
                "solver": "smatrix",
                "requested_outputs": ["R", "T", "A"],
            },
            "uncertainties": [
                {"layer_index": 0, "distribution": "normal", "sigma_nm": 1.0}
            ],
            "metric": metric,
            "target": {"metric": metric, "constraint": "at_least", "value": 0.0},
            "sample_count": sample_count,
            "seed": 3,
        },
    }
    return ToleranceTaskContract.model_validate(document).tolerance


def test_wilson_score_interval_matches_standard_binomial_formula() -> None:
    interval = wilson_interval(5, 10)
    assert interval == pytest.approx([0.236593090512564, 0.763406909487436], rel=1e-12)
    assert wilson_interval(0, 10)[0] == 0.0
    assert wilson_interval(10, 10)[1] == pytest.approx(1.0)

    with pytest.raises(ValueError):
        wilson_interval(-1, 10)
    with pytest.raises(ValueError):
        wilson_interval(11, 10)
    with pytest.raises(ValueError):
        wilson_interval(0, 0)


def test_tolerance_yield_uses_completed_trials_and_reports_overall_fraction(
    tmp_path: Path,
) -> None:
    output = tmp_path / "yield"
    execute_tolerance(_request(), output)
    result = json.loads((output / "TOLERANCE_RESULT.json").read_text(encoding="utf-8"))
    completed = [sample for sample in result["samples"] if sample["status"] == "completed"]
    successes = sum(bool(sample["target_passed"]) for sample in completed)

    assert result["sample_count"] == 12
    assert result["completed_sample_count"] == len(completed)
    assert result["failed_sample_count"] == 0
    assert result["requested_sample_count"] == 12
    assert result["conditional_yield"] == pytest.approx(successes / len(completed))
    assert result["overall_success_fraction"] == pytest.approx(successes / 12.0)
    assert result["yield"] == result["conditional_yield"]
    assert result["target_pass_probability"] == result["yield"]
    assert result["yield_ci_method"] == "wilson_score_interval"
    assert result["conditional_yield_ci_denominator"] == len(completed)
    assert result["yield_ci95"] == pytest.approx(
        wilson_interval(successes, len(completed))
    )
    for name in ("mean", "std", "p01", "p05", "p50", "p95", "p99", "worst_case"):
        assert name in result["statistics"]


def test_controlled_failures_separate_conditional_yield_from_overall_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def controlled_sample(*_args):
        nonlocal calls
        index = calls
        calls += 1
        if index >= 8:
            raise FloatingPointError("controlled numerical failure")
        return (0.9, True) if index < 6 else (0.1, False)

    monkeypatch.setattr(scientific_analysis, "_evaluate_tolerance_sample", controlled_sample)
    output = tmp_path / "controlled"
    envelope = execute_tolerance(_request(sample_count=10), output)
    result = json.loads((output / "TOLERANCE_RESULT.json").read_text(encoding="utf-8"))

    assert envelope["ok"] is True
    assert result["requested_sample_count"] == 10
    assert result["completed_sample_count"] == 8
    assert result["failed_sample_count"] == 2
    assert result["target_pass_count"] == 6
    assert result["conditional_yield"] == pytest.approx(6 / 8)
    assert result["overall_success_fraction"] == pytest.approx(6 / 10)
    assert result["conditional_yield_ci95"] == pytest.approx(wilson_interval(6, 8))
    assert result["conditional_yield_ci_denominator"] == 8
    assert result["failure_taxonomy"]["numerical_failure"] == 2


def test_all_failed_samples_have_null_conditional_yield(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_sample(*_args):
        raise RuntimeError("controlled runtime failure")

    monkeypatch.setattr(scientific_analysis, "_evaluate_tolerance_sample", fail_sample)
    output = tmp_path / "all-failed"
    envelope = execute_tolerance(_request(sample_count=10), output)
    result = json.loads((output / "TOLERANCE_RESULT.json").read_text(encoding="utf-8"))

    assert envelope["ok"] is False
    assert result["status"] == "insufficient_valid_samples"
    assert result["completed_sample_count"] == 0
    assert result["failed_sample_count"] == 10
    assert result["conditional_yield"] is None
    assert result["conditional_yield_ci95"] is None
    assert result["yield"] is None
    assert result["overall_success_fraction"] == 0.0
    assert all(value is None for value in result["statistics"].values())


def test_yield_accounting_handles_zero_passes_with_all_samples_completed() -> None:
    accounting = yield_accounting(0, 10, 10)
    assert accounting["conditional_yield"] == 0.0
    assert accounting["overall_success_fraction"] == 0.0
    assert accounting["conditional_yield_ci95"] == wilson_interval(0, 10)


def test_tolerance_rejects_solver_audit_failure_before_counting_completion() -> None:
    class _InvalidForward:
        audit = {
            "nonfinite_value_count": 1,
            "passivity_check_passed": False,
            "energy_conservation_max_abs_error": 0.0,
        }

    class _Workbench:
        def simulate(self, _simulation):
            return _InvalidForward()

    with pytest.raises(PhysicsEngineError) as captured:
        _evaluate_tolerance_sample(_Workbench(), object(), _request())  # type: ignore[arg-type]
    assert captured.value.failure.code.value == "numerical_nonfinite"
