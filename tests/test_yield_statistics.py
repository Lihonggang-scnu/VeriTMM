from __future__ import annotations

import json
from pathlib import Path

import pytest

from tmm_engine.protocol import ToleranceTaskContract
from tmm_engine.scientific_analysis import execute_tolerance, wilson_interval


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


def test_tolerance_yield_uses_all_trials_and_reports_ci(tmp_path: Path) -> None:
    output = tmp_path / "yield"
    execute_tolerance(_request(), output)
    result = json.loads((output / "TOLERANCE_RESULT.json").read_text(encoding="utf-8"))
    completed = [sample for sample in result["samples"] if sample["status"] == "completed"]
    successes = sum(bool(sample["target_passed"]) for sample in completed)

    assert result["sample_count"] == 12
    assert result["completed_sample_count"] == len(completed)
    assert result["failed_sample_count"] == 0
    assert result["yield"] == pytest.approx(successes / 12.0)
    assert result["target_pass_probability"] == result["yield"]
    assert result["yield_ci_method"] == "wilson_score_interval"
    assert result["yield_ci95"] == pytest.approx(wilson_interval(successes, 12))
    for name in ("mean", "std", "p01", "p05", "p50", "p95", "p99", "worst_case"):
        assert name in result["statistics"]
