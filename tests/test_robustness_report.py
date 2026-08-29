from __future__ import annotations

import json
from pathlib import Path

import pytest

from tmm_engine import scientific_analysis
from tmm_engine.protocol import RunResultEnvelope, ToleranceTaskContract
from tmm_engine.scientific_analysis import execute_tolerance, wilson_interval


def _request():
    metric = {"name": "mean_R", "observable": "R", "aggregation": "mean"}
    document = {
        "schema_version": "tolerance-task-v1",
        "mode": "tolerance",
        "tolerance": {
            "simulation": {
                "stack": {
                    "layers": [
                        {"constant_n": 2.0, "thickness_nm": 10.0, "optimizable": False}
                    ],
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
                {"layer_index": 0, "distribution": "uniform", "half_width_nm": 20.0}
            ],
            "metric": metric,
            "target": {
                "metric": metric,
                "constraint": "at_least",
                "value": 1.0,
            },
            "sample_count": 24,
            "seed": 123,
        },
    }
    return ToleranceTaskContract.model_validate(document).tolerance


def test_robustness_report_is_separate_from_nominal_physics_certificate_and_keeps_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    original = scientific_analysis._evaluate_tolerance_sample

    def fail_periodically(*args):
        nonlocal calls
        calls += 1
        if calls % 4 == 0:
            raise FloatingPointError("controlled numerical failure")
        return original(*args)

    monkeypatch.setattr(
        scientific_analysis, "_evaluate_tolerance_sample", fail_periodically
    )
    output = tmp_path / "robustness"
    envelope = execute_tolerance(_request(), output)
    certificate = json.loads(
        (output / "PHYSICS_ACCEPTANCE_CERTIFICATE.json").read_text(encoding="utf-8")
    )
    tolerance = json.loads((output / "TOLERANCE_RESULT.json").read_text(encoding="utf-8"))
    report = json.loads((output / "ROBUSTNESS_REPORT.json").read_text(encoding="utf-8"))

    assert envelope["ok"] is True
    RunResultEnvelope.model_validate(envelope)
    assert {item["kind"] for item in envelope["artifacts"]} >= {
        "physics_certificate",
        "tolerance_result",
        "robustness_report",
    }
    assert certificate["accepted"] is True
    assert certificate["uncertainty_budget"]["parameter_components"]
    assert certificate["uncertainty_budget"]["sampling_components"]
    assert certificate["evidence_coverage"]["uncertainty_quantified"] == "verified"
    assert report["physics_validity_is_separate"] is True
    assert report["nominal_physics_accepted"] is True
    assert report["nominal_physics_certificate_id"] == certificate["certificate_id"]
    assert report["status"] == "evaluated"
    assert tolerance["failed_sample_count"] > 0
    assert tolerance["completed_sample_count"] + tolerance["failed_sample_count"] == 24
    failed = [sample for sample in tolerance["samples"] if sample["status"] == "failed"]
    assert failed
    assert all(sample["target_passed"] is None for sample in failed)
    assert all(sample["failure_category"] == "numerical_failure" for sample in failed)
    assert tolerance["yield"] == 0.0
    assert tolerance["yield_ci95"] == wilson_interval(
        0, tolerance["completed_sample_count"]
    )
    assert tolerance["overall_success_fraction"] == 0.0
    assert report["yield"] == tolerance["yield"]
    assert report["yield_ci95"] == tolerance["yield_ci95"]
    certificate_text = json.dumps(certificate, sort_keys=True)
    assert "yield_ci95" not in certificate_text
    assert "failed_sample_count" not in certificate_text
    assert report["nominal_physics_certificate_id"] != report.get("certificate_id")
