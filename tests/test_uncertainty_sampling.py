from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tmm_engine.protocol import ToleranceTaskContract
from tmm_engine.scientific_analysis import execute_tolerance


def _request(
    *,
    distribution: str,
    seed: int = 17,
    nominal_thickness_nm: float = 100.0,
    spread_nm: float = 2.0,
):
    uncertainty = (
        {"layer_index": 0, "distribution": "normal", "sigma_nm": spread_nm}
        if distribution == "normal"
        else {
            "layer_index": 0,
            "distribution": "uniform",
            "half_width_nm": spread_nm,
        }
    )
    document = {
        "schema_version": "tolerance-task-v1",
        "mode": "tolerance",
        "tolerance": {
            "simulation": {
                "stack": {
                    "layers": [
                        {
                            "constant_n": 2.0,
                            "thickness_nm": nominal_thickness_nm,
                            "optimizable": False,
                        },
                        {"constant_n": 1.4, "thickness_nm": 80.0, "optimizable": False},
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
            "uncertainties": [uncertainty],
            "metric": {
                "name": "mean_T",
                "observable": "T",
                "aggregation": "mean",
            },
            "target": {
                "metric": {
                    "name": "mean_T",
                    "observable": "T",
                    "aggregation": "mean",
                },
                "constraint": "at_least",
                "value": 0.0,
            },
            "sample_count": 12,
            "seed": seed,
        },
    }
    return ToleranceTaskContract.model_validate(document).tolerance


def _sample_thicknesses(output: Path) -> list[float]:
    result = json.loads((output / "TOLERANCE_RESULT.json").read_text(encoding="utf-8"))
    return [sample["thicknesses_nm"][0] for sample in result["samples"]]


def test_seeded_normal_sampling_is_reproducible_and_keeps_fixed_layers(
    tmp_path: Path,
) -> None:
    first_output = tmp_path / "normal-a"
    second_output = tmp_path / "normal-b"
    request = _request(distribution="normal")
    execute_tolerance(request, first_output)
    execute_tolerance(request, second_output)

    first = _sample_thicknesses(first_output)
    second = _sample_thicknesses(second_output)
    expected = 100.0 + np.random.default_rng(17).normal(0.0, 2.0, size=12)
    np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(first, expected, rtol=0.0, atol=1e-12)
    result = json.loads((first_output / "TOLERANCE_RESULT.json").read_text(encoding="utf-8"))
    assert all(sample["thicknesses_nm"][1] == 80.0 for sample in result["samples"])
    assert result["uncertainties"][0]["distribution"] == "normal"


def test_seeded_uniform_sampling_matches_rng_and_differs_from_normal(
    tmp_path: Path,
) -> None:
    uniform_output = tmp_path / "uniform"
    normal_output = tmp_path / "normal"
    execute_tolerance(_request(distribution="uniform"), uniform_output)
    execute_tolerance(_request(distribution="normal"), normal_output)

    uniform = np.asarray(_sample_thicknesses(uniform_output))
    normal = np.asarray(_sample_thicknesses(normal_output))
    expected = 100.0 + np.random.default_rng(17).uniform(-2.0, 2.0, size=12)
    np.testing.assert_allclose(uniform, expected, rtol=0.0, atol=1e-12)
    assert np.all(uniform >= 98.0)
    assert np.all(uniform <= 102.0)
    assert not np.array_equal(uniform, normal)
    result = json.loads((uniform_output / "TOLERANCE_RESULT.json").read_text(encoding="utf-8"))
    assert result["uncertainties"][0]["distribution"] == "uniform"


def test_different_seed_changes_sampled_draws(tmp_path: Path) -> None:
    first_output = tmp_path / "seed-a"
    second_output = tmp_path / "seed-b"
    execute_tolerance(_request(distribution="normal", seed=17), first_output)
    execute_tolerance(_request(distribution="normal", seed=18), second_output)
    assert _sample_thicknesses(first_output) != _sample_thicknesses(second_output)


def test_truncate_boundary_policy_prevents_silent_negative_thickness(
    tmp_path: Path,
) -> None:
    output = tmp_path / "bounded"
    execute_tolerance(
        _request(
            distribution="uniform",
            nominal_thickness_nm=0.2,
            spread_nm=2.0,
        ),
        output,
    )
    result = json.loads((output / "TOLERANCE_RESULT.json").read_text(encoding="utf-8"))
    assert result["uncertainty_model"] == {
        "boundary_policy": "truncate",
        "min_thickness_physical_nm": 0.1,
        "seed": 17,
    }
    assert any(sample["raw_thicknesses_nm"][0] < 0.1 for sample in result["samples"])
    assert all(sample["thicknesses_nm"][0] >= 0.1 for sample in result["samples"])
    assert any(sample["boundary_adjusted"] for sample in result["samples"])
    assert result["failed_sample_count"] == 0
