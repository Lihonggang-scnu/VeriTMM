from __future__ import annotations

import json

import pytest
from jsonschema import Draft202012Validator

from tmm_engine.protocol import ToleranceTaskContract, export_schema


def _metric(name: str = "mean_T") -> dict[str, object]:
    return {
        "name": name,
        "observable": "T",
        "wavelength_min_nm": 500.0,
        "wavelength_max_nm": 600.0,
        "aggregation": "mean",
        "angle_deg": 0.0,
        "polarization": "unpolarized",
    }


def _document(
    *,
    uncertainties: list[dict[str, object]] | None = None,
    metric: dict[str, object] | None = None,
    target_metric: dict[str, object] | None = None,
) -> dict[str, object]:
    selected_metric = metric or _metric()
    return {
        "schema_version": "tolerance-task-v1",
        "mode": "tolerance",
        "tolerance": {
            "simulation": {
                "stack": {
                    "layers": [
                        {"constant_n": 2.0, "thickness_nm": 100.0, "optimizable": False},
                        {"constant_n": 1.4, "thickness_nm": 80.0, "optimizable": False},
                    ],
                    "incident": {"constant_n": 1.0},
                    "exit": {"constant_n": 1.5},
                },
                "spectrum": {"start_nm": 500.0, "stop_nm": 600.0, "points": 21},
                "illumination": {
                    "angles_deg": [0.0],
                    "polarizations": ["unpolarized"],
                },
                "solver": "smatrix",
                "requested_outputs": ["R", "T", "A"],
            },
            "uncertainties": uncertainties
            or [
                {"layer_index": 0, "distribution": "normal", "sigma_nm": 2.0},
                {"layer_index": 1, "distribution": "uniform", "half_width_nm": 1.5},
            ],
            "metric": selected_metric,
            "target": {
                "metric": target_metric or selected_metric,
                "constraint": "at_least",
                "value": 0.9,
            },
            "sample_count": 16,
            "seed": 42,
        },
    }


def test_tolerance_contract_round_trip_and_schema() -> None:
    document = _document()
    contract = ToleranceTaskContract.model_validate(document)
    restored = ToleranceTaskContract.model_validate(contract.model_dump(mode="json"))
    assert restored == contract
    assert [item.distribution for item in contract.tolerance.uncertainties] == [
        "normal",
        "uniform",
    ]
    schema = export_schema("tolerance")
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(document)
    json.dumps(schema, ensure_ascii=False)


@pytest.mark.parametrize(
    "uncertainty",
    [
        {"layer_index": 0, "distribution": "normal"},
        {"layer_index": 0, "distribution": "uniform"},
        {
            "layer_index": 0,
            "distribution": "normal",
            "sigma_nm": 2.0,
            "half_width_nm": 1.0,
        },
        {
            "layer_index": 0,
            "distribution": "uniform",
            "half_width_nm": 1.0,
            "sigma_nm": 2.0,
        },
    ],
)
def test_uncertainty_distribution_parameters_are_explicit_and_exclusive(
    uncertainty: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="uncertainty"):
        ToleranceTaskContract.model_validate(_document(uncertainties=[uncertainty]))


def test_tolerance_contract_rejects_duplicate_layers_and_mismatched_target_metric() -> None:
    duplicate = [
        {"layer_index": 0, "distribution": "normal", "sigma_nm": 2.0},
        {"layer_index": 0, "distribution": "uniform", "half_width_nm": 1.0},
    ]
    with pytest.raises(ValueError, match="layer_index values must be unique"):
        ToleranceTaskContract.model_validate(_document(uncertainties=duplicate))

    with pytest.raises(ValueError, match="target.metric must equal"):
        ToleranceTaskContract.model_validate(
            _document(target_metric=_metric("different_metric"))
        )


def test_tolerance_contract_rejects_invalid_sample_count_and_unknown_field() -> None:
    document = _document()
    tolerance = document["tolerance"]
    assert isinstance(tolerance, dict)
    tolerance["sample_count"] = 0
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        ToleranceTaskContract.model_validate(document)

    document = _document()
    tolerance = document["tolerance"]
    assert isinstance(tolerance, dict)
    tolerance["certainty_score"] = 0.99
    with pytest.raises(ValueError, match="extra_forbidden"):
        ToleranceTaskContract.model_validate(document)

