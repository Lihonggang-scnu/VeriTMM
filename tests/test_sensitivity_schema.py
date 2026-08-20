from __future__ import annotations

import json

import pytest
from jsonschema import Draft202012Validator

from tmm_engine.protocol import SensitivityTaskContract, export_schema


def _document(**overrides: object) -> dict[str, object]:
    sensitivity: dict[str, object] = {
        "simulation": {
            "stack": {
                "layers": [
                    {"constant_n": 2.0, "thickness_nm": 100.0, "optimizable": True},
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
        "metric": {
            "name": "mean_R",
            "observable": "R",
            "wavelength_min_nm": 500.0,
            "wavelength_max_nm": 600.0,
            "aggregation": "mean",
        },
        "parameters": "optimizable_thicknesses",
        "finite_difference_step_nm": 0.01,
        "relative_error_tolerance": 1e-3,
        "absolute_error_tolerance": 1e-7,
    }
    sensitivity.update(overrides)
    return {
        "schema_version": "sensitivity-task-v1",
        "mode": "sensitivity",
        "sensitivity": sensitivity,
    }


def test_sensitivity_contract_round_trip_and_json_schema() -> None:
    document = _document()
    contract = SensitivityTaskContract.model_validate(document)
    restored = SensitivityTaskContract.model_validate(contract.model_dump(mode="json"))
    assert restored == contract
    assert restored.model_dump(mode="json") == contract.model_dump(mode="json")

    schema = export_schema("sensitivity")
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(document)
    json.dumps(schema, ensure_ascii=False)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("parameters", "all_thicknesses", "optimizable_thicknesses"),
        ("finite_difference_step_nm", 0.0, "greater than 0"),
        ("relative_error_tolerance", 0.0, "greater than 0"),
        ("absolute_error_tolerance", 0.0, "greater than 0"),
    ],
)
def test_sensitivity_contract_rejects_invalid_numeric_and_parameter_contracts(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SensitivityTaskContract.model_validate(_document(**{field: value}))


def test_sensitivity_contract_forbids_unknown_fields() -> None:
    document = _document()
    sensitivity = document["sensitivity"]
    assert isinstance(sensitivity, dict)
    sensitivity["research_override"] = True
    with pytest.raises(ValueError, match="extra_forbidden"):
        SensitivityTaskContract.model_validate(document)

