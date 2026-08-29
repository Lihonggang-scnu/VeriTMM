from __future__ import annotations

import json

import pytest
from jsonschema import Draft202012Validator

from tmm_engine.protocol import SweepTaskContract, export_schema


def _document(
    *,
    parameters: list[dict[str, object]] | None = None,
    metrics: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
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
                    "angles_deg": [0.0, 30.0],
                    "polarizations": ["unpolarized"],
                },
            },
            "parameters": parameters
            or [{"path": "/stack/layers/0/thickness_nm", "values": [90, 100]}],
            "metrics": metrics
            or [
                {
                    "name": "mean_R",
                    "observable": "R",
                    "wavelength_min_nm": 500.0,
                    "wavelength_max_nm": 600.0,
                    "aggregation": "mean",
                    "angle_deg": 0.0,
                    "polarization": "unpolarized",
                }
            ],
        },
    }


def test_sweep_contract_round_trips_and_exports_valid_json_schema() -> None:
    document = _document(
        parameters=[
            {"path": "/stack/layers/0/thickness_nm", "values": [90, 100]},
            {"path": "/illumination/angles_deg/1", "values": [20, 30]},
            {"path": "/spectrum/points", "values": [11, 21]},
        ],
        metrics=[
            {
                "name": "worst_R",
                "observable": "R",
                "aggregation": "worst_case",
                "threshold_direction": "at_least",
            }
        ],
    )
    contract = SweepTaskContract.model_validate(document)
    restored = SweepTaskContract.model_validate(contract.model_dump(mode="json"))
    assert restored == contract
    assert restored.model_dump(mode="json") == contract.model_dump(mode="json")

    schema = export_schema("sweep")
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(document)
    json.dumps(schema, ensure_ascii=False)


@pytest.mark.parametrize(
    "path",
    [
        "/stack/layers/0/material",
        "/stack/layers/0/constant_n",
        "/stack/layers/0/thickness_nm/extra",
        "/simulation/stack/layers/0/thickness_nm",
        "/illumination/polarizations/0",
        "/arbitrary/code",
    ],
)
def test_sweep_parameter_paths_are_fail_closed(path: str) -> None:
    with pytest.raises(ValueError, match="allow-list"):
        SweepTaskContract.model_validate(
            _document(parameters=[{"path": path, "values": [1]}])
        )


def test_sweep_rejects_duplicate_axes_and_metric_names() -> None:
    with pytest.raises(ValueError, match="parameter paths must be unique"):
        SweepTaskContract.model_validate(
            _document(
                parameters=[
                    {"path": "/stack/layers/0/thickness_nm", "values": [90]},
                    {"path": "/stack/layers/0/thickness_nm", "values": [100]},
                ]
            )
        )
    with pytest.raises(ValueError, match="metric names must be unique"):
        SweepTaskContract.model_validate(
            _document(
                metrics=[
                    {"name": "R", "observable": "R"},
                    {"name": "R", "observable": "T"},
                ]
            )
        )


@pytest.mark.parametrize(
    "metric",
    [
        {"name": "value", "observable": "R", "aggregation": "value_at_wavelength"},
        {
            "name": "band",
            "observable": "R",
            "aggregation": "threshold_band_width",
        },
    ],
)
def test_metric_contract_requires_parameters_for_special_aggregations(
    metric: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        SweepTaskContract.model_validate(_document(metrics=[metric]))


def test_sweep_contract_preserves_declared_unsupported_physics_for_preflight() -> None:
    document = _document()
    sweep = document["sweep"]
    assert isinstance(sweep, dict)
    base = sweep["base_simulation"]
    assert isinstance(base, dict)
    base["physics"] = {"geometry_class": "lateral_periodic"}
    contract = SweepTaskContract.model_validate(document)
    assert contract.sweep.base_simulation.physics.geometry_class == "lateral_periodic"

