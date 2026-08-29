"""Schema tests for experimental FitTask contracts."""

from __future__ import annotations

import json

from tmm_engine.fitting.fit_task import (
    FitParameter,
    FitTask,
    IdentifiabilityReport,
    MeasuredDataPoint,
    MeasurementType,
)


def test_fit_parameter_schema() -> None:
    parameter = FitParameter(
        name="thickness_layer_0",
        layer_index=0,
        bounds=(10.0, 200.0),
        initial_guess=50.0,
    )
    assert parameter.name == "thickness_layer_0"
    assert parameter.bounds == (10.0, 200.0)


def test_measured_data_point_schema_accepts_all_measurement_types() -> None:
    for measurement_type in MeasurementType:
        measurement = MeasuredDataPoint(
            wavelength_nm=550.0,
            measurement_type=measurement_type,
            value=0.25,
        )
        assert measurement.measurement_type == measurement_type


def test_identifiability_report_contains_rank_and_correlation() -> None:
    report = IdentifiabilityReport(
        rmse=0.005,
        degrees_of_freedom=10,
        jacobian_condition_number=50.0,
        singular_values=[1.0, 0.5, 0.02],
        effective_rank=2,
        parameter_correlation_matrix=[[1.0, 0.95], [0.95, 1.0]],
        identifiability_status="weakly_identifiable",
    )
    assert report.identifiability_status == "weakly_identifiable"
    assert len(report.parameter_correlation_matrix) == 2


def test_fit_task_json_roundtrip() -> None:
    task = FitTask(
        structure={"materials": ["SiO2"], "thicknesses_nm": [100.0], "substrate": "Si"},
        measurements=[
            MeasuredDataPoint(
                wavelength_nm=550.0,
                measurement_type=MeasurementType.REFLECTANCE,
                value=0.3,
            )
        ],
        fit_parameters=[FitParameter(name="thickness_0", bounds=(50.0, 150.0))],
    )
    payload = task.model_dump(mode="json")
    restored = FitTask.model_validate(json.loads(json.dumps(payload)))
    assert restored == task


def test_fit_task_rejects_duplicate_parameter_names() -> None:
    try:
        FitTask(
            structure={"layers": [{"constant_n": 2.0, "thickness_nm": 100.0}]},
            measurements=[
                MeasuredDataPoint(
                    wavelength_nm=550.0,
                    measurement_type=MeasurementType.REFLECTANCE,
                    value=0.3,
                )
            ],
            fit_parameters=[
                FitParameter(name="thickness_0", bounds=(50.0, 150.0)),
                FitParameter(name="thickness_0", bounds=(50.0, 150.0)),
            ],
        )
    except ValueError as exc:
        assert "unique" in str(exc)
    else:  # pragma: no cover - assertion branch documents the contract
        raise AssertionError("duplicate fit parameter names must be rejected")
