"""Integration tests for bounded fitting and identifiability analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tmm_engine.cli import main as cli_main
from tmm_engine.fitting.fit_task import (
    FitParameter,
    FitTask,
    MeasuredDataPoint,
    MeasurementType,
)
from tmm_engine.fitting.optimizer import (
    build_simulation_task,
    execute_forward_simulation,
    extract_simulation_value,
    fit_task,
)


def _synthetic_measurements(structure: dict, thickness: float) -> list[MeasuredDataPoint]:
    wavelengths = (500.0, 550.0, 600.0, 650.0)
    placeholders = [
        MeasuredDataPoint(
            wavelength_nm=wavelength,
            measurement_type=MeasurementType.REFLECTANCE,
            value=0.0,
        )
        for wavelength in wavelengths
    ]
    forward = execute_forward_simulation(
        build_simulation_task(
            structure,
            placeholders,
            {"thickness_layer_0": thickness},
        )
    )
    return [
        item.model_copy(
            update={"value": extract_simulation_value(forward, item)}
        )
        for item in placeholders
    ]


def test_synthetic_exact_recovery_and_fit_certificate_boundary() -> None:
    structure = {
        "layers": [{"constant_n": 2.0, "thickness_nm": 100.0}],
        "substrate": {"constant_n": 1.5},
    }
    task = FitTask(
        structure=structure,
        measurements=_synthetic_measurements(structure, 100.0),
        fit_parameters=[
            FitParameter(
                name="thickness_layer_0",
                layer_index=0,
                bounds=(80.0, 120.0),
                initial_guess=90.0,
            )
        ],
        tolerance=1e-10,
    )

    result = fit_task(task)

    assert result.converged
    assert result.best_fit_parameters["thickness_layer_0"] == pytest.approx(
        100.0, abs=1e-3
    )
    assert result.identifiability.identifiability_status == "well_determined"
    assert result.fit_certificate["physics_certificate"] is None
    assert result.fit_certificate["physics_validity"] == "not_certified"


def test_correlated_parameters_detected_as_non_identifiable() -> None:
    structure = {
        "layers": [
            {"constant_n": 2.0, "thickness_nm": 100.0},
            {"constant_n": 1.8, "thickness_nm": 80.0},
        ],
        "substrate": {"constant_n": 1.5},
    }
    measurements = _synthetic_measurements(
        {"layers": [structure["layers"][0]], "substrate": structure["substrate"]},
        100.0,
    )[:1]
    task = FitTask(
        structure=structure,
        measurements=measurements,
        fit_parameters=[
            FitParameter(name="thickness_layer_0", bounds=(80.0, 120.0)),
            FitParameter(name="thickness_layer_1", bounds=(60.0, 100.0)),
        ],
    )

    result = fit_task(task)

    assert result.identifiability.identifiability_status == "non_identifiable"
    assert result.identifiability.effective_rank < 2
    assert len(result.identifiability.parameter_correlation_matrix) == 2


def test_ellipsometry_observables_are_extractable() -> None:
    structure = {
        "layers": [{"constant_n": 2.0, "thickness_nm": 100.0}],
        "substrate": {"constant_n": 1.5},
    }
    measurements = [
        MeasuredDataPoint(
            wavelength_nm=550.0,
            angle_deg=45.0,
            measurement_type=MeasurementType.ELLIPSOMETRY_PSI,
            value=0.0,
        ),
        MeasuredDataPoint(
            wavelength_nm=550.0,
            angle_deg=45.0,
            measurement_type=MeasurementType.ELLIPSOMETRY_DELTA,
            value=0.0,
        ),
    ]
    forward = execute_forward_simulation(
        build_simulation_task(structure, measurements, {"thickness_layer_0": 100.0})
    )
    assert np.isfinite(extract_simulation_value(forward, measurements[0]))
    assert np.isfinite(extract_simulation_value(forward, measurements[1]))


def test_fit_cli_writes_result(tmp_path: Path) -> None:
    source = Path("benchmarks/cases/fitting/fit_synthetic_ar_coating.json")
    output = tmp_path / "fit_result.json"
    assert cli_main(["fit", str(source), "--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["converged"] is True
    assert payload["identifiability"]["identifiability_status"] == "well_determined"
