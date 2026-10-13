from __future__ import annotations

from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest

from tmm_engine.fitting.fit_task import (
    FitParameter,
    FitResult,
    FitTask,
    IdentifiabilityReport,
    MeasuredDataPoint,
    MeasurementType,
)
from tmm_engine.fitting.measurement_plan import (
    MeasurementAction,
    MeasurementPlanError,
    MeasurementPlanTask,
    _finite_difference_action,
    build_measurement_plan,
)
from tmm_engine.fitting.optimizer import (
    build_simulation_task,
    execute_forward_simulation,
    extract_simulation_value,
    fit_task,
)


@pytest.fixture(params=["layers", "materials"])
def structure(request: pytest.FixtureRequest) -> dict:
    result = {"substrate": {"constant_n": 1.5}}
    if request.param == "layers":
        result["layers"] = [
            {"constant_n": 2.1, "thickness_nm": 75.0},
            {"constant_n": 1.7, "thickness_nm": 125.0},
        ]
    else:
        result["materials"] = [{"constant_n": 2.1}, {"constant_n": 1.7}]
        result["thicknesses_nm"] = [75.0, 125.0]
    return result


def _measurement(wavelength: float = 550.0) -> MeasuredDataPoint:
    return MeasuredDataPoint(
        wavelength_nm=wavelength,
        measurement_type=MeasurementType.REFLECTANCE,
        value=0.0,
        uncertainty=0.02,
    )


def _with_thickness(structure: dict, index: int, value: float) -> dict:
    result = deepcopy(structure)
    if "layers" in result:
        result["layers"][index]["thickness_nm"] = value
    else:
        result["thicknesses_nm"][index] = value
    return result


def _prediction(structure: dict, measurement: MeasuredDataPoint) -> float:
    forward = execute_forward_simulation(build_simulation_task(structure, measurement, {}))
    return extract_simulation_value(forward, measurement)


def _fit_result(parameter: FitParameter, value: float, structure: dict) -> FitResult:
    return FitResult(
        task=FitTask(
            structure=structure,
            measurements=[_measurement()],
            fit_parameters=[parameter],
        ),
        converged=True,
        iterations=1,
        best_fit_parameters={parameter.name: value},
        residuals=[0.0],
        jacobian=None,
        identifiability=IdentifiabilityReport(
            rmse=0.0,
            degrees_of_freedom=0,
            jacobian_condition_number=1.0,
            singular_values=[1.0],
            effective_rank=1,
            parameter_correlation_matrix=[[1.0]],
            identifiability_status="well_determined",
        ),
        fit_certificate={"physics_validity": "not_certified"},
    )


@pytest.mark.parametrize("name", ["film", "thickness_layer_0", "n"])
def test_explicit_binding_overrides_name_and_only_changes_thickness(structure, name):
    parameter = FitParameter(name=name, layer_index=1, bounds=(100.0, 150.0))
    task = build_simulation_task(
        structure, _measurement(), {name: 140.0}, fit_parameters=[parameter]
    )

    assert [layer.thickness_nm for layer in task.stack.layers] == [75.0, 140.0]
    assert [layer.constant_n for layer in task.stack.layers] == [2.1, 1.7]
    assert [layer.constant_k for layer in task.stack.layers] == [0.0, 0.0]


def test_legacy_names_bind_by_parameter_position_or_suffix(structure):
    positional = build_simulation_task(structure, _measurement(), {"front": 80.0, "rear": 140.0})
    suffixed = build_simulation_task(structure, _measurement(), {"thickness_layer_1": 140.0})
    single = build_simulation_task(structure, _measurement(), {"front": 80.0})

    assert [layer.thickness_nm for layer in positional.stack.layers] == [80.0, 140.0]
    assert [layer.thickness_nm for layer in suffixed.stack.layers] == [75.0, 140.0]
    assert [layer.thickness_nm for layer in single.stack.layers] == [80.0, 125.0]


def test_typed_parameter_order_is_independent_of_value_dictionary_order(structure):
    parameters = [
        FitParameter(name="front", bounds=(50.0, 150.0)),
        FitParameter(name="rear", bounds=(50.0, 150.0)),
    ]
    task = build_simulation_task(
        structure,
        _measurement(),
        {"rear": 140.0, "front": 80.0},
        fit_parameters=parameters,
    )

    assert [layer.thickness_nm for layer in task.stack.layers] == [80.0, 140.0]


@pytest.mark.parametrize("explicit", [False, True])
def test_missing_layer_binding_is_rejected(structure, explicit):
    parameter = FitParameter(name="thickness_7", layer_index=4, bounds=(50.0, 150.0))
    with pytest.raises(ValueError, match="targets missing layer"):
        build_simulation_task(
            structure,
            _measurement(),
            {parameter.name: 100.0},
            fit_parameters=[parameter] if explicit else None,
        )


def test_duplicate_layer_binding_is_rejected(structure):
    parameters = [
        FitParameter(name="front", layer_index=0, bounds=(50.0, 150.0)),
        FitParameter(name="rear", layer_index=0, bounds=(50.0, 150.0)),
    ]
    with pytest.raises(ValueError, match="multiple fit parameters target layer 0"):
        build_simulation_task(
            structure,
            _measurement(),
            {"front": 80.0, "rear": 140.0},
            fit_parameters=parameters,
        )


def test_parameter_definition_mismatch_is_rejected(structure):
    parameter = FitParameter(name="film", layer_index=1, bounds=(50.0, 150.0))
    with pytest.raises(ValueError, match="match parameter values by unique name"):
        build_simulation_task(
            structure, _measurement(), {"other": 140.0}, fit_parameters=[parameter]
        )


@pytest.mark.parametrize("name", ["film", "thickness_layer_0"])
def test_real_multilayer_fit_preserves_explicit_binding(structure, name):
    target = _with_thickness(structure, 1, 142.0)
    measurements = [_measurement(wavelength) for wavelength in (450.0, 500.0, 550.0, 650.0)]
    measurements = [
        item.model_copy(update={"value": _prediction(target, item)}) for item in measurements
    ]
    result = fit_task(
        FitTask(
            structure=structure,
            measurements=measurements,
            fit_parameters=[
                FitParameter(
                    name=name,
                    layer_index=1,
                    bounds=(130.0, 150.0),
                    initial_guess=135.0,
                )
            ],
            tolerance=1e-11,
        )
    )

    assert result.converged
    assert result.best_fit_parameters[name] == pytest.approx(142.0, abs=1e-6)
    assert result.identifiability.rmse < 1e-8
    assert result.task.structure == structure


@pytest.mark.parametrize("name,index", [("film", 1), ("thickness_layer_0", 1), ("front", None)])
def test_default_planner_and_rebuilt_baseline_preserve_bindings(structure, name, index):
    parameter = FitParameter(name=name, layer_index=index, bounds=(50.0, 150.0))
    value = 100.0
    fit = _fit_result(parameter, value, structure)
    action = MeasurementAction(
        wavelength_nm=610.0,
        measurement_type=MeasurementType.REFLECTANCE,
        sigma=0.05,
    )
    result = build_measurement_plan(MeasurementPlanTask(fit_result=fit, candidates=[action]))
    physical_index = 0 if index is None else index
    step = 0.001

    def derivative(measurement):
        plus = _prediction(_with_thickness(structure, physical_index, value + step), measurement)
        minus = _prediction(_with_thickness(structure, physical_index, value - step), measurement)
        return (plus - minus) / (2.0 * step)

    baseline = derivative(fit.task.measurements[0]) / 0.02
    expected = derivative(_measurement(610.0))
    assert result.candidate_scores[0].jacobian == pytest.approx([expected / action.sigma], rel=1e-8)
    assert result.fisher_information_before[0][0] == pytest.approx(baseline**2, rel=1e-8)
    assert result.fisher_information_after[0][0] == pytest.approx(
        baseline**2 + (expected / action.sigma) ** 2, rel=1e-8
    )


@pytest.mark.parametrize("fraction", [0.0, 0.25, 0.5, 1.0])
def test_narrow_bounds_adapt_step_without_silently_zeroing_derivative(structure, fraction):
    low, high = 100.0, 100.00000001
    value = low + fraction * (high - low)
    parameter = FitParameter(name="film", layer_index=1, bounds=(low, high))
    fit = _fit_result(parameter, value, structure)
    action = MeasurementAction(
        wavelength_nm=550.0, measurement_type=MeasurementType.REFLECTANCE, sigma=1.0
    )
    visited = []

    def evaluate(_fit, _action, params, **_kwargs):
        visited.append(params["film"])
        assert low <= params["film"] <= high
        return 3.0 * (params["film"] - value)

    with patch("tmm_engine.fitting.measurement_plan._evaluate_action", side_effect=evaluate):
        row = _finite_difference_action(fit, action, workbench=None)

    assert row == pytest.approx([3.0], rel=1e-12)
    assert len(set(visited)) == 2


def test_adjacent_float_bounds_use_actual_parameter_interval(structure):
    low = 100.0
    high = float(np.nextafter(low, np.inf))
    parameter = FitParameter(name="film", layer_index=1, bounds=(low, high))
    fit = _fit_result(parameter, low, structure)
    action = MeasurementAction(
        wavelength_nm=550.0, measurement_type=MeasurementType.REFLECTANCE, sigma=1.0
    )
    with patch(
        "tmm_engine.fitting.measurement_plan._evaluate_action",
        side_effect=lambda _fit, _action, params, **_kwargs: params["film"] - low,
    ):
        row = _finite_difference_action(fit, action, workbench=None)

    assert row == pytest.approx([1.0])


def test_narrow_bounds_near_boundary_use_available_room(structure):
    low, high = 100.0, 100.00000001
    value = float(np.nextafter(low, high))
    parameter = FitParameter(name="film", layer_index=1, bounds=(low, high))
    fit = _fit_result(parameter, value, structure)
    action = MeasurementAction(
        wavelength_nm=550.0, measurement_type=MeasurementType.REFLECTANCE, sigma=1.0
    )
    visited = []

    def evaluate(_fit, _action, params, **_kwargs):
        visited.append(params["film"])
        return 1.0 + 0.003 * (params["film"] - value)

    with patch("tmm_engine.fitting.measurement_plan._evaluate_action", side_effect=evaluate):
        row = _finite_difference_action(fit, action, workbench=None)

    assert row == pytest.approx([0.003], rel=1e-5)
    assert visited == [value, high]


@pytest.mark.parametrize("value", [50.0, 100.0, 150.0])
@pytest.mark.parametrize("phase", [0.0, np.pi])
def test_delta_derivative_uses_circular_difference_at_phase_wrap(structure, value, phase):
    parameter = FitParameter(name="film", layer_index=1, bounds=(50.0, 150.0))
    fit = _fit_result(parameter, value, structure)
    action = MeasurementAction(
        wavelength_nm=550.0, measurement_type=MeasurementType.ELLIPSOMETRY_DELTA, sigma=1.0
    )

    def evaluate(_fit, _action, params, **_kwargs):
        raw = phase + 0.7 * (params["film"] - value)
        return raw % (2.0 * np.pi) if phase == 0.0 else np.arctan2(np.sin(raw), np.cos(raw))

    with patch("tmm_engine.fitting.measurement_plan._evaluate_action", side_effect=evaluate):
        row = _finite_difference_action(fit, action, workbench=None)

    assert row == pytest.approx([0.7], rel=1e-9)


def test_real_ellipsometry_jacobian_crosses_phase_branch_without_spike(structure):
    angle = 60.0
    value = 550.0 / (2.1 * 2.0 * np.sqrt(1.0 - (np.sin(np.deg2rad(angle)) / 2.1) ** 2))
    optical_structure = {"substrate": {"constant_n": 1.5}}
    if "layers" in structure:
        optical_structure["layers"] = [{"constant_n": 2.1, "thickness_nm": value}]
    else:
        optical_structure["materials"] = [{"constant_n": 2.1}]
        optical_structure["thicknesses_nm"] = [value]
    parameter = FitParameter(name="thickness_layer_0", layer_index=0, bounds=(value - 1, value + 1))
    fit = _fit_result(parameter, value, optical_structure)
    action = MeasurementAction(
        wavelength_nm=550.0,
        angle_deg=angle,
        measurement_type=MeasurementType.ELLIPSOMETRY_DELTA,
        sigma=1.0,
    )
    measurement = _measurement().model_copy(
        update={"angle_deg": angle, "measurement_type": MeasurementType.ELLIPSOMETRY_DELTA}
    )
    step = value * 1e-5
    plus = _prediction(_with_thickness(optical_structure, 0, value + step), measurement)
    minus = _prediction(_with_thickness(optical_structure, 0, value - step), measurement)
    expected = np.angle(np.exp(1j * plus) / np.exp(1j * minus)) / (2.0 * step)

    row = _finite_difference_action(fit, action, workbench=None)

    assert abs(plus - minus) > 6.0
    assert abs(expected) < 1.0
    assert row == pytest.approx([expected], rel=1e-8)


def test_nonfinite_fit_parameter_has_typed_diagnostic(structure):
    parameter = FitParameter(name="film", layer_index=1, bounds=(0.0, np.inf))
    fit = _fit_result(parameter, np.inf, structure)
    action = MeasurementAction(
        wavelength_nm=550.0, measurement_type=MeasurementType.REFLECTANCE, sigma=1.0
    )
    with pytest.raises(MeasurementPlanError) as caught:
        _finite_difference_action(fit, action, workbench=None)

    assert caught.value.code == "nonfinite_fit_parameter"
