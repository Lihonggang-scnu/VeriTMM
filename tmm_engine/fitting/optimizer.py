"""Bounded least-squares fitting and local identifiability analysis."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from scipy.linalg import svd
from scipy.optimize import least_squares

from ..material_registry import MaterialRegistry
from ..schemas import (
    IlluminationSpec,
    LayerSpec,
    MediumSpec,
    SimulationTask,
    SpectralGrid,
    StackSpec,
)
from ..workbench import ForwardSimulationResult, TMMWorkbench
from .fit_task import (
    FitParameter,
    FitResult,
    FitTask,
    IdentifiabilityReport,
    MeasuredDataPoint,
    MeasurementType,
)


def _parameter_layer_index(parameter: FitParameter, position: int) -> int:
    if parameter.layer_index is not None:
        return int(parameter.layer_index)
    match = re.search(r"(?:layer[_-]?)?(\d+)$", parameter.name.casefold())
    return int(match.group(1)) if match else position


def _medium_from_structure(value: Any, *, default_material: str | None = None) -> MediumSpec:
    if isinstance(value, str):
        return MediumSpec(material=value)
    if isinstance(value, dict):
        return MediumSpec(
            material=value.get("material"),
            constant_n=value.get("constant_n"),
            constant_k=float(value.get("constant_k", 0.0)),
            provider=value.get("provider"),
            dataset_id=value.get("dataset_id"),
        )
    if default_material is not None:
        return MediumSpec(material=default_material)
    return MediumSpec(constant_n=1.5)


def _structure_layers(structure: dict[str, Any], parameters: dict[str, float]) -> tuple[LayerSpec, ...]:
    raw_layers = structure.get("layers")
    if isinstance(raw_layers, list):
        layers: list[LayerSpec] = []
        for index, raw in enumerate(raw_layers):
            if not isinstance(raw, dict):
                raise ValueError("structure.layers entries must be objects")
            thickness = float(raw.get("thickness_nm", 100.0))
            for name, value in parameters.items():
                parameter_index = _parameter_layer_index(
                    FitParameter(name=name, bounds=(0.0, max(1.0, value + 1.0))), index
                )
                if parameter_index == index:
                    thickness = float(value)
            material = raw.get("material")
            constant_n = raw.get("constant_n")
            layers.append(
                LayerSpec(
                    material=material,
                    constant_n=None if material is not None else constant_n,
                    constant_k=float(raw.get("constant_k", 0.0)),
                    thickness_nm=thickness,
                    coherence=str(raw.get("coherence", "coherent")),
                    provider=raw.get("provider"),
                    dataset_id=raw.get("dataset_id"),
                    optimizable=False,
                    label=raw.get("label"),
                )
            )
        return tuple(layers)

    materials = structure.get("materials")
    thicknesses = structure.get("thicknesses_nm")
    if not isinstance(materials, list) or not isinstance(thicknesses, list):
        raise ValueError("structure requires layers or materials plus thicknesses_nm")
    if len(materials) != len(thicknesses):
        raise ValueError("structure materials and thicknesses_nm lengths must match")
    layers = []
    for index, (material, thickness) in enumerate(zip(materials, thicknesses, strict=True)):
        material_name = None if isinstance(material, dict) else str(material)
        material_data = material if isinstance(material, dict) else {}
        value = float(thickness)
        for name, parameter_value in parameters.items():
            if _parameter_layer_index(
                FitParameter(name=name, bounds=(0.0, max(1.0, parameter_value + 1.0))),
                index,
            ) == index:
                value = float(parameter_value)
        layers.append(
            LayerSpec(
                material=material_name,
                constant_n=material_data.get("constant_n"),
                constant_k=float(material_data.get("constant_k", 0.0)),
                thickness_nm=value,
                optimizable=False,
                label=material_data.get("label") or material_name,
            )
        )
    return tuple(layers)


def build_simulation_task(
    structure: dict[str, Any],
    measurements: list[MeasuredDataPoint] | tuple[MeasuredDataPoint, ...] | MeasuredDataPoint,
    params: dict[str, float],
) -> SimulationTask:
    """Build one validated forward task for the current parameter vector."""

    measurement_list = [measurements] if isinstance(measurements, MeasuredDataPoint) else list(measurements)
    if not measurement_list:
        raise ValueError("at least one measurement is required")
    wavelengths = tuple(sorted({float(item.wavelength_nm) for item in measurement_list}))
    angles = tuple(sorted({float(item.angle_deg) for item in measurement_list}))
    polarizations = tuple(
        dict.fromkeys(str(item.polarization) for item in measurement_list)
    )
    requested = {"R", "T"}
    if any(
        item.measurement_type
        in {MeasurementType.ELLIPSOMETRY_PSI, MeasurementType.ELLIPSOMETRY_DELTA}
        for item in measurement_list
    ):
        requested.add("ellipsometry")
    if requested == {"R", "T"} and not any(
        item.measurement_type in {MeasurementType.REFLECTANCE, MeasurementType.TRANSMITTANCE}
        for item in measurement_list
    ):
        requested = {"ellipsometry"}
    exit_medium = structure.get("substrate", structure.get("exit"))
    incident_value = structure.get("incident")
    incident = (
        MediumSpec.air()
        if incident_value is None
        else _medium_from_structure(incident_value)
    )
    task = SimulationTask(
        stack=StackSpec(
            layers=_structure_layers(structure, params),
            incident=incident,
            exit=_medium_from_structure(exit_medium, default_material="sio2"),
            name=str(structure.get("name", "fit_structure")),
        ),
        spectrum=SpectralGrid(values_nm=wavelengths),
        illumination=IlluminationSpec(angles_deg=angles, polarizations=polarizations),
        solver="byrnes" if "ellipsometry" in requested else "smatrix",
        requested_outputs=tuple(sorted(requested)),
    )
    task.validate()
    return task


def execute_forward_simulation(
    task: SimulationTask,
    *,
    workbench: TMMWorkbench | None = None,
) -> ForwardSimulationResult:
    """Run a fit forward model without issuing a physics certificate."""

    return (workbench or TMMWorkbench(MaterialRegistry())).simulate(task)


def _channel_key(angle_deg: float, polarization: str) -> str:
    return f"angle={float(angle_deg):g}|pol={polarization}"


def _interpolate(values: Any, wavelengths: np.ndarray, wavelength_nm: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.interp(float(wavelength_nm), wavelengths, array))


def extract_simulation_value(
    result: ForwardSimulationResult,
    measurement: MeasuredDataPoint,
) -> float:
    """Extract and interpolate one R/T/Ψ/Δ observable from a forward result."""

    wavelengths = np.asarray(result.wavelengths_nm, dtype=np.float64)
    if measurement.measurement_type in {
        MeasurementType.ELLIPSOMETRY_PSI,
        MeasurementType.ELLIPSOMETRY_DELTA,
    }:
        extra = result.extras.get(f"ellipsometry|angle={float(measurement.angle_deg):g}")
        if not isinstance(extra, dict):
            raise ValueError("forward result does not contain requested ellipsometry angle")
        key = "psi_rad" if measurement.measurement_type == MeasurementType.ELLIPSOMETRY_PSI else "delta_rad"
        return _interpolate(extra[key], wavelengths, measurement.wavelength_nm)
    channel = result.channels.get(_channel_key(measurement.angle_deg, measurement.polarization))
    if channel is None:
        raise ValueError("forward result does not contain requested angle/polarization channel")
    key = measurement.measurement_type.value
    return _interpolate(channel[key], wavelengths, measurement.wavelength_nm)


def _residual_difference(measured: float, simulated: float, kind: MeasurementType) -> float:
    difference = float(measured) - float(simulated)
    if kind == MeasurementType.ELLIPSOMETRY_DELTA:
        return float(np.arctan2(np.sin(difference), np.cos(difference)))
    return difference


def _weighted_residual(measurement: MeasuredDataPoint, simulated: float) -> float:
    difference = _residual_difference(measurement.value, simulated, measurement.measurement_type)
    if measurement.uncertainty is not None:
        return difference / float(measurement.uncertainty)
    return difference * float(measurement.weight)


def _identifiability(
    jacobian: np.ndarray,
    residuals: np.ndarray,
    parameter_count: int,
) -> IdentifiabilityReport:
    _, singular_values, _ = svd(jacobian, full_matrices=False)
    singular_values = np.asarray(singular_values, dtype=np.float64)
    threshold = float(singular_values[0] * 1e-6) if singular_values.size else 0.0
    effective_rank = int(np.sum(singular_values > threshold)) if singular_values.size else 0
    condition_number = (
        float(singular_values[0] / singular_values[-1])
        if singular_values.size and singular_values[-1] > 0
        else float("inf")
    )
    jtj = np.asarray(jacobian, dtype=np.float64).T @ np.asarray(jacobian, dtype=np.float64)
    covariance = np.linalg.pinv(jtj) if parameter_count else np.empty((0, 0))
    diagonal = np.diag(covariance) if covariance.size else np.empty(0)
    denominator = np.sqrt(np.outer(np.maximum(diagonal, 0.0), np.maximum(diagonal, 0.0)))
    correlation = np.divide(
        covariance,
        denominator,
        out=np.zeros_like(covariance, dtype=np.float64),
        where=denominator > 0,
    )
    for index, value in enumerate(diagonal):
        if value > 0:
            correlation[index, index] = 1.0
    if effective_rank < parameter_count or condition_number >= 100:
        status = "non_identifiable"
    elif condition_number >= 10:
        status = "weakly_identifiable"
    else:
        status = "well_determined"
    dof = max(0, int(residuals.size) - parameter_count)
    rmse = float(np.sqrt(np.mean(np.square(residuals)))) if residuals.size else 0.0
    chi_squared = float(np.sum(np.square(residuals))) if residuals.size else 0.0
    return IdentifiabilityReport(
        rmse=rmse,
        chi_squared=chi_squared,
        degrees_of_freedom=dof,
        jacobian_condition_number=condition_number,
        singular_values=singular_values.tolist(),
        effective_rank=effective_rank,
        parameter_correlation_matrix=correlation.tolist(),
        identifiability_status=status,
    )


def fit_task(task: FitTask) -> FitResult:
    """Fit bounded parameters and always return an identifiability report."""

    parameter_names = [item.name for item in task.fit_parameters]
    lower = np.asarray([item.bounds[0] for item in task.fit_parameters], dtype=np.float64)
    upper = np.asarray([item.bounds[1] for item in task.fit_parameters], dtype=np.float64)
    initial = np.asarray(
        [
            item.initial_guess
            if item.initial_guess is not None
            else (item.bounds[0] + item.bounds[1]) / 2.0
            for item in task.fit_parameters
        ],
        dtype=np.float64,
    )
    workbench = TMMWorkbench(MaterialRegistry())

    def residual_fn(values: np.ndarray) -> np.ndarray:
        params = {name: float(value) for name, value in zip(parameter_names, values, strict=True)}
        simulation_task = build_simulation_task(task.structure, task.measurements, params)
        forward = execute_forward_simulation(simulation_task, workbench=workbench)
        return np.asarray(
            [
                _weighted_residual(measurement, extract_simulation_value(forward, measurement))
                for measurement in task.measurements
            ],
            dtype=np.float64,
        )

    optimizer_method = "lm" if task.method == "bounded_lm" and np.all(np.isfinite(lower)) else "trf"
    if optimizer_method == "lm":
        # ``lm`` cannot honor bounds; retain bounded semantics for this public
        # contract by using trust-region reflective when bounds are supplied.
        optimizer_method = "trf"
    result = least_squares(
        residual_fn,
        np.clip(initial, lower, upper),
        bounds=(lower, upper),
        max_nfev=int(task.max_iterations),
        ftol=float(task.tolerance),
        xtol=float(task.tolerance),
        gtol=float(task.tolerance),
        method=optimizer_method,
    )
    best_parameters = {
        name: float(value) for name, value in zip(parameter_names, result.x, strict=True)
    }
    residuals = np.asarray(result.fun, dtype=np.float64)
    identifiability = _identifiability(
        np.asarray(result.jac, dtype=np.float64),
        residuals,
        len(parameter_names),
    )
    fit_certificate = {
        "fit_quality": "acceptable" if identifiability.rmse < 0.01 else "poor",
        "physics_certificate": None,
        "physics_validity": "not_certified",
        "certificate_authority": "fit_quality_only",
        "identifiability_status": identifiability.identifiability_status,
    }
    return FitResult(
        task=task,
        converged=bool(result.success),
        iterations=int(result.nfev),
        best_fit_parameters=best_parameters,
        residuals=residuals.tolist(),
        jacobian=np.asarray(result.jac, dtype=np.float64).tolist(),
        identifiability=identifiability,
        fit_certificate=fit_certificate,
    )


__all__ = [
    "build_simulation_task",
    "execute_forward_simulation",
    "extract_simulation_value",
    "fit_task",
]
