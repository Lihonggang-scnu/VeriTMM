"""JSON adapters for the model-agnostic TMM task contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .run_artifacts import write_json
from .schemas import (
    IlluminationSpec,
    LayerSpec,
    MediumSpec,
    OptimizationTask,
    OptimizerSpec,
    PhysicsRequirements,
    RobustnessSpec,
    SimulationTask,
    SpectralGrid,
    SpectralTarget,
    StackSpec,
    dataclass_to_dict,
)


def _tuple(value: Any, default: Sequence[Any]) -> tuple[Any, ...]:
    if value is None:
        value = default
    if not isinstance(value, (list, tuple)):
        raise ValueError("expected a JSON array")
    return tuple(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def medium_from_dict(data: Mapping[str, Any]) -> MediumSpec:
    return MediumSpec(
        material=data.get("material"),
        constant_n=_optional_float(data.get("constant_n")),
        constant_k=float(data.get("constant_k", 0.0)),
        provider=data.get("provider"),
        dataset_id=data.get("dataset_id"),
    )


def layer_from_dict(data: Mapping[str, Any]) -> LayerSpec:
    return LayerSpec(
        material=data.get("material"),
        thickness_nm=float(data["thickness_nm"]),
        coherence=data.get("coherence", "coherent"),
        provider=data.get("provider"),
        dataset_id=data.get("dataset_id"),
        optimizable=data.get("optimizable", True),
        min_thickness_nm=_optional_float(data.get("min_thickness_nm")),
        max_thickness_nm=_optional_float(data.get("max_thickness_nm")),
        label=data.get("label"),
        constant_n=_optional_float(data.get("constant_n")),
        constant_k=float(data.get("constant_k", 0.0)),
    )


def simulation_task_from_dict(data: Mapping[str, Any]) -> SimulationTask:
    stack_data = data["stack"]
    spectrum_data = data["spectrum"]
    illumination_data = data.get("illumination", {})
    physics_data = data.get("physics", {})
    spectrum = SpectralGrid(
        start_nm=_optional_float(spectrum_data.get("start_nm")),
        stop_nm=_optional_float(spectrum_data.get("stop_nm")),
        points=(None if spectrum_data.get("points") is None else int(spectrum_data["points"])),
        values_nm=None
        if spectrum_data.get("values_nm") is None
        else tuple(float(x) for x in spectrum_data["values_nm"]),
    )
    task = SimulationTask(
        stack=StackSpec(
            layers=tuple(layer_from_dict(item) for item in stack_data["layers"]),
            incident=medium_from_dict(stack_data.get("incident", {"constant_n": 1.0})),
            exit=medium_from_dict(stack_data.get("exit", {"material": "sio2"})),
            name=stack_data.get("name", "multilayer_stack"),
        ),
        spectrum=spectrum,
        illumination=IlluminationSpec(
            angles_deg=tuple(float(x) for x in _tuple(illumination_data.get("angles_deg"), (0.0,))),
            polarizations=tuple(
                str(x) for x in _tuple(illumination_data.get("polarizations"), ("unpolarized",))
            ),
        ),
        solver=data.get("solver", "smatrix"),
        allow_material_extrapolation=data.get("allow_material_extrapolation", False),
        requested_outputs=tuple(data.get("requested_outputs", ("R", "T", "A"))),
        physics=PhysicsRequirements(**physics_data),
    )
    task.validate()
    return task


def optimization_task_from_dict(data: Mapping[str, Any]) -> OptimizationTask:
    optimizer_data = data.get("optimizer", {})
    targets = tuple(
        SpectralTarget(
            observable=str(item["observable"]),
            target=float(item["target"]),
            wavelength_min_nm=float(item["wavelength_min_nm"]),
            wavelength_max_nm=float(item["wavelength_max_nm"]),
            weight=float(item.get("weight", 1.0)),
            angle_deg=float(item.get("angle_deg", 0.0)),
            polarization=str(item.get("polarization", "unpolarized")),
            tolerance=_optional_float(item.get("tolerance")),
            name=item.get("name"),
            constraint=str(item.get("constraint", "match")),
            aggregation=str(item.get("aggregation", "mean")),
        )
        for item in data["targets"]
    )
    optimizer = OptimizerSpec(
        method=str(optimizer_data.get("method", "adam_lbfgs")),
        max_steps=int(optimizer_data.get("max_steps", 120)),
        learning_rate=float(optimizer_data.get("learning_rate", 0.05)),
        starts=int(optimizer_data.get("starts", 4)),
        seed=int(optimizer_data.get("seed", 42)),
        early_stop_patience=int(optimizer_data.get("early_stop_patience", 20)),
        improvement_tolerance=float(optimizer_data.get("improvement_tolerance", 1e-7)),
        gradient_clip_norm=float(optimizer_data.get("gradient_clip_norm", 10.0)),
        thickness_window_nm=float(optimizer_data.get("thickness_window_nm", 150.0)),
        quantization_nm=_optional_float(optimizer_data.get("quantization_nm")),
    )
    robustness_data = data.get("robustness")
    robustness = (
        None
        if robustness_data is None
        else RobustnessSpec(
            enabled=robustness_data.get("enabled", True),
            distribution=str(robustness_data.get("distribution", "normal")),
            objective=str(robustness_data.get("objective", "expected_loss")),
            samples_per_step=int(robustness_data.get("samples_per_step", 8)),
            final_samples=int(robustness_data.get("final_samples", 128)),
            seed=int(robustness_data.get("seed", 42)),
            thickness_sigma_nm=float(robustness_data.get("thickness_sigma_nm", 2.0)),
            k_sigma=float(robustness_data.get("k_sigma", 2.0)),
            cvar_alpha=_optional_float(robustness_data.get("cvar_alpha")),
            boundary_policy=str(robustness_data.get("boundary_policy", "truncate")),
            min_thickness_physical_nm=float(
                robustness_data.get("min_thickness_physical_nm", 0.1)
            ),
        )
    )
    task = OptimizationTask(
        simulation=simulation_task_from_dict(data["simulation"]),
        targets=targets,
        optimizer=optimizer,
        robustness=robustness,
    )
    task.validate()
    return task


def load_task(path: str | Path) -> tuple[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    mode = str(payload.get("mode", "simulate")).strip().casefold()
    if mode == "simulate":
        source = payload.get("simulation", payload)
        return mode, simulation_task_from_dict(source)
    if mode == "optimize":
        source = payload.get("optimization", payload)
        return mode, optimization_task_from_dict(source)
    if mode == "sweep":
        from .protocol.models import SweepTaskContract

        contract = SweepTaskContract.model_validate(payload)
        return mode, contract.sweep
    if mode == "sensitivity":
        from .protocol.models import SensitivityTaskContract

        contract = SensitivityTaskContract.model_validate(payload)
        return mode, contract.sensitivity
    if mode == "tolerance":
        from .protocol.models import ToleranceTaskContract

        contract = ToleranceTaskContract.model_validate(payload)
        return mode, contract.tolerance
    raise ValueError("mode must be simulate, optimize, sweep, sensitivity, or tolerance")


def write_normalized_task(path: str | Path, mode: str, task: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        target,
        {
            "mode": mode,
            "simulation" if mode == "simulate" else "optimization": dataclass_to_dict(task),
        },
    )


__all__ = [
    "layer_from_dict",
    "load_task",
    "medium_from_dict",
    "optimization_task_from_dict",
    "simulation_task_from_dict",
    "write_normalized_task",
]
