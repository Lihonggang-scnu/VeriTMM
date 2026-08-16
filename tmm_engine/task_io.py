"""JSON adapters for the model-agnostic TMM task contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .schemas import (
    IlluminationSpec,
    LayerSpec,
    MediumSpec,
    OptimizationTask,
    OptimizerSpec,
    PhysicsRequirements,
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


def medium_from_dict(data: Mapping[str, Any]) -> MediumSpec:
    return MediumSpec(
        material=data.get("material"),
        constant_n=data.get("constant_n"),
        constant_k=data.get("constant_k", 0.0),
        provider=data.get("provider"),
        dataset_id=data.get("dataset_id"),
    )


def layer_from_dict(data: Mapping[str, Any]) -> LayerSpec:
    return LayerSpec(
        material=data.get("material"),
        thickness_nm=data["thickness_nm"],
        coherence=data.get("coherence", "coherent"),
        provider=data.get("provider"),
        dataset_id=data.get("dataset_id"),
        optimizable=data.get("optimizable", True),
        min_thickness_nm=data.get("min_thickness_nm"),
        max_thickness_nm=data.get("max_thickness_nm"),
        label=data.get("label"),
        constant_n=data.get("constant_n"),
        constant_k=data.get("constant_k", 0.0),
    )


def simulation_task_from_dict(data: Mapping[str, Any]) -> SimulationTask:
    stack_data = data["stack"]
    spectrum_data = data["spectrum"]
    illumination_data = data.get("illumination", {})
    physics_data = data.get("physics", {})
    spectrum = SpectralGrid(
        start_nm=spectrum_data.get("start_nm"),
        stop_nm=spectrum_data.get("stop_nm"),
        points=spectrum_data.get("points"),
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
            polarizations=tuple(str(x) for x in _tuple(illumination_data.get("polarizations"), ("unpolarized",))),
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
    task = OptimizationTask(
        simulation=simulation_task_from_dict(data["simulation"]),
        targets=tuple(SpectralTarget(**item) for item in data["targets"]),
        optimizer=OptimizerSpec(**optimizer_data),
    )
    task.validate()
    return task


def load_task(path: str | Path) -> tuple[str, SimulationTask | OptimizationTask]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    mode = str(payload.get("mode", "simulate")).strip().casefold()
    if mode == "simulate":
        source = payload.get("simulation", payload)
        return mode, simulation_task_from_dict(source)
    if mode == "optimize":
        source = payload.get("optimization", payload)
        return mode, optimization_task_from_dict(source)
    raise ValueError("mode must be simulate or optimize")


def write_normalized_task(path: str | Path, mode: str, task: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "mode": mode,
                "simulation" if mode == "simulate" else "optimization": dataclass_to_dict(task),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


__all__ = [
    "layer_from_dict",
    "load_task",
    "medium_from_dict",
    "optimization_task_from_dict",
    "simulation_task_from_dict",
    "write_normalized_task",
]
