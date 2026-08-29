"""Deterministic task preflight for AI and command-line callers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .capabilities import (
    FailureRecord,
    assess_tmm_capability,
    enrich_failure_actions,
    failure_from_exception,
)
from .material_registry import MaterialRegistry
from .protocol.models import PROTOCOL_VERSION
from .schemas import OptimizationTask, SimulationTask
from .task_io import load_task

PREFLIGHT_SCHEMA_VERSION = "veritmm-preflight-v1"
def _warning(code: str, message: str, **context: Any) -> dict[str, Any]:
    return {
        "code": code,
        "severity": "warning",
        "message": message,
        "context": context,
    }


def _constant_material(position: str, item: Any) -> dict[str, Any]:
    return {
        "stack_position": position,
        "material_model": "constant_nk",
        "resolved": True,
        "provider": "constant",
        "dataset_id": None,
        "n": float(item.constant_n),
        "k": float(item.constant_k),
        "extrapolated": False,
    }


def _material_items(task: SimulationTask) -> Iterable[tuple[str, Any]]:
    yield "incident", task.stack.incident
    for index, layer in enumerate(task.stack.layers):
        yield f"layer_{index}", layer
    yield "exit", task.stack.exit


def _resolve_materials(
    task: SimulationTask,
    registry: MaterialRegistry,
    warnings: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[FailureRecord]]:
    wavelengths_nm = task.spectrum.wavelengths_nm()
    sample_um = np.asarray(wavelengths_nm, dtype=np.float64) * 1e-3
    resolved: list[dict[str, Any]] = []
    failures: list[FailureRecord] = []
    for position, item in _material_items(task):
        if item.constant_n is not None:
            resolved.append(_constant_material(position, item))
            continue
        try:
            sampled = registry.sample(
                str(item.material),
                sample_um,
                provider=item.provider,
                dataset_id=item.dataset_id,
                allow_extrapolation=task.allow_material_extrapolation,
            )
            provenance = dict(sampled.provenance or {})
            ref = sampled.ref
            extrapolated = bool(np.any(sampled.extrapolated_mask))
            record = {
                "stack_position": position,
                "material_model": "tabulated_nk",
                "material": str(item.material),
                "resolved": True,
                "provider": getattr(ref, "provider", None)
                or provenance.get("provider"),
                "dataset_id": getattr(ref, "dataset_id", None)
                or provenance.get("dataset_id"),
                "book": getattr(ref, "book", None),
                "page": getattr(ref, "page", None),
                "available_range_um": getattr(ref, "range", None),
                "requested_range_um": [float(sample_um[0]), float(sample_um[-1])],
                "sampled_wavelength_count": int(sample_um.size),
                "extrapolated": extrapolated,
                "source": provenance.get("source") or provenance.get("filepath"),
            }
            resolved.append(record)
            if extrapolated:
                warnings.append(
                    _warning(
                        "material_extrapolation_in_use",
                        "A selected optical-constant dataset is being extrapolated by explicit request.",
                        stack_position=position,
                        material=str(item.material),
                        provider=record["provider"],
                        dataset_id=record["dataset_id"],
                    )
                )
        except Exception as exc:
            failure = failure_from_exception(exc)
            context = dict(failure.context)
            context.setdefault("stack_position", position)
            context.setdefault("material", str(item.material))
            alternatives: list[dict[str, Any]] = []
            try:
                candidates = registry.search(
                    str(item.material),
                    wavelength_range=(float(sample_um[0]), float(sample_um[-1])),
                )
                for candidate in candidates:
                    if not candidate.full_coverage:
                        continue
                    try:
                        registry.sample(
                            candidate.ref,
                            sample_um,
                            allow_extrapolation=False,
                        )
                    except Exception:
                        continue
                    alternatives.append(
                        {
                            "provider": candidate.provider,
                            "dataset_id": candidate.dataset_id,
                            "book": candidate.book,
                            "page": candidate.page,
                            "range_um": candidate.range,
                            "verified_on_requested_grid": True,
                        }
                    )
                    if len(alternatives) >= 5:
                        break
            except Exception:
                alternatives = []
            if alternatives:
                context["alternative_covering_datasets"] = alternatives
                context["alternatives_are_suggestions_only"] = True
            failures.append(
                enrich_failure_actions(
                    FailureRecord(
                        code=failure.code,
                        message=failure.message,
                        recoverable=failure.recoverable,
                        suggested_solver_family=failure.suggested_solver_family,
                        context=context,
                        severity=failure.severity,
                        requires_user_choice=failure.requires_user_choice,
                        # Rebuild navigation actions from the expanded context
                        # so verified alternative datasets are visible to the
                        # caller without being selected automatically.
                        actions=(),
                    )
                )
            )
            resolved.append(
                {
                    "stack_position": position,
                    "material_model": "tabulated_nk",
                    "material": str(item.material),
                    "resolved": False,
                    "provider": item.provider,
                    "dataset_id": item.dataset_id,
                    "failure_code": failure.code.value,
                }
            )
    return resolved, failures


def _numerical_risks(task: SimulationTask) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    wavelengths = task.spectrum.wavelengths_nm()
    minimum_wavelength = float(wavelengths[0])
    layer_count = len(task.stack.layers)
    if wavelengths.size < 31:
        warnings.append(
            _warning(
                "sparse_spectral_grid",
                "The wavelength grid may miss narrow spectral features.",
                wavelength_points=int(wavelengths.size),
            )
        )
    if any(float(angle) >= 80.0 for angle in task.illumination.angles_deg):
        warnings.append(
            _warning(
                "near_grazing_incidence",
                "Angles at or above 80 degrees can be numerically sensitive.",
                maximum_angle_deg=max(float(x) for x in task.illumination.angles_deg),
            )
        )
    if layer_count > 200:
        warnings.append(
            _warning(
                "very_large_stack",
                "The stack contains more than 200 finite layers.",
                layer_count=layer_count,
            )
        )
    thick = [
        index
        for index, layer in enumerate(task.stack.layers)
        if layer.coherence == "coherent"
        and float(layer.thickness_nm) > 100.0 * minimum_wavelength
    ]
    if thick:
        warnings.append(
            _warning(
                "very_thick_coherent_layers",
                "One or more coherent layers are extremely thick relative to wavelength; verify the coherence model.",
                layer_indices=thick,
            )
        )
    thin = [
        index
        for index, layer in enumerate(task.stack.layers)
        if float(layer.thickness_nm) < 1e-4 * minimum_wavelength
    ]
    if thin:
        warnings.append(
            _warning(
                "extremely_thin_layers",
                "One or more layers are extremely thin relative to the shortest wavelength.",
                layer_indices=thin,
            )
        )
    if "phase_dispersion" in task.requested_outputs and wavelengths.size < 9:
        warnings.append(
            _warning(
                "phase_dispersion_grid_too_small",
                "Phase and numerical dispersion derivatives benefit from at least nine wavelength samples.",
                wavelength_points=int(wavelengths.size),
            )
        )
    return warnings


def _work_estimate(mode: str, task: SimulationTask | OptimizationTask) -> dict[str, Any]:
    simulation = task if isinstance(task, SimulationTask) else task.simulation
    points = int(simulation.spectrum.wavelengths_nm().size)
    channels = len(simulation.illumination.angles_deg) * len(
        simulation.illumination.polarizations
    )
    estimate = {
        "layer_count": len(simulation.stack.layers),
        "wavelength_points": points,
        "declared_angle_polarization_channels": channels,
        "forward_channel_spectra": points * channels,
        "optimization_max_evaluations": 0,
    }
    if mode == "optimize" and isinstance(task, OptimizationTask):
        estimate["optimization_max_evaluations"] = int(
            task.optimizer.max_steps * task.optimizer.starts
        )
        estimate["optimization_starts"] = int(task.optimizer.starts)
        estimate["optimization_max_steps_per_start"] = int(task.optimizer.max_steps)
        estimate["torch_available_in_current_runtime"] = bool(
            importlib.util.find_spec("torch")
        )
        if task.robustness is not None and task.robustness.enabled:
            estimate["robust_training_samples_per_step"] = int(
                task.robustness.samples_per_step
            )
            estimate["formal_robustness_samples_per_candidate"] = int(
                task.robustness.final_samples
            )
            estimate["robust_objective"] = task.robustness.objective
    return estimate


def _metric_contract_failures(
    simulation: SimulationTask,
    metrics: Iterable[Any],
    *,
    differentiable: bool = False,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    wavelengths = simulation.spectrum.wavelengths_nm()
    channels = {
        (float(angle), str(pol))
        for angle in simulation.illumination.angles_deg
        for pol in simulation.illumination.polarizations
    }
    for metric in metrics:
        try:
            if (float(metric.angle_deg), str(metric.polarization)) not in channels:
                raise ValueError(
                    f"metric {metric.name!r} requests an undeclared angle/polarization channel"
                )
            if metric.observable == "E_system":
                if differentiable:
                    raise ValueError("E_system is unsupported by differentiable sensitivity")
                if "system_emissivity" not in simulation.requested_outputs:
                    raise ValueError(
                        "E_system metric requires requested_outputs to include system_emissivity"
                    )
            lo = (
                float(wavelengths[0])
                if metric.wavelength_min_nm is None
                else float(metric.wavelength_min_nm)
            )
            hi = (
                float(wavelengths[-1])
                if metric.wavelength_max_nm is None
                else float(metric.wavelength_max_nm)
            )
            if not np.any((wavelengths >= lo) & (wavelengths <= hi)):
                raise ValueError(f"metric {metric.name!r} does not overlap the wavelength grid")
            if metric.wavelength_nm is not None and not (
                float(wavelengths[0]) <= float(metric.wavelength_nm) <= float(wavelengths[-1])
            ):
                raise ValueError(f"metric {metric.name!r} wavelength_nm lies outside the grid")
        except Exception as exc:
            failures.append(failure_from_exception(exc).to_dict())
    return failures


def preflight_task(
    mode: str,
    task: SimulationTask | OptimizationTask,
    registry: MaterialRegistry | None = None,
) -> dict[str, Any]:
    """Validate contract, capability, materials, routing and numerical risks.

    Material data are sampled on the complete declared wavelength grid so an
    internal tabulation gap cannot pass an endpoint-only check.  This function
    never executes a TMM spectrum calculation or optimization.
    """

    registry = registry or MaterialRegistry()
    failures: list[FailureRecord] = []
    warnings: list[dict[str, Any]] = []
    mode_token = str(mode)
    expected_type = (
        SimulationTask
        if mode_token == "simulate"
        else (OptimizationTask if mode_token == "optimize" else None)
    )
    simulation: SimulationTask | None = None
    if expected_type is None:
        failures.append(
            failure_from_exception(
                ValueError("mode must be exactly 'simulate' or 'optimize'")
            )
        )
    elif not isinstance(task, expected_type):
        failures.append(
            failure_from_exception(
                TypeError(
                    f"{mode_token} mode requires {expected_type.__name__}, got {type(task).__name__}"
                )
            )
        )
    else:
        simulation = task if isinstance(task, SimulationTask) else task.simulation
        try:
            task.validate()
        except Exception as exc:
            failures.append(failure_from_exception(exc))
        if (
            mode_token == "optimize"
            and isinstance(task, OptimizationTask)
            and task.simulation.stack.has_incoherent_layers
        ):
            failures.append(
                failure_from_exception(
                    ValueError("differentiable optimization currently requires coherent layers")
                )
            )

    assessment = None
    if simulation is not None and not any(
        item.code.value == "invalid_task" for item in failures
    ):
        assessment = assess_tmm_capability(simulation)
        failures.extend(enrich_failure_actions(item) for item in assessment.failures)
        warnings.extend(
            _warning("backend_routing_notice", item)
            for item in assessment.warnings
        )

    materials: list[dict[str, Any]] = []
    if simulation is not None and not failures:
        materials, material_failures = _resolve_materials(simulation, registry, warnings)
        failures.extend(material_failures)
        warnings.extend(_numerical_risks(simulation))

    # Preserve first occurrence while avoiding duplicate contract/capability
    # failures emitted by both the task and assessment validators.
    unique_failures: list[FailureRecord] = []
    seen: set[tuple[str, str]] = set()
    for failure in failures:
        marker = (failure.code.value, failure.message)
        if marker not in seen:
            seen.add(marker)
            unique_failures.append(failure)

    requested_solver = None if simulation is None else simulation.solver
    resolved_solver = None if assessment is None else assessment.resolved_solver
    routing_reason = "requested_solver_supported"
    if resolved_solver and resolved_solver != requested_solver:
        routing_reason = "requested_outputs_or_mixed_coherence_require_reference_backend"
    if unique_failures:
        resolved_solver = None
        routing_reason = "task_rejected_before_execution"

    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "ok": not unique_failures,
        "operation": "preflight",
        "mode": mode_token if mode_token in {"simulate", "optimize"} else "unknown",
        "status": "ready" if not unique_failures else "rejected",
        "contract_valid": not any(
            item.code.value == "invalid_task" for item in unique_failures
        ),
        "capability": None if assessment is None else assessment.to_dict(),
        "backend_resolution": {
            "requested_solver": requested_solver,
            "resolved_solver": resolved_solver,
            "reason": routing_reason,
        },
        "materials": materials,
        "warnings": warnings,
        "failures": [item.to_dict() for item in unique_failures],
        "estimated_work": (
            _work_estimate(mode_token, task)
            if not any(item.code.value == "invalid_task" for item in unique_failures)
            else {}
        ),
    }


def preflight_path(
    path: str | Path, registry: MaterialRegistry | None = None
) -> dict[str, Any]:
    """Load a JSON task and always return a machine-readable preflight report."""

    try:
        mode, task = load_task(path)
    except Exception as exc:
        failure = failure_from_exception(exc)
        return {
            "schema_version": PREFLIGHT_SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "ok": False,
            "operation": "preflight",
            "mode": "unknown",
            "status": "rejected",
            "contract_valid": False,
            "capability": None,
            "backend_resolution": {
                "requested_solver": None,
                "resolved_solver": None,
                "reason": "contract_load_failed",
            },
            "materials": [],
            "warnings": [],
            "failures": [failure.to_dict()],
            "estimated_work": {},
        }
    if mode == "sweep":
        from .sweep import expand_sweep
        from .task_io import simulation_task_from_dict

        try:
            expanded = expand_sweep(task)
            child_failures: list[dict[str, Any]] = []
            base_report: dict[str, Any] | None = None
            for child in expanded:
                simulation = simulation_task_from_dict(child["simulation"])
                report = preflight_task("simulate", simulation, registry)
                if base_report is None:
                    base_report = report
                if not report["ok"]:
                    child_failures.append(
                        {
                            "child_index": child["index"],
                            "parameters": child["parameters"],
                            "failures": report["failures"],
                        }
                    )
            assert base_report is not None
            flattened = [
                failure
                for child in child_failures
                for failure in child["failures"]
            ]
            try:
                base_simulation = simulation_task_from_dict(
                    task.base_simulation.model_dump(mode="python")
                )
                flattened.extend(
                    _metric_contract_failures(base_simulation, task.metrics)
                )
            except Exception as exc:
                flattened.append(failure_from_exception(exc).to_dict())
            return {
                **base_report,
                "ok": not flattened,
                "mode": "sweep",
                "status": "ready" if not flattened else "rejected",
                "failures": flattened,
                "estimated_work": {
                    **base_report.get("estimated_work", {}),
                    "sweep_child_count": len(expanded),
                    "total_forward_channel_spectra": len(expanded)
                    * int(base_report.get("estimated_work", {}).get("forward_channel_spectra", 0)),
                },
                "study": {
                    "schema_version": "sweep-task-v1",
                    "child_count": len(expanded),
                    "invalid_children": child_failures,
                },
            }
        except Exception as exc:
            failure = failure_from_exception(exc)
            return {
                "schema_version": PREFLIGHT_SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "ok": False,
                "operation": "preflight",
                "mode": "sweep",
                "status": "rejected",
                "contract_valid": False,
                "capability": None,
                "backend_resolution": {
                    "requested_solver": None,
                    "resolved_solver": None,
                    "reason": "study_contract_invalid",
                },
                "materials": [],
                "warnings": [],
                "failures": [failure.to_dict()],
                "estimated_work": {},
                "study": None,
            }
    if mode in {"sensitivity", "tolerance"}:
        from .task_io import simulation_task_from_dict

        try:
            simulation_contract = task.simulation
            simulation = simulation_task_from_dict(
                simulation_contract.model_dump(mode="python")
            )
            report = preflight_task("simulate", simulation, registry)
            study_failures: list[dict[str, Any]] = []
            study_failures.extend(
                _metric_contract_failures(
                    simulation,
                    [task.metric],
                    differentiable=mode == "sensitivity",
                )
            )
            if mode == "sensitivity":
                if simulation.stack.has_incoherent_layers:
                    study_failures.append(
                        failure_from_exception(
                            ValueError("sensitivity currently requires coherent layers")
                        ).to_dict()
                    )
                if task.metric.aggregation == "threshold_band_width":
                    study_failures.append(
                        failure_from_exception(
                            ValueError(
                                "threshold_band_width is non-differentiable and unsupported for sensitivity"
                            )
                        ).to_dict()
                    )
                if importlib.util.find_spec("torch") is None:
                    study_failures.append(
                        failure_from_exception(
                            RuntimeError("PyTorch is required for sensitivity analysis")
                        ).to_dict()
                    )
            else:
                for uncertainty in task.uncertainties:
                    if uncertainty.layer_index >= len(simulation.stack.layers):
                        study_failures.append(
                            failure_from_exception(
                                ValueError(
                                    f"uncertainty layer_index {uncertainty.layer_index} is outside the stack"
                                )
                            ).to_dict()
                        )
            failures = [*report.get("failures", []), *study_failures]
            report.update(
                {
                    "ok": not failures,
                    "mode": mode,
                    "status": "ready" if not failures else "rejected",
                    "failures": failures,
                    "study": {
                        "schema_version": f"{mode}-task-v1",
                        "sample_count": (
                            None if mode == "sensitivity" else int(task.sample_count)
                        ),
                        "differentiable_backend_required": mode == "sensitivity",
                    },
                }
            )
            if mode == "tolerance":
                report["estimated_work"]["tolerance_forward_simulations"] = int(
                    task.sample_count
                )
            return report
        except Exception as exc:
            failure = failure_from_exception(exc)
            return {
                "schema_version": PREFLIGHT_SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "ok": False,
                "operation": "preflight",
                "mode": mode,
                "status": "rejected",
                "contract_valid": False,
                "capability": None,
                "backend_resolution": {
                    "requested_solver": None,
                    "resolved_solver": None,
                    "reason": "study_contract_invalid",
                },
                "materials": [],
                "warnings": [],
                "failures": [failure.to_dict()],
                "estimated_work": {},
                "study": None,
            }
    return preflight_task(mode, task, registry)


__all__ = [
    "PREFLIGHT_SCHEMA_VERSION",
    "preflight_path",
    "preflight_task",
]
