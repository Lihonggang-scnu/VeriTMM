"""Verifier-first execution and machine-readable physics certificates."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Optional

import numpy as np

from .capabilities import (
    FailureCode,
    FailureRecord,
    PhysicsEngineError,
    assess_tmm_capability,
)
from .convergence import SpectralConvergenceSettings, audit_spectral_convergence
from .material_registry import (
    MaterialAmbiguityError,
    MaterialNotFoundError,
    MaterialRangeError,
)
from .schemas import SimulationTask, SpectralGrid, dataclass_to_dict
from .workbench import ForwardSimulationResult, TMMWorkbench


@dataclass(frozen=True)
class AcceptanceSettings:
    require_spectral_convergence: bool = True
    require_independent_solver: bool = True
    cross_solver_tolerance: float = 1e-7
    energy_tolerance: float = 1e-7
    convergence: SpectralConvergenceSettings = field(default_factory=SpectralConvergenceSettings)


@dataclass
class CertifiedSimulation:
    result: Optional[ForwardSimulationResult]
    certificate: Dict[str, Any]


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cross_solver_check(
    workbench: TMMWorkbench,
    task: SimulationTask,
    primary: ForwardSimulationResult,
    tolerance: float,
) -> Dict[str, Any]:
    if task.stack.has_incoherent_layers:
        return {
            "status": "unavailable",
            "reason": "No second mixed-coherence implementation is currently registered.",
        }
    reference_solver = "smatrix" if primary.solver == "byrnes" else "byrnes"
    reference_task = replace(
        task,
        spectrum=SpectralGrid(values_nm=tuple(float(x) for x in primary.wavelengths_nm)),
        solver=reference_solver,
        requested_outputs=("R", "T", "A"),
    )
    reference = workbench.simulate(reference_task)
    maximum = 0.0
    per_channel: Dict[str, Any] = {}
    for channel_key, values in primary.channels.items():
        metrics: Dict[str, float] = {}
        for observable in ("R", "T", "A"):
            difference = float(
                np.max(
                    np.abs(
                        np.asarray(values[observable], dtype=np.float64)
                        - np.asarray(reference.channels[channel_key][observable], dtype=np.float64)
                    )
                )
            )
            metrics[observable] = difference
            maximum = max(maximum, difference)
        per_channel[channel_key] = metrics
    return {
        "status": "passed" if maximum <= tolerance else "failed",
        "primary_solver": primary.solver,
        "reference_solver": reference.solver,
        "maximum_absolute_difference": maximum,
        "tolerance": tolerance,
        "channels": per_channel,
    }


def _failure_from_exception(exc: Exception) -> FailureRecord:
    if isinstance(exc, PhysicsEngineError):
        return exc.failure
    if isinstance(exc, MaterialRangeError):
        return FailureRecord(
            FailureCode.MATERIAL_RANGE_ERROR,
            str(exc),
            True,
            context={
                "material": exc.material,
                "requested_range": exc.requested_range,
                "available_range": exc.available_range,
            },
        )
    if isinstance(exc, MaterialAmbiguityError):
        return FailureRecord(FailureCode.MATERIAL_AMBIGUITY, str(exc), True)
    if isinstance(exc, MaterialNotFoundError):
        return FailureRecord(FailureCode.MATERIAL_NOT_FOUND, str(exc), True)
    if isinstance(exc, ImportError):
        return FailureRecord(FailureCode.OPTIONAL_DEPENDENCY_MISSING, str(exc), True)
    return FailureRecord(FailureCode.INVALID_TASK, f"{type(exc).__name__}: {exc}", False)


def certify_simulation(
    workbench: TMMWorkbench,
    task: SimulationTask,
    settings: AcceptanceSettings | None = None,
) -> CertifiedSimulation:
    """Execute a task and let deterministic checks decide acceptance."""

    settings = settings or AcceptanceSettings()
    assessment = assess_tmm_capability(task)
    task_payload = dataclass_to_dict(task)
    base: Dict[str, Any] = {
        "schema_version": "physics-acceptance-certificate-v1",
        "task_sha256": _stable_hash(task_payload),
        "capability_assessment": assessment.to_dict(),
        "accepted": False,
        "status": "rejected_physics",
        "failures": [],
    }
    if not assessment.supported:
        base["failures"] = [item.to_dict() for item in assessment.failures]
        base["certificate_id"] = _stable_hash(base)
        return CertifiedSimulation(None, base)

    try:
        initial = workbench.simulate(task)
        if settings.require_spectral_convergence:
            convergence = audit_spectral_convergence(
                workbench, task, settings.convergence, initial_result=initial
            )
            result = convergence.final_result
            convergence_report = convergence.report_dict()
        else:
            result = initial
            convergence_report = {"status": "not_requested", "passed": None}

        failures: List[FailureRecord] = []
        audit = result.audit
        if int(audit.get("nonfinite_value_count", 0)):
            failures.append(
                FailureRecord(FailureCode.NUMERICAL_NONFINITE, "The solver returned non-finite observables.", False)
            )
        if not bool(audit.get("passivity_check_passed", False)):
            failures.append(
                FailureRecord(FailureCode.PASSIVITY_VIOLATION, "Passive-observable bounds were violated.", False)
            )
        if float(audit.get("energy_conservation_max_abs_error", float("inf"))) > settings.energy_tolerance:
            failures.append(
                FailureRecord(
                    FailureCode.ENERGY_CONSERVATION_FAILURE,
                    "Energy conservation exceeded the acceptance tolerance.",
                    False,
                    context={"tolerance": settings.energy_tolerance},
                )
            )
        if settings.require_spectral_convergence and not bool(convergence_report.get("passed")):
            failures.append(
                FailureRecord(
                    FailureCode.SPECTRAL_CONVERGENCE_FAILURE,
                    "The spectrum did not converge within the configured refinement budget.",
                    True,
                )
            )

        cross_solver = (
            _cross_solver_check(workbench, task, result, settings.cross_solver_tolerance)
            if settings.require_independent_solver
            else {"status": "not_requested"}
        )
        if cross_solver["status"] == "failed":
            failures.append(
                FailureRecord(
                    FailureCode.SOLVER_DISAGREEMENT,
                    "The primary and reference solvers disagree beyond tolerance.",
                    True,
                    context={"maximum_absolute_difference": cross_solver["maximum_absolute_difference"]},
                )
            )

        accepted = not failures
        limited = bool(settings.require_independent_solver and cross_solver["status"] == "unavailable")
        certificate = {
            **base,
            "accepted": accepted,
            "status": (
                "physically_valid_with_limits"
                if accepted and limited
                else ("physically_valid" if accepted else "rejected_physics")
            ),
            "solver": result.solver,
            "physics_audit": result.audit,
            "spectral_convergence": convergence_report,
            "independent_solver_check": cross_solver,
            "material_provenance_sha256": _stable_hash(result.material_provenance),
            "material_catalog": workbench.registry.catalog_status(),
            "runtime": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "numpy": np.__version__,
            },
            "failures": [item.to_dict() for item in failures],
        }
        certificate["certificate_id"] = _stable_hash(certificate)
        return CertifiedSimulation(result, certificate)
    except Exception as exc:
        failure = _failure_from_exception(exc)
        base["failures"] = [failure.to_dict()]
        base["certificate_id"] = _stable_hash(base)
        return CertifiedSimulation(None, base)


__all__ = ["AcceptanceSettings", "CertifiedSimulation", "certify_simulation"]
