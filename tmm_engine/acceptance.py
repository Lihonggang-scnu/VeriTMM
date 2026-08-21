"""Verifier-first execution and machine-readable physics certificates."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

import numpy as np

from .capabilities import (
    FailureCode,
    FailureRecord,
    assess_tmm_capability,
    enrich_failure_actions,
    failure_from_exception,
)
from .convergence import SpectralConvergenceSettings, audit_spectral_convergence
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


def _missing_requested_outputs(
    task: SimulationTask,
    result: ForwardSimulationResult,
) -> list[str]:
    """Return requested observables absent from the concrete solver result.

    Capability routing is the first defence.  This result-side check prevents
    an implementation or optional-backend regression from being certified as
    successful when an advertised output was silently omitted.
    """

    missing: list[str] = []
    channels = result.channels
    extras = result.extras
    for requested in task.requested_outputs:
        if requested in {"R", "T", "A"}:
            if not channels or any(requested not in values for values in channels.values()):
                missing.append(requested)
        elif requested == "system_emissivity":
            if not channels or any("E_system" not in values for values in channels.values()):
                missing.append(requested)
        elif requested == "amplitudes":
            if not channels or any(
                "r" not in values or "t" not in values for values in channels.values()
            ):
                missing.append(requested)
        elif requested == "phase_dispersion":
            if any(f"phase_dispersion|{key}" not in extras for key in channels):
                missing.append(requested)
        elif requested == "layer_absorption":
            if any(f"layer_absorption|{key}" not in extras for key in channels):
                missing.append(requested)
        elif requested == "ellipsometry":
            if any(
                f"ellipsometry|angle={float(angle):g}" not in extras
                for angle in task.illumination.angles_deg
            ):
                missing.append(requested)
    return sorted(set(missing))


def _compute_tightest_margin(
    physics_audit: Dict[str, Any],
    cross_solver: Dict[str, Any],
    settings: "AcceptanceSettings",
) -> Optional[Dict[str, Any]]:
    """Return the acceptance check closest to its threshold.

    ``normalized_margin = (threshold - observed) / threshold``; a value near
    zero means the result barely passed — the caller should treat it carefully.
    Returns ``None`` when no continuous check values are available (e.g. all
    checks were skipped or unavailable).
    """
    import math as _math

    checks: List[tuple] = []  # (check_name, observed, threshold)

    energy_err = physics_audit.get("energy_conservation_max_abs_error")
    if isinstance(energy_err, (int, float)) and _math.isfinite(float(energy_err)):
        checks.append(("energy_conservation", float(energy_err), settings.energy_tolerance))

    if cross_solver.get("status") in ("passed", "failed"):
        max_diff = cross_solver.get("maximum_absolute_difference")
        if isinstance(max_diff, (int, float)) and _math.isfinite(float(max_diff)):
            checks.append(
                ("cross_solver_agreement", float(max_diff), settings.cross_solver_tolerance)
            )

    if not checks:
        return None

    def _norm(observed: float, threshold: float) -> float:
        return (threshold - observed) / threshold if threshold > 0 else float("inf")

    name, observed, threshold = min(checks, key=lambda c: _norm(c[1], c[2]))
    margin = threshold - observed
    return {
        "check": name,
        "observed_value": observed,
        "acceptance_limit": threshold,
        "distance_to_limit": float(margin),
        "normalized_margin": float(_norm(observed, threshold)),
    }


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
        "task_hash_scope": "simulation_task_payload_without_operation_wrapper",
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
        missing_outputs = _missing_requested_outputs(task, result)
        if missing_outputs:
            failures.append(
                FailureRecord(
                    FailureCode.REQUESTED_OUTPUT_MISSING,
                    "The selected backend did not emit every requested observable.",
                    True,
                    context={
                        "requested_outputs": list(task.requested_outputs),
                        "missing_outputs": missing_outputs,
                        "solver": result.solver,
                    },
                )
            )
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

        tightest_margin = _compute_tightest_margin(
            result.audit, cross_solver, settings
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
            "tightest_margin": tightest_margin,
            "material_provenance_sha256": _stable_hash(result.material_provenance),
            "material_catalog": workbench.registry.catalog_status(),
            "runtime": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "numpy": np.__version__,
            },
            "failures": [enrich_failure_actions(item).to_dict() for item in failures],
        }
        certificate["certificate_id"] = _stable_hash(certificate)
        return CertifiedSimulation(result, certificate)
    except Exception as exc:
        failure = failure_from_exception(exc)
        base["failures"] = [failure.to_dict()]
        base["certificate_id"] = _stable_hash(base)
        return CertifiedSimulation(None, base)


__all__ = ["AcceptanceSettings", "CertifiedSimulation", "certify_simulation"]
