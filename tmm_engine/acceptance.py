"""Verifier-first execution and machine-readable physics certificates."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

import numpy as np

from ._version import __version__
from .capabilities import (
    FailureCode,
    FailureRecord,
    assess_tmm_capability,
    enrich_failure_actions,
    failure_from_exception,
)
from .convergence import SpectralConvergenceSettings, audit_spectral_convergence
from .protocol.evidence import from_certificate
from .protocol.uncertainty_budget import (
    UncertaintyBudget,
    applicability_gaps_from_certificate,
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


def _evidence_coverage_payload(certificate: Dict[str, Any]) -> dict[str, str]:
    """Return JSON-ready additive evidence metadata for a certificate."""

    return from_certificate(certificate).model_dump(mode="json")


def attach_uncertainty_budget(
    certificate: Dict[str, Any],
    budget: UncertaintyBudget,
) -> Dict[str, Any]:
    """Attach a budget and recompute additive coverage and certificate identity."""

    certificate["uncertainty_budget"] = budget.model_dump(mode="json")
    certificate.pop("certificate_id", None)
    certificate["evidence_coverage"] = _evidence_coverage_payload(certificate)
    certificate["certificate_id"] = _stable_hash(certificate)
    return certificate


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
    offending_channel: Optional[str] = None
    offending_observable: Optional[str] = None
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
            if difference > maximum:
                maximum = difference
                offending_channel = channel_key
                offending_observable = observable
        per_channel[channel_key] = metrics
    return {
        "status": "passed" if maximum <= tolerance else "failed",
        "primary_solver": primary.solver,
        "reference_solver": reference.solver,
        "maximum_absolute_difference": maximum,
        "tolerance": tolerance,
        "channels": per_channel,
        "offending_channel": offending_channel,
        "offending_observable": offending_observable,
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
    result_dict: Dict[str, Any] = {
        "check": name,
        "observed_value": observed,
        "acceptance_limit": threshold,
        "distance_to_limit": float(margin),
        "normalized_margin": float(_norm(observed, threshold)),
    }
    # Attach worst-case location so the High-Precision Referee can target the
    # right channel/wavelength without guessing.
    if name == "energy_conservation":
        result_dict["worst_case_channel"] = physics_audit.get("energy_worst_case_channel")
        result_dict["worst_case_wavelength_nm"] = physics_audit.get("energy_worst_case_wavelength_nm")
        result_dict["worst_case_wavelength_idx"] = physics_audit.get("energy_worst_case_wavelength_idx")
    elif name == "cross_solver_agreement":
        result_dict["worst_case_channel"] = cross_solver.get("offending_channel")
        result_dict["worst_case_observable"] = cross_solver.get("offending_observable")
    return result_dict


def _maybe_run_referee(
    *,
    workbench: "TMMWorkbench",
    task: SimulationTask,
    result: Any,
    cross_solver: Dict[str, Any],
    tightest_margin: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return a high-precision referee report or None if not triggered."""
    import re as _re

    from .high_precision import TRIGGER_THRESHOLD, is_available, run_referee

    # Determine whether to trigger
    solver_disagreement = cross_solver.get("status") == "failed"
    barely_passed = (
        tightest_margin is not None
        and float(tightest_margin.get("normalized_margin", 1.0)) < TRIGGER_THRESHOLD
    )
    if not (solver_disagreement or barely_passed):
        return {"status": "not_triggered", "reason": "all checks comfortably passed"}

    if not is_available():
        return {
            "status": "unavailable",
            "reason": "mpmath not installed; install with: pip install mpmath>=1.3",
        }

    illumination = task.illumination

    # Select the channel to referee based on what triggered the check.
    # Solver disagreement → channel with the largest cross-solver difference.
    # Energy margin       → channel with the worst energy conservation error.
    # Cross-solver margin → channel with the largest cross-solver difference.
    if solver_disagreement:
        offending_channel = cross_solver.get("offending_channel")
    elif tightest_margin is not None and tightest_margin.get("check") == "energy_conservation":
        offending_channel = result.audit.get("energy_worst_case_channel")
    else:
        offending_channel = cross_solver.get("offending_channel")
    if offending_channel and offending_channel in result.channels:
        channel_key = offending_channel
        m = _re.match(r"angle=([^|]+)\|pol=(.+)$", channel_key)
        if m:
            angle_deg = float(m.group(1))
            pol = m.group(2)
            if pol == "unpolarized":
                pol = "s"
        else:
            # Unexpected key format — fall back to first channel
            angle_deg = float(illumination.angles_deg[0])
            pol = str(illumination.polarizations[0])
            if pol == "unpolarized":
                pol = "s"
            channel_key = "angle=%g|pol=%s" % (angle_deg, pol)
    else:
        angle_deg = float(illumination.angles_deg[0])
        pol = str(illumination.polarizations[0])
        if pol == "unpolarized":
            pol = "s"
        channel_key = "angle=%g|pol=%s" % (angle_deg, pol)

    if channel_key not in result.channels:
        return {"status": "error", "reason": "channel not found: %s" % channel_key}

    # P0-1: resolve dispersive nk arrays — shape (N_wavelengths,) per medium
    try:
        media_list, wavelengths_nm, _ = workbench._resolve_stack(task)
    except Exception as exc:
        return {"status": "error", "reason": "stack resolution failed: %s" % exc}

    d_nm = [float(layer.thickness_nm) for layer in task.stack.layers]
    ch = result.channels[channel_key]
    primary_R = list(np.asarray(ch.get("R", []), dtype=np.float64))
    primary_T = list(np.asarray(ch.get("T", []), dtype=np.float64))

    # P0-2: re-run the reference solver to obtain secondary R/T for closer_solver
    secondary_R: Optional[list] = None
    secondary_T: Optional[list] = None
    reference_solver_name = cross_solver.get("reference_solver")
    if reference_solver_name and not task.stack.has_incoherent_layers:
        try:
            ref_task = replace(
                task,
                spectrum=SpectralGrid(
                    values_nm=tuple(float(w) for w in wavelengths_nm)
                ),
                solver=reference_solver_name,
                requested_outputs=("R", "T"),
            )
            ref_result = workbench.simulate(ref_task)
            if channel_key in ref_result.channels:
                ref_ch = ref_result.channels[channel_key]
                secondary_R = list(np.asarray(ref_ch.get("R", []), dtype=np.float64))
                secondary_T = list(np.asarray(ref_ch.get("T", []), dtype=np.float64))
        except Exception:
            pass  # secondary comparison is best-effort; don't fail the referee

    report = run_referee(
        n_by_wavelength=media_list,
        d_nm=d_nm,
        wavelengths_nm=wavelengths_nm.tolist(),
        angle_deg=angle_deg,
        polarization=pol,
        primary_R=primary_R,
        primary_T=primary_T,
        secondary_R=secondary_R,
        secondary_T=secondary_T,
    )
    report["triggered_by"] = (
        "solver_disagreement" if solver_disagreement else "tightest_margin"
    )
    report["channel"] = channel_key
    return report


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
        "veritmm_version": __version__,
        "capability_assessment": assessment.to_dict(),
        "accepted": False,
        "status": "rejected_physics",
        "failures": [],
    }
    if not assessment.supported:
        base["failures"] = [item.to_dict() for item in assessment.failures]
        applicability_gaps = applicability_gaps_from_certificate(base)
        if applicability_gaps:
            base["uncertainty_budget"] = UncertaintyBudget(
                applicability_gaps=applicability_gaps
            ).model_dump(mode="json")
        base["evidence_coverage"] = _evidence_coverage_payload(base)
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

        # ------------------------------------------------------------------
        # High-precision referee (optional, purely informational)
        # Triggered when: solvers disagree OR result barely passed.
        # Result is added to the certificate but NEVER changes ``accepted``.
        # ------------------------------------------------------------------
        high_precision_referee = _maybe_run_referee(
            workbench=workbench,
            task=task,
            result=result,
            cross_solver=cross_solver,
            tightest_margin=tightest_margin,
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
            "high_precision_referee": high_precision_referee,
            "material_provenance_sha256": _stable_hash(result.material_provenance),
            "material_catalog": workbench.registry.catalog_status(),
            "runtime": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "numpy": np.__version__,
            },
            "failures": [enrich_failure_actions(item).to_dict() for item in failures],
        }
        certificate["evidence_coverage"] = _evidence_coverage_payload(certificate)
        certificate["certificate_id"] = _stable_hash(certificate)
        return CertifiedSimulation(result, certificate)
    except Exception as exc:
        failure = failure_from_exception(exc)
        base["failures"] = [failure.to_dict()]
        base["evidence_coverage"] = _evidence_coverage_payload(base)
        base["certificate_id"] = _stable_hash(base)
        return CertifiedSimulation(None, base)


__all__ = [
    "AcceptanceSettings",
    "CertifiedSimulation",
    "attach_uncertainty_budget",
    "certify_simulation",
]
