"""Verifier-first execution and machine-readable physics certificates."""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field, replace
from dataclasses import fields as dataclass_fields
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from ._version import __version__
from .capabilities import (
    CapabilityAssessment,
    FailureCode,
    FailureRecord,
    assess_tmm_capability,
    enrich_failure_actions,
    failure_from_exception,
)
from .convergence import SpectralConvergenceSettings, audit_spectral_convergence
from .hashing import IDENTITY_SCHEME, stable_sha256
from .protocol.evidence import from_certificate
from .protocol.uncertainty_budget import (
    UncertaintyBudget,
    applicability_gaps_from_certificate,
)
from .reciprocity import check_reciprocity
from .schemas import SimulationTask, SpectralGrid, dataclass_to_dict
from .workbench import ForwardSimulationResult, TMMWorkbench


@dataclass(frozen=True)
class AcceptanceSettings:
    require_spectral_convergence: bool = True
    require_independent_solver: bool = True
    require_reciprocity: bool = False
    cross_solver_tolerance: float = 1e-7
    energy_tolerance: float = 1e-7
    convergence: SpectralConvergenceSettings = field(default_factory=SpectralConvergenceSettings)


VERIFIER_DEPENDENCY_GRAPH: Dict[str, Dict[str, Any]] = {
    "energy_conservation": {
        "depends_on": [
            "solver.R",
            "solver.T",
            "independent_absorption.layer_integral",
        ],
        "forbidden_dependency": ["A_closure"],
    },
    "cross_solver_agreement": {
        "depends_on": [
            "solver.R",
            "solver.T",
            "independent_absorption.layer_integral",
            "reference_solver.R",
            "reference_solver.T",
            "reference_solver.independent_absorption.layer_integral",
        ],
        "forbidden_dependency": ["A_closure"],
    },
}
"""Static declaration of what each acceptance check is allowed to consume.

``depends_on`` names the raw inputs a check's verdict is derived from;
``forbidden_dependency`` names quantities the check must never read — the
energy checks judge recorded residuals and must never rebuild absorption from
the closure (``A = 1 - R - T``), which would make the identity they test
tautological.  ``tests/test_energy_accounting_certificate.py`` enforces this
mechanically.
"""


ENERGY_FALLBACK_REASON_UNAVAILABLE = (
    "independent layer-absorption integral unavailable for this run; the "
    "energy verdict was derived from the closure ledger alone"
)
ENERGY_FALLBACK_REASON_PRE_LEDGER = (
    "evidence predates the independent absorption ledger; the energy verdict "
    "was derived from the closure ledger alone"
)


@dataclass
class CertifiedSimulation:
    result: Optional[ForwardSimulationResult]
    certificate: Dict[str, Any]
    evidence: Optional["VerificationEvidence"] = None


class VerificationArtifactError(ValueError):
    """Raised when a persisted verification evidence or policy artifact is
    malformed, bound to a different identity, or from an unsupported schema."""


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
    certificate["certificate_id"] = stable_sha256(certificate)
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

    primary_independent = primary.independent_absorption
    reference_independent = reference.independent_absorption
    independent_available = bool(primary.channels) and (
        set(primary_independent) == set(primary.channels)
        and set(reference_independent) == set(reference.channels)
        and all(value is not None for value in primary_independent.values())
        and all(value is not None for value in reference_independent.values())
    )

    # Agreement is reported per ledger instead of folding the reference's
    # absorption into the closure: R, T, the two independently integrated
    # absorptions, and the full energy accounting must each agree.
    observables = ("R", "T", "A_independent", "energy_accounting")
    metric_maxima: Dict[str, float] = {name: 0.0 for name in observables}
    metric_channels: Dict[str, Optional[str]] = {name: None for name in observables}
    metric_indexes: Dict[str, int] = {name: 0 for name in observables}
    per_channel: Dict[str, Any] = {}
    for channel_key, values in primary.channels.items():
        reference_values = reference.channels[channel_key]
        primary_r = np.asarray(values["R"], dtype=np.float64)
        primary_t = np.asarray(values["T"], dtype=np.float64)
        reference_r = np.asarray(reference_values["R"], dtype=np.float64)
        reference_t = np.asarray(reference_values["T"], dtype=np.float64)
        diff_arrays: Dict[str, np.ndarray] = {
            "R": primary_r - reference_r,
            "T": primary_t - reference_t,
        }
        if independent_available:
            primary_a = np.asarray(primary_independent[channel_key], dtype=np.float64)
            reference_a = np.asarray(reference_independent[channel_key], dtype=np.float64)
            diff_arrays["A_independent"] = primary_a - reference_a
            diff_arrays["energy_accounting"] = (
                1.0 - primary_r - primary_t - primary_a
            ) - (1.0 - reference_r - reference_t - reference_a)
        diffs: Dict[str, Optional[float]] = {
            name: (float(np.max(np.abs(diff_arrays[name]))) if name in diff_arrays else None)
            for name in observables
        }
        for name, difference in diffs.items():
            if difference is None:
                continue
            if difference > metric_maxima[name]:
                metric_maxima[name] = difference
                metric_channels[name] = channel_key
                metric_indexes[name] = int(np.argmax(np.abs(diff_arrays[name])))
        per_channel[channel_key] = diffs

    metrics: Dict[str, Any] = {}
    for name in observables:
        if not independent_available and name in ("A_independent", "energy_accounting"):
            metrics[name] = {
                "status": "unavailable",
                "reason": (
                    "the independent layer-absorption integral is unavailable on "
                    "at least one side of the comparison"
                ),
            }
            continue
        metrics[name] = {
            "maximum_absolute_difference": metric_maxima[name],
            "tolerance": tolerance,
            "status": "passed" if metric_maxima[name] <= tolerance else "failed",
        }

    maximum = 0.0
    offending_channel: Optional[str] = None
    offending_observable: Optional[str] = None
    for name in observables:
        metric = metrics[name]
        if metric.get("status") == "unavailable":
            continue
        if metric["maximum_absolute_difference"] > maximum or offending_channel is None:
            maximum = metric["maximum_absolute_difference"]
            offending_channel = metric_channels[name]
            offending_observable = name
    status = "passed" if all(
        metric.get("status") != "failed" for metric in metrics.values()
    ) else "failed"
    worst_wavelength: Optional[float] = None
    if offending_channel is not None and offending_observable is not None:
        index = metric_indexes.get(offending_observable, 0)
        if 0 <= index < len(primary.wavelengths_nm):
            worst_wavelength = float(primary.wavelengths_nm[index])
    return {
        "status": status,
        "primary_solver": primary.solver,
        "reference_solver": reference.solver,
        "metrics": metrics,
        "maximum_absolute_difference": maximum,
        "tolerance": tolerance,
        "channels": per_channel,
        "offending_channel": offending_channel,
        "offending_observable": offending_observable,
        "offending_wavelength_nm": worst_wavelength,
    }


def _missing_requested_outputs_from_evidence(
    requested_outputs: List[str],
    illumination_angles_deg: List[float],
    channel_output_keys: Dict[str, List[str]],
    extras_keys: List[str],
) -> list[str]:
    """Return requested observables absent from the recorded solver outputs.

    Consumes only the raw output-key inventory carried by the verification
    evidence: capability routing is the first defence, and this result-side
    check prevents an implementation or optional-backend regression from being
    certified as successful when an advertised output was silently omitted.
    """

    missing: list[str] = []
    extras = set(extras_keys)
    channels = channel_output_keys
    for requested in requested_outputs:
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
                for angle in illumination_angles_deg
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


EVIDENCE_SCHEMA_VERSION = "veritmm-verification-evidence-v1"
VERIFICATION_POLICY_SCHEMA_VERSION = "veritmm-verification-policy-v1"

_POLICY_TOP_LEVEL_FIELDS = (
    "schema_version",
    "identity_scheme",
    "require_spectral_convergence",
    "require_independent_solver",
    "require_reciprocity",
    "cross_solver_tolerance",
    "energy_tolerance",
    "convergence",
)


def verification_policy_dict(settings: AcceptanceSettings) -> Dict[str, Any]:
    """Serialize the acceptance policy actually used for one certificate.

    Evidence is what was observed; this policy artifact is the judgement rule
    that was applied to it.  Persisting both keeps ``evaluate_evidence``
    replayable even when future releases change default tolerances.
    """

    return {
        "schema_version": VERIFICATION_POLICY_SCHEMA_VERSION,
        "identity_scheme": IDENTITY_SCHEME,
        "require_spectral_convergence": settings.require_spectral_convergence,
        "require_independent_solver": settings.require_independent_solver,
        "require_reciprocity": settings.require_reciprocity,
        "cross_solver_tolerance": settings.cross_solver_tolerance,
        "energy_tolerance": settings.energy_tolerance,
        "convergence": {
            item.name: getattr(settings.convergence, item.name)
            for item in dataclass_fields(SpectralConvergenceSettings)
        },
    }


def _checked_policy_flag(payload: Mapping[str, Any], name: str) -> bool:
    value = payload.get(name)
    if not isinstance(value, bool):
        raise VerificationArtifactError(
            f"verification policy field {name!r} must be a boolean"
        )
    return value


def _checked_policy_number(
    payload: Mapping[str, Any], name: str
) -> float:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VerificationArtifactError(
            f"verification policy field {name!r} must be a finite number"
        )
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        raise VerificationArtifactError(
            f"verification policy field {name!r} must be finite"
        )
    return value


def verification_policy_from_dict(payload: Mapping[str, Any]) -> AcceptanceSettings:
    """Rebuild acceptance settings from a persisted policy artifact.

    Parsing is strict: unknown fields, a foreign ``schema_version`` or
    ``identity_scheme``, and mistyped values all fail closed so a tampered or
    foreign policy can never silently re-judge evidence.
    """

    if not isinstance(payload, Mapping):
        raise VerificationArtifactError("verification policy must be a JSON object")
    if payload.get("schema_version") != VERIFICATION_POLICY_SCHEMA_VERSION:
        raise VerificationArtifactError(
            "verification policy has an unsupported schema_version: "
            f"{payload.get('schema_version')!r}"
        )
    if payload.get("identity_scheme") != IDENTITY_SCHEME:
        raise VerificationArtifactError(
            "verification policy identity_scheme mismatch: "
            f"{payload.get('identity_scheme')!r}"
        )
    unknown = sorted(set(payload) - set(_POLICY_TOP_LEVEL_FIELDS))
    if unknown:
        raise VerificationArtifactError(
            f"verification policy has unknown fields: {unknown}"
        )
    convergence_payload = payload.get("convergence")
    if not isinstance(convergence_payload, Mapping):
        raise VerificationArtifactError(
            "verification policy convergence settings must be a JSON object"
        )
    convergence_kwargs: Dict[str, Any] = {}
    for item in dataclass_fields(SpectralConvergenceSettings):
        if item.name not in convergence_payload:
            raise VerificationArtifactError(
                f"verification policy convergence is missing {item.name!r}"
            )
        raw = convergence_payload[item.name]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise VerificationArtifactError(
                f"verification policy convergence field {item.name!r} "
                "must be a finite number"
            )
        cast = type(item.default) if isinstance(item.default, int) else float
        convergence_kwargs[item.name] = cast(raw)
    extra_convergence = sorted(
        set(convergence_payload) - {item.name for item in dataclass_fields(SpectralConvergenceSettings)}
    )
    if extra_convergence:
        raise VerificationArtifactError(
            f"verification policy convergence has unknown fields: {extra_convergence}"
        )
    # Opt-in switch added after the policy schema was frozen: a policy written
    # before it existed loads as ``False`` so historical run directories keep
    # replaying; when present it must still be a real boolean.
    raw_reciprocity = payload.get("require_reciprocity", False)
    if not isinstance(raw_reciprocity, bool):
        raise VerificationArtifactError(
            "verification policy field 'require_reciprocity' must be a boolean"
        )
    return AcceptanceSettings(
        require_spectral_convergence=_checked_policy_flag(
            payload, "require_spectral_convergence"
        ),
        require_independent_solver=_checked_policy_flag(
            payload, "require_independent_solver"
        ),
        require_reciprocity=raw_reciprocity,
        cross_solver_tolerance=_checked_policy_number(payload, "cross_solver_tolerance"),
        energy_tolerance=_checked_policy_number(payload, "energy_tolerance"),
        convergence=SpectralConvergenceSettings(**convergence_kwargs),
    )


@dataclass(frozen=True)
class VerificationEvidence:
    """Everything the acceptance decision consumes — and nothing it decides.

    Fields hold the raw decision inputs produced by solvers and audits
    (deviations, counts, reports, provenance) plus the policy-independent
    identity of the task.  Final verdicts (``accepted``/``status``/
    ``certificate_id``) are deliberately absent: ``evaluate_evidence()``
    recomputes them, so a replay re-judges the recorded evidence instead of
    re-reading an old verdict.  Where a check uses a tolerance, the evidence
    carries the observed quantity and the acceptance settings carry the
    tolerance, so judgement cannot be smuggled into collection.
    """

    schema_version: str
    task_payload: Dict[str, Any]
    task_sha256: str
    veritmm_version: str
    capability_assessment: Dict[str, Any]
    capability_failure_dicts: List[Dict[str, Any]]
    runtime: Dict[str, Any]
    requested_outputs: List[str]
    illumination_angles_deg: List[float]
    channel_output_keys: Dict[str, List[str]]
    extras_keys: List[str]
    identity_scheme: Optional[str] = None
    run_id: Optional[str] = None
    solver_name: Optional[str] = None
    physics_audit: Optional[Dict[str, Any]] = None
    material_provenance: Optional[Any] = None
    material_catalog: Optional[Dict[str, Any]] = None
    spectral_convergence: Optional[Dict[str, Any]] = None
    independent_solver_check: Optional[Dict[str, Any]] = None
    tightest_margin: Optional[Dict[str, Any]] = None
    high_precision_referee: Optional[Dict[str, Any]] = None
    execution_error: Optional[Dict[str, Any]] = None
    energy_accounting: Optional[Dict[str, Any]] = None
    """Raw independent-energy ledger (independence class + worst-case
    observable summaries).  Optional: evidence written before this ledger
    existed loads as ``None`` and is judged on the closure residual alone."""
    reciprocity_check: Optional[Dict[str, Any]] = None
    """Lorentz reciprocity report; populated only when the acceptance policy
    requested it (``require_reciprocity``).  Optional for the same reason as
    ``energy_accounting``."""

    def to_dict(self) -> Dict[str, Any]:
        """Return the JSON-ready evidence payload."""

        return {
            "schema_version": self.schema_version,
            "task_payload": self.task_payload,
            "task_sha256": self.task_sha256,
            "veritmm_version": self.veritmm_version,
            "capability_assessment": self.capability_assessment,
            "capability_failure_dicts": self.capability_failure_dicts,
            "runtime": self.runtime,
            "requested_outputs": self.requested_outputs,
            "illumination_angles_deg": self.illumination_angles_deg,
            "channel_output_keys": self.channel_output_keys,
            "extras_keys": self.extras_keys,
            "identity_scheme": self.identity_scheme,
            "run_id": self.run_id,
            "solver_name": self.solver_name,
            "physics_audit": self.physics_audit,
            "material_provenance": self.material_provenance,
            "material_catalog": self.material_catalog,
            "spectral_convergence": self.spectral_convergence,
            "independent_solver_check": self.independent_solver_check,
            "tightest_margin": self.tightest_margin,
            "high_precision_referee": self.high_precision_referee,
            "execution_error": self.execution_error,
            "energy_accounting": self.energy_accounting,
            "reciprocity_check": self.reciprocity_check,
        }


def _certificate_base(
    task_payload: Dict[str, Any],
    capability_assessment: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": "physics-acceptance-certificate-v1",
        "task_sha256": stable_sha256(task_payload),
        "task_hash_scope": "simulation_task_payload_without_operation_wrapper",
        "veritmm_version": __version__,
        "capability_assessment": capability_assessment,
        "accepted": False,
        "status": "rejected_physics",
        "failures": [],
    }


def _observable_worst_case(result: ForwardSimulationResult) -> Dict[str, Any]:
    """Worst-case value and location of each energy ledger observable.

    Records what the energy verdict consumes — maxima over all channels and
    wavelengths with their location — never the full spectra.
    """

    wavelengths = result.wavelengths_nm
    summary: Dict[str, Any] = {}
    for name, key in (("R", "R"), ("T", "T"), ("A_closure", "A")):
        worst_value: Optional[float] = None
        worst_channel: Optional[str] = None
        worst_index = 0
        for channel_key, values in result.channels.items():
            array = np.asarray(values[key], dtype=np.float64)
            index = int(np.argmax(array))
            value = float(array[index])
            if worst_value is None or value > worst_value:
                worst_value = value
                worst_channel = channel_key
                worst_index = index
        summary[name] = {
            "maximum": worst_value,
            "channel": worst_channel,
            "wavelength_nm": (
                float(wavelengths[worst_index])
                if worst_channel is not None and wavelengths is not None
                else None
            ),
        }
    independent = result.independent_absorption
    independent_complete = bool(result.channels) and (
        set(independent) == set(result.channels)
        and all(value is not None for value in independent.values())
    )
    if not independent_complete:
        summary["A_independent"] = None
        return summary
    worst_value = None
    worst_channel = None
    worst_index = 0
    for channel_key, array in independent.items():
        values = np.asarray(array, dtype=np.float64)
        index = int(np.argmax(values))
        value = float(values[index])
        if worst_value is None or value > worst_value:
            worst_value = value
            worst_channel = channel_key
            worst_index = index
    summary["A_independent"] = {
        "maximum": worst_value,
        "channel": worst_channel,
        "wavelength_nm": (
            float(wavelengths[worst_index])
            if worst_channel is not None and wavelengths is not None
            else None
        ),
    }
    return summary


def collect_verification_evidence(
    workbench: TMMWorkbench,
    task: SimulationTask,
    settings: AcceptanceSettings,
    assessment: Optional[CapabilityAssessment] = None,
) -> tuple[VerificationEvidence, Optional[ForwardSimulationResult]]:
    """Run the solver and audits, recording what a decision will consume.

    This is the evidence-producing half of acceptance.  It executes the task,
    the configured convergence audit, the independent-solver comparison, and
    the (never decisive) high-precision referee, and returns the resulting
    :class:`VerificationEvidence` together with the live solver result.  It
    never decides ``accepted`` and never builds a certificate.
    """

    if assessment is None:
        assessment = assess_tmm_capability(task)
    task_payload = dataclass_to_dict(task)
    runtime = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    base_evidence = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "task_payload": task_payload,
        "task_sha256": stable_sha256(task_payload),
        "veritmm_version": __version__,
        "identity_scheme": IDENTITY_SCHEME,
        "capability_assessment": assessment.to_dict(),
        "capability_failure_dicts": [item.to_dict() for item in assessment.failures],
        "runtime": runtime,
        "requested_outputs": list(task.requested_outputs),
        "illumination_angles_deg": [
            float(angle) for angle in task.illumination.angles_deg
        ],
    }
    if not assessment.supported:
        return (
            VerificationEvidence(
                **base_evidence,
                channel_output_keys={},
                extras_keys=[],
            ),
            None,
        )

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

    cross_solver = (
        _cross_solver_check(workbench, task, result, settings.cross_solver_tolerance)
        if settings.require_independent_solver
        else {"status": "not_requested"}
    )
    reciprocity_report = (
        check_reciprocity(
            workbench,
            task.stack,
            np.asarray(task.spectrum.wavelengths_nm(), dtype=np.float64),
            angles_deg=task.illumination.angles_deg,
            polarizations=task.illumination.polarizations,
            solver=task.solver,
        )
        if settings.require_reciprocity
        else None
    )
    tightest_margin = _compute_tightest_margin(result.audit, cross_solver, settings)

    # ------------------------------------------------------------------
    # High-precision referee (optional, purely informational)
    # Triggered when: solvers disagree OR result barely passed.
    # Result is recorded as evidence but NEVER changes ``accepted``.
    # ------------------------------------------------------------------
    high_precision_referee = _maybe_run_referee(
        workbench=workbench,
        task=task,
        result=result,
        cross_solver=cross_solver,
        tightest_margin=tightest_margin,
    )

    evidence = VerificationEvidence(
        **base_evidence,
        channel_output_keys={
            key: sorted(values.keys()) for key, values in result.channels.items()
        },
        extras_keys=sorted(result.extras.keys()),
        solver_name=result.solver,
        physics_audit=result.audit,
        material_provenance=result.material_provenance,
        material_catalog=workbench.registry.catalog_status(),
        spectral_convergence=convergence_report,
        independent_solver_check=cross_solver,
        tightest_margin=tightest_margin,
        high_precision_referee=high_precision_referee,
        reciprocity_check=reciprocity_report,
        energy_accounting={
            "independence": dict(result.independence_audit),
            "observables": _observable_worst_case(result),
        },
    )
    return evidence, result


def _energy_accounting_certificate_block(
    evidence: "VerificationEvidence",
) -> Optional[Dict[str, Any]]:
    """Summarize the energy ledger for the certificate (no full spectra)."""

    audit = evidence.physics_audit or {}
    if not audit:
        return None
    ledger = evidence.energy_accounting
    independence = (ledger or {}).get("independence") or {}
    observables = (ledger or {}).get("observables") or {}
    residual_independent = independence.get("energy_independent_max_abs_error")
    available = residual_independent is not None
    block: Dict[str, Any] = {
        "method": "layer_absorption_integral",
        "independence_class": (
            "layer_absorption_integral" if available else "derived_closure_fallback"
        ),
        "R": observables.get("R"),
        "T": observables.get("T"),
        "A_closure": observables.get("A_closure"),
        "A_independent": observables.get("A_independent"),
        "residual_closure": audit.get("energy_conservation_max_abs_error"),
        "residual_independent": residual_independent,
        "worst_case": {
            "closure_channel": audit.get("energy_worst_case_channel"),
            "closure_wavelength_nm": audit.get("energy_worst_case_wavelength_nm"),
            "independent_channel": independence.get("energy_independent_worst_channel"),
            "independent_wavelength_nm": independence.get(
                "energy_independent_worst_wavelength_nm"
            ),
        },
    }
    if not available:
        block["fallback_reason"] = (
            ENERGY_FALLBACK_REASON_PRE_LEDGER
            if ledger is None
            else ENERGY_FALLBACK_REASON_UNAVAILABLE
        )
    return block


def evaluate_evidence(
    evidence: VerificationEvidence,
    settings: AcceptanceSettings | None = None,
) -> Dict[str, Any]:
    """Recompute the certificate from recorded evidence and policy settings.

    This is the decision half of acceptance: a pure function of the evidence
    and the acceptance settings.  It performs no solver work, so the same
    evidence always yields the same certificate — including the same
    ``certificate_id``.
    """

    settings = settings or AcceptanceSettings()
    base = _certificate_base(evidence.task_payload, evidence.capability_assessment)

    if evidence.execution_error is not None:
        base["failures"] = [evidence.execution_error]
        base["evidence_coverage"] = _evidence_coverage_payload(base)
        base["certificate_id"] = stable_sha256(base)
        return base

    if not evidence.capability_assessment.get("supported"):
        base["failures"] = [dict(item) for item in evidence.capability_failure_dicts]
        applicability_gaps = applicability_gaps_from_certificate(base)
        if applicability_gaps:
            base["uncertainty_budget"] = UncertaintyBudget(
                applicability_gaps=applicability_gaps
            ).model_dump(mode="json")
        base["evidence_coverage"] = _evidence_coverage_payload(base)
        base["certificate_id"] = stable_sha256(base)
        return base

    failures: List[FailureRecord] = []
    audit = evidence.physics_audit or {}
    missing_outputs = _missing_requested_outputs_from_evidence(
        evidence.requested_outputs,
        evidence.illumination_angles_deg,
        evidence.channel_output_keys,
        evidence.extras_keys,
    )
    if missing_outputs:
        failures.append(
            FailureRecord(
                FailureCode.REQUESTED_OUTPUT_MISSING,
                "The selected backend did not emit every requested observable.",
                True,
                context={
                    "requested_outputs": list(evidence.requested_outputs),
                    "missing_outputs": missing_outputs,
                    "solver": evidence.solver_name,
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
    residual_closure = float(audit.get("energy_conservation_max_abs_error", float("inf")))
    ledger = evidence.energy_accounting or {}
    independence = ledger.get("independence") or {}
    residual_independent = independence.get("energy_independent_max_abs_error")
    energy_context: Dict[str, Any] = {"tolerance": settings.energy_tolerance}
    energy_failed = False
    if ledger:
        if residual_independent is None:
            # Declared fallback: the independent ledger could not be produced
            # for this run.  The verdict degrades to the closure residual and
            # says so instead of pretending the independent check ran.
            energy_context["independence_class"] = "derived_closure_fallback"
            energy_context["fallback_reason"] = ENERGY_FALLBACK_REASON_UNAVAILABLE
            energy_failed = residual_closure > settings.energy_tolerance
            if energy_failed:
                energy_context["triggered_by"] = "closure"
        else:
            exceeds_closure = residual_closure > settings.energy_tolerance
            exceeds_independent = float(residual_independent) > settings.energy_tolerance
            energy_failed = exceeds_closure or exceeds_independent
            if energy_failed:
                energy_context["independence_class"] = "layer_absorption_integral"
                energy_context["triggered_by"] = (
                    "independent" if exceeds_independent else "closure"
                )
                energy_context["residual_closure"] = residual_closure
                energy_context["residual_independent"] = float(residual_independent)
    else:
        # Pre-ledger evidence: historical closure-only judgement.
        energy_context["ledger"] = "closure_only"
        energy_failed = residual_closure > settings.energy_tolerance
        if energy_failed:
            energy_context["triggered_by"] = "closure"
    if energy_failed:
        failures.append(
            FailureRecord(
                FailureCode.ENERGY_CONSERVATION_FAILURE,
                "Energy conservation exceeded the acceptance tolerance.",
                False,
                context=energy_context,
            )
        )

    convergence_report = evidence.spectral_convergence or {
        "status": "not_requested",
        "passed": None,
    }
    if settings.require_spectral_convergence and not bool(convergence_report.get("passed")):
        failures.append(
            FailureRecord(
                FailureCode.SPECTRAL_CONVERGENCE_FAILURE,
                "The spectrum did not converge within the configured refinement budget.",
                True,
            )
        )

    cross_solver = evidence.independent_solver_check or {"status": "not_requested"}
    if cross_solver["status"] == "failed":
        failures.append(
            FailureRecord(
                FailureCode.SOLVER_DISAGREEMENT,
                "The primary and reference solvers disagree beyond tolerance.",
                True,
                context={"maximum_absolute_difference": cross_solver["maximum_absolute_difference"]},
            )
        )

    reciprocity = evidence.reciprocity_check
    if settings.require_reciprocity and reciprocity and reciprocity.get("status") == "failed":
        failures.append(
            FailureRecord(
                FailureCode.RECIPROCITY_FAILURE,
                "Forward and reversed power transmittance disagree beyond the "
                "reciprocity tolerance.",
                False,
                context={
                    "max_relative_deviation": reciprocity.get("max_relative_deviation"),
                    "worst_wavelength_nm": reciprocity.get("worst_wavelength_nm"),
                },
            )
        )

    accepted = not failures
    limited = bool(
        settings.require_independent_solver and cross_solver["status"] == "unavailable"
    )
    certificate = {
        **base,
        "accepted": accepted,
        "status": (
            "physically_valid_with_limits"
            if accepted and limited
            else ("physically_valid" if accepted else "rejected_physics")
        ),
        "solver": evidence.solver_name,
        "physics_audit": evidence.physics_audit,
        "energy_accounting": _energy_accounting_certificate_block(evidence),
        "spectral_convergence": evidence.spectral_convergence,
        "independent_solver_check": cross_solver,
        "tightest_margin": evidence.tightest_margin,
        "high_precision_referee": evidence.high_precision_referee,
        "material_provenance_sha256": stable_sha256(evidence.material_provenance),
        "material_catalog": evidence.material_catalog,
        "runtime": evidence.runtime,
        "failures": [enrich_failure_actions(item).to_dict() for item in failures],
    }
    if settings.require_reciprocity and evidence.reciprocity_check is not None:
        certificate["reciprocity_check"] = evidence.reciprocity_check
    certificate["evidence_coverage"] = _evidence_coverage_payload(certificate)
    certificate["certificate_id"] = stable_sha256(certificate)
    return certificate


def certify_simulation(
    workbench: TMMWorkbench,
    task: SimulationTask,
    settings: AcceptanceSettings | None = None,
) -> CertifiedSimulation:
    """Execute a task and let deterministic checks decide acceptance.

    Thin orchestration: collect the verification evidence, then recompute the
    certificate from it.  The two halves meet only through the evidence, which
    is what later makes third-party replay (``verify-run``) possible without
    re-running the solver.
    """

    settings = settings or AcceptanceSettings()
    assessment = assess_tmm_capability(task)
    try:
        evidence, result = collect_verification_evidence(
            workbench, task, settings, assessment=assessment
        )
        certificate = evaluate_evidence(evidence, settings)
        return CertifiedSimulation(result, certificate, evidence=evidence)
    except Exception as exc:
        failure = failure_from_exception(exc)
        base = _certificate_base(dataclass_to_dict(task), assessment.to_dict())
        base["failures"] = [failure.to_dict()]
        base["evidence_coverage"] = _evidence_coverage_payload(base)
        base["certificate_id"] = stable_sha256(base)
        return CertifiedSimulation(None, base)


__all__ = [
    "AcceptanceSettings",
    "CertifiedSimulation",
    "EVIDENCE_SCHEMA_VERSION",
    "ENERGY_FALLBACK_REASON_PRE_LEDGER",
    "ENERGY_FALLBACK_REASON_UNAVAILABLE",
    "VERIFICATION_POLICY_SCHEMA_VERSION",
    "VERIFIER_DEPENDENCY_GRAPH",
    "VerificationArtifactError",
    "VerificationEvidence",
    "attach_uncertainty_budget",
    "certify_simulation",
    "collect_verification_evidence",
    "evaluate_evidence",
    "verification_policy_dict",
    "verification_policy_from_dict",
]
