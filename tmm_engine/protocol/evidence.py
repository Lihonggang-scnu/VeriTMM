"""Evidence coverage tracking for scientific reproducibility.

``EvidenceCoverage`` is an additive, typed ledger of which verification and
validation steps were performed.  It does not change the semantics of a
physics certificate's ``accepted`` field.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EvidenceStatus(str, Enum):
    """Status of one evidence dimension."""

    VERIFIED = "verified"
    NOT_EVALUATED = "not_evaluated"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


class EvidenceCoverage(BaseModel):
    """Structured coverage of evidence associated with a simulation result.

    Each dimension is independent: ``verified`` means that the corresponding
    check was performed and passed, not that the result has a scalar confidence
    score or that every other dimension was verified.
    """

    model_config = ConfigDict(extra="forbid")

    capability_domain: EvidenceStatus = Field(
        default=EvidenceStatus.NOT_EVALUATED,
        description="Task is inside the declared planar isotropic capability domain.",
    )
    numerical_convergence: EvidenceStatus = Field(
        default=EvidenceStatus.NOT_EVALUATED,
        description="Spectral or grid-refinement convergence was evaluated.",
    )
    passivity: EvidenceStatus = Field(
        default=EvidenceStatus.NOT_EVALUATED,
        description="Energy conservation and passive-observable bounds were evaluated.",
    )
    independent_solver: EvidenceStatus = Field(
        default=EvidenceStatus.NOT_EVALUATED,
        description="An independent solver comparison was evaluated.",
    )
    high_precision_referee: EvidenceStatus = Field(
        default=EvidenceStatus.NOT_EVALUATED,
        description="The optional high-precision referee was evaluated.",
    )
    material_provenance: EvidenceStatus = Field(
        default=EvidenceStatus.NOT_EVALUATED,
        description="Material source identity and interpolation provenance were recorded.",
    )
    uncertainty_quantified: EvidenceStatus = Field(
        default=EvidenceStatus.NOT_EVALUATED,
        description="Parameter uncertainty or tolerance was quantified.",
    )
    experimental_fit: EvidenceStatus = Field(
        default=EvidenceStatus.NOT_EVALUATED,
        description="The result was fitted to experimental measurement data.",
    )
    reproducibility: EvidenceStatus = Field(
        default=EvidenceStatus.NOT_EVALUATED,
        description="Task identity, package version, and replay metadata were recorded.",
    )


def _status_from_mapping(
    value: Any,
    *,
    positive_keys: tuple[str, ...] = ("passed", "agreement"),
) -> EvidenceStatus:
    """Map common legacy check shapes to one typed evidence status."""

    if isinstance(value, bool):
        return EvidenceStatus.VERIFIED if value else EvidenceStatus.FAILED
    if not isinstance(value, Mapping):
        return EvidenceStatus.NOT_EVALUATED
    if value.get("available") is False:
        return EvidenceStatus.UNAVAILABLE
    for key in positive_keys:
        candidate = value.get(key)
        if isinstance(candidate, bool):
            return EvidenceStatus.VERIFIED if candidate else EvidenceStatus.FAILED
    if value.get("triggered") is False:
        return EvidenceStatus.NOT_EVALUATED
    status = str(value.get("status", "")).lower()
    if status in {"passed", "verified", "supported", "ready", "success", "completed"}:
        return EvidenceStatus.VERIFIED
    if status in {"unavailable", "not_available"}:
        return EvidenceStatus.UNAVAILABLE
    if status in {"failed", "failure", "error", "rejected"}:
        return EvidenceStatus.FAILED
    if status in {"not_triggered", "not_requested", "not_evaluated", "skipped"}:
        return EvidenceStatus.NOT_EVALUATED
    return EvidenceStatus.NOT_EVALUATED


def from_certificate(cert: Mapping[str, Any]) -> EvidenceCoverage:
    """Build coverage from an existing or legacy certificate mapping.

    Missing fields intentionally remain ``not_evaluated``.  The function only
    derives additive evidence metadata and never changes ``accepted`` or any
    other stored certificate field.
    """

    coverage = EvidenceCoverage()

    capability = cert.get("capability_assessment")
    if isinstance(capability, Mapping):
        if capability.get("supported") is True or capability.get("status") == "supported":
            coverage.capability_domain = EvidenceStatus.VERIFIED
        elif capability.get("supported") is False or capability.get("status") in {
            "unsupported",
            "rejected",
        }:
            coverage.capability_domain = EvidenceStatus.FAILED
    elif cert.get("accepted") is True:
        # Backward-compatible certificates did not persist capability details.
        coverage.capability_domain = EvidenceStatus.VERIFIED

    audit = cert.get("physics_audit")
    if isinstance(audit, Mapping) and "passivity_check_passed" in audit:
        passed = audit.get("passivity_check_passed")
        if isinstance(passed, bool):
            coverage.passivity = (
                EvidenceStatus.VERIFIED if passed else EvidenceStatus.FAILED
            )
    elif "passivity_check" in cert:
        coverage.passivity = _status_from_mapping(cert["passivity_check"])
    elif "energy_conservation_check" in cert:
        coverage.passivity = _status_from_mapping(cert["energy_conservation_check"])

    for key in ("spectral_convergence", "convergence_check", "spectral_refinement"):
        if key not in cert:
            continue
        convergence = cert[key]
        # Adaptive refinement is only verified after the ledger explicitly
        # proves convergence.  A legacy ``status=passed`` remains supported,
        # but budget exhaustion and depth limits must never be upgraded by the
        # generic status mapper.
        if isinstance(convergence, Mapping) and "refinement_status" in convergence:
            refinement_status = str(convergence.get("refinement_status", "")).lower()
            if refinement_status == "converged":
                coverage.numerical_convergence = EvidenceStatus.VERIFIED
            elif refinement_status in {"budget_exhausted", "max_depth_reached"}:
                coverage.numerical_convergence = EvidenceStatus.FAILED
            else:
                coverage.numerical_convergence = EvidenceStatus.NOT_EVALUATED
        else:
            coverage.numerical_convergence = _status_from_mapping(convergence)
        break

    for key in ("independent_solver_check", "cross_solver_check"):
        if key in cert:
            coverage.independent_solver = _status_from_mapping(cert[key])
            break

    if "high_precision_referee" in cert:
        coverage.high_precision_referee = _status_from_mapping(cert["high_precision_referee"])

    if cert.get("material_provenance_sha256") is not None or cert.get("material_provenance"):
        coverage.material_provenance = EvidenceStatus.VERIFIED

    for key in ("uncertainty", "uncertainty_report", "tolerance_result"):
        if key in cert:
            coverage.uncertainty_quantified = _status_from_mapping(cert[key])
            break
    budget = cert.get("uncertainty_budget")
    if isinstance(budget, Mapping):
        quantitative_keys = (
            "numerical_components",
            "material_components",
            "parameter_components",
            "sampling_components",
        )
        if any(isinstance(budget.get(key), list) and budget.get(key) for key in quantitative_keys):
            coverage.uncertainty_quantified = EvidenceStatus.VERIFIED

    for key in ("experimental_fit", "fit_report"):
        if key in cert:
            coverage.experimental_fit = _status_from_mapping(cert[key])
            break

    if cert.get("task_sha256") and cert.get("veritmm_version"):
        coverage.reproducibility = EvidenceStatus.VERIFIED

    return coverage


__all__ = ["EvidenceCoverage", "EvidenceStatus", "from_certificate"]
