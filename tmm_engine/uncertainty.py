"""Shared uncertainty semantics for tolerance and robust optimization.

Sampling ensembles may use different seeds, but every consumer applies the
same declared distribution and thickness boundary policy.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np

from .capabilities import (
    FailureCode,
    FailureRecord,
    PhysicsEngineError,
    failure_from_exception,
)
from .material_registry import MaterialRegistryError

BoundaryPolicy = Literal["truncate"]
SampleFailureCategory = Literal[
    "invalid_perturbed_design",
    "numerical_failure",
    "material_failure",
    "unexpected_runtime_failure",
]

DEFAULT_MIN_THICKNESS_PHYSICAL_NM = 0.1
DEFAULT_ENERGY_AUDIT_TOLERANCE = 1e-7
FINAL_ROBUSTNESS_SEED_OFFSET = 1_000_003
SAMPLE_FAILURE_CATEGORIES: tuple[SampleFailureCategory, ...] = (
    "invalid_perturbed_design",
    "numerical_failure",
    "material_failure",
    "unexpected_runtime_failure",
)


def wilson_interval(
    successes: int,
    trials: int,
    z: float = 1.959963984540054,
) -> list[float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""

    if trials < 1 or successes < 0 or successes > trials:
        raise ValueError("Wilson interval requires 0 <= successes <= trials and trials > 0")
    n = float(trials)
    p = float(successes) / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def yield_accounting(
    target_pass_count: int,
    completed_sample_count: int,
    requested_sample_count: int,
) -> dict[str, Any]:
    """Separate conditional scientific yield from end-to-end success."""

    passed = int(target_pass_count)
    completed = int(completed_sample_count)
    requested = int(requested_sample_count)
    if requested < 1 or completed < 0 or completed > requested or passed < 0 or passed > completed:
        raise ValueError("invalid requested/completed/pass sample counts")
    conditional = None if completed == 0 else float(passed / completed)
    return {
        "requested_sample_count": requested,
        "completed_sample_count": completed,
        "failed_sample_count": requested - completed,
        "target_pass_count": passed,
        "conditional_yield": conditional,
        "conditional_yield_ci95": None
        if completed == 0
        else wilson_interval(passed, completed),
        "conditional_yield_ci_method": "wilson_score_interval",
        "conditional_yield_ci_denominator": completed,
        "overall_success_fraction": float(passed / requested),
    }


def final_robustness_seed(training_seed: int) -> int:
    """Return the deterministic, disjoint final-proof seed."""

    return int(training_seed) + FINAL_ROBUSTNESS_SEED_OFFSET


def sample_normal_offsets(
    *,
    seed: int,
    sample_count: int,
    layer_count: int,
    sigma_nm: float,
) -> np.ndarray:
    """Draw a reproducible unbounded normal perturbation ensemble."""

    if sample_count < 1 or layer_count < 1 or float(sigma_nm) <= 0:
        raise ValueError("normal perturbation dimensions and sigma must be positive")
    return np.random.default_rng(int(seed)).normal(
        0.0,
        float(sigma_nm),
        size=(int(sample_count), int(layer_count)),
    )


def apply_thickness_boundary_policy(
    thicknesses_nm: Any,
    *,
    boundary_policy: BoundaryPolicy,
    min_thickness_physical_nm: float,
) -> np.ndarray:
    """Apply the public boundary policy to NumPy-compatible thicknesses."""

    minimum = float(min_thickness_physical_nm)
    if not np.isfinite(minimum) or minimum <= 0:
        raise ValueError("min_thickness_physical_nm must be finite and positive")
    values = np.asarray(thicknesses_nm, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("sampled thicknesses must be finite before boundary handling")
    if boundary_policy == "truncate":
        return np.maximum(values, minimum)
    raise ValueError(f"unsupported thickness boundary policy: {boundary_policy}")


def apply_thickness_boundary_policy_torch(
    thicknesses_nm: Any,
    *,
    boundary_policy: BoundaryPolicy,
    min_thickness_physical_nm: float,
) -> Any:
    """Torch-preserving counterpart with identical boundary semantics."""

    minimum = float(min_thickness_physical_nm)
    if not math.isfinite(minimum) or minimum <= 0:
        raise ValueError("min_thickness_physical_nm must be finite and positive")
    if boundary_policy == "truncate":
        return thicknesses_nm.clamp_min(minimum)
    raise ValueError(f"unsupported thickness boundary policy: {boundary_policy}")


def classify_sample_failure(exc: Exception) -> SampleFailureCategory:
    """Classify a failed perturbation without treating it as target failure."""

    failure = failure_from_exception(exc)
    if isinstance(exc, MaterialRegistryError) or failure.code in {
        FailureCode.MATERIAL_NOT_FOUND,
        FailureCode.MATERIAL_AMBIGUITY,
        FailureCode.MATERIAL_RANGE_ERROR,
    }:
        return "material_failure"
    if isinstance(exc, (FloatingPointError, np.linalg.LinAlgError)) or failure.code in {
        FailureCode.NUMERICAL_NONFINITE,
        FailureCode.PASSIVITY_VIOLATION,
        FailureCode.ENERGY_CONSERVATION_FAILURE,
        FailureCode.SPECTRAL_CONVERGENCE_FAILURE,
        FailureCode.SOLVER_DISAGREEMENT,
    }:
        return "numerical_failure"
    if isinstance(exc, ValueError):
        return "invalid_perturbed_design"
    if isinstance(exc, PhysicsEngineError) and failure.code == FailureCode.INVALID_TASK:
        return "invalid_perturbed_design"
    return "unexpected_runtime_failure"


def validate_uncertainty_forward(forward: Any) -> None:
    """Reject invalid solver output before statistical accounting."""

    audit = getattr(forward, "audit", None)
    if not isinstance(audit, dict):
        return
    nonfinite = int(audit.get("nonfinite_value_count", 0) or 0)
    if nonfinite > 0:
        raise PhysicsEngineError(
            FailureRecord(
                FailureCode.NUMERICAL_NONFINITE,
                "perturbed simulation produced non-finite observables",
                True,
                context={"nonfinite_value_count": nonfinite},
            )
        )
    if audit.get("passivity_check_passed") is False:
        raise PhysicsEngineError(
            FailureRecord(
                FailureCode.PASSIVITY_VIOLATION,
                "perturbed simulation failed passive-observable bounds",
                True,
                context={
                    "minimum_observable": audit.get("minimum_observable"),
                    "maximum_observable": audit.get("maximum_observable"),
                },
            )
        )
    energy_error = audit.get("energy_conservation_max_abs_error")
    if energy_error is not None and (
        not math.isfinite(float(energy_error))
        or float(energy_error) > DEFAULT_ENERGY_AUDIT_TOLERANCE
    ):
        raise PhysicsEngineError(
            FailureRecord(
                FailureCode.ENERGY_CONSERVATION_FAILURE,
                "perturbed simulation failed energy-conservation audit",
                True,
                context={
                    "energy_conservation_max_abs_error": energy_error,
                    "tolerance": DEFAULT_ENERGY_AUDIT_TOLERANCE,
                },
            )
        )


def empty_failure_taxonomy() -> dict[str, int]:
    return {category: 0 for category in SAMPLE_FAILURE_CATEGORIES}


__all__ = [
    "BoundaryPolicy",
    "DEFAULT_MIN_THICKNESS_PHYSICAL_NM",
    "DEFAULT_ENERGY_AUDIT_TOLERANCE",
    "FINAL_ROBUSTNESS_SEED_OFFSET",
    "SAMPLE_FAILURE_CATEGORIES",
    "apply_thickness_boundary_policy",
    "apply_thickness_boundary_policy_torch",
    "classify_sample_failure",
    "empty_failure_taxonomy",
    "final_robustness_seed",
    "sample_normal_offsets",
    "validate_uncertainty_forward",
    "wilson_interval",
    "yield_accounting",
]
