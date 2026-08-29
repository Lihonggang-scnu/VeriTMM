"""Active challenge verification: deterministic search for weak evidence.

The search operates on the public ``SimulationTask`` contract and sends every
candidate through the same deterministic physics certificate path as normal
execution.  It is a diagnostic proposal mechanism, never a certificate
authority.
"""

from __future__ import annotations

import copy
import hashlib
import math
import random
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..acceptance import AcceptanceSettings, certify_simulation
from ..capabilities import failure_from_exception
from ..material_registry import MaterialRegistry
from ..run_artifacts import stable_payload_sha256
from ..task_io import simulation_task_from_dict
from ..workbench import TMMWorkbench


class ChallengeObjective(str, Enum):
    """Quantity that the bounded search tries to make worst."""

    MIN_MARGIN = "min_margin"
    MAX_SOLVER_DISAGREEMENT = "max_solver_disagreement"
    MAX_CONVERGENCE_RESIDUAL = "max_convergence_residual"
    METAMORPHIC_VIOLATION = "metamorphic_violation"


class ChallengeSpec(BaseModel):
    """Specification for a deterministic, bounded challenge search."""

    model_config = ConfigDict(extra="forbid")

    seed: int
    budget: int = Field(default=100, ge=1, le=10_000)
    num_layers: tuple[int, int] = (2, 8)
    thickness_range_nm: tuple[float, float] = (10.0, 500.0)
    wavelength_range_nm: tuple[float, float] = (400.0, 1000.0)
    angle_range_deg: tuple[float, float] = (0.0, 80.0)
    material_pool: list[str] = Field(
        default_factory=lambda: ["SiO2", "Si3N4", "TiO2"],
        min_length=1,
    )
    objective: ChallengeObjective = ChallengeObjective.MIN_MARGIN
    minimize_on_find: bool = True

    @model_validator(mode="after")
    def _validate_bounds(self) -> "ChallengeSpec":
        if self.num_layers[0] < 1 or self.num_layers[1] < self.num_layers[0]:
            raise ValueError("num_layers must be an ordered positive range")
        for name, values in (
            ("thickness_range_nm", self.thickness_range_nm),
            ("wavelength_range_nm", self.wavelength_range_nm),
            ("angle_range_deg", self.angle_range_deg),
        ):
            low, high = (float(values[0]), float(values[1]))
            if not math.isfinite(low) or not math.isfinite(high) or high <= low:
                raise ValueError(f"{name} must be a finite increasing range")
        if self.thickness_range_nm[0] <= 0 or self.wavelength_range_nm[0] <= 0:
            raise ValueError("thickness and wavelength ranges must be positive")
        if self.angle_range_deg[0] < 0 or self.angle_range_deg[1] >= 90:
            raise ValueError("angle_range_deg must stay within [0, 90)")
        if any(not str(material).strip() for material in self.material_pool):
            raise ValueError("material_pool entries must be non-empty")
        return self


class ChallengeResult(BaseModel):
    """Bounded result and evidence from one challenge search."""

    model_config = ConfigDict(extra="forbid")

    spec: ChallengeSpec
    candidates_evaluated: int
    worst_candidate: dict[str, Any] | None = None
    worst_margin: float | None = None
    worst_disagreement: float | None = None
    minimized_candidate: dict[str, Any] | None = None
    certificate: dict[str, Any] | None = None
    accepted: bool = False
    canonical_task_path: str | None = None
    failure_count: int = 0
    trajectory_sha256: str | None = None


_MATERIAL_INDEX = {
    "sio2": 1.46,
    "si3n4": 2.00,
    "tio2": 2.35,
}


def _constant_index(material: str) -> float:
    key = "".join(character for character in str(material).casefold() if character.isalnum())
    if key in _MATERIAL_INDEX:
        return _MATERIAL_INDEX[key]
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return 1.30 + (int.from_bytes(digest[:2], "big") / 65535.0) * 1.10


def generate_challenge_candidate(spec: ChallengeSpec, iteration: int) -> dict[str, Any]:
    """Generate one public-contract candidate deterministically."""

    iteration = int(iteration)
    if iteration < 0:
        raise ValueError("iteration must be non-negative")
    seed = int(spec.seed) + iteration
    rng = random.Random(seed)
    num_layers = rng.randint(*spec.num_layers)
    materials = [rng.choice(spec.material_pool) for _ in range(num_layers)]
    thicknesses = [
        rng.uniform(*spec.thickness_range_nm) for _ in range(num_layers)
    ]
    # The candidate parameters are random, but the verification grid is a
    # deterministic dense grid so a sparse-grid warning does not dominate the
    # challenge objective or manufacture convergence failures.
    wavelengths = np.linspace(
        spec.wavelength_range_nm[0],
        spec.wavelength_range_nm[1],
        81,
        dtype=np.float64,
    ).tolist()
    angle = rng.uniform(*spec.angle_range_deg)
    layers = [
        {
            "constant_n": _constant_index(material),
            "thickness_nm": thickness,
            "optimizable": False,
            "label": str(material),
        }
        for material, thickness in zip(materials, thicknesses, strict=True)
    ]
    return {
        "mode": "simulate",
        "simulation": {
            "stack": {
                "name": f"challenge_seed_{spec.seed}_iteration_{iteration}",
                "incident": {"constant_n": 1.0},
                "exit": {"constant_n": 1.5},
                "layers": layers,
            },
            "spectrum": {"values_nm": wavelengths},
            "illumination": {
                "angles_deg": [angle],
                "polarizations": [rng.choice(["s", "p"])],
            },
            "solver": "smatrix",
            "requested_outputs": ["R", "T", "A"],
        },
    }


def _convergence_residual(certificate: dict[str, Any]) -> float:
    convergence = certificate.get("spectral_convergence")
    if not isinstance(convergence, dict):
        return 0.0
    rounds = convergence.get("rounds", [])
    if not isinstance(rounds, list):
        return 0.0
    values: list[float] = []
    for item in rounds:
        if not isinstance(item, dict):
            continue
        for key in ("max_pointwise_deviation", "max_integral_deviation"):
            value = item.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values.append(float(value))
    return max(values, default=0.0)


def _challenge_score(
    certificate: dict[str, Any], objective: ChallengeObjective
) -> float:
    if objective == ChallengeObjective.MIN_MARGIN:
        margin = (certificate.get("tightest_margin") or {}).get("normalized_margin")
        if isinstance(margin, (int, float)) and math.isfinite(float(margin)):
            return float(margin)
        return 1.0 if certificate.get("accepted") else 0.0
    if objective == ChallengeObjective.MAX_SOLVER_DISAGREEMENT:
        value = (certificate.get("independent_solver_check") or {}).get(
            "maximum_absolute_difference"
        )
        return float(value) if isinstance(value, (int, float)) else 0.0
    if objective == ChallengeObjective.MAX_CONVERGENCE_RESIDUAL:
        return _convergence_residual(certificate)
    violation = certificate.get("metamorphic_violation")
    if isinstance(violation, bool):
        return 1.0 if violation else 0.0
    if isinstance(violation, (int, float)):
        return float(violation)
    return 0.0


def _is_better(score: float, best: float | None, objective: ChallengeObjective) -> bool:
    if best is None:
        return True
    if objective == ChallengeObjective.MIN_MARGIN:
        return score < best
    return score > best


def _evaluate_candidate(
    candidate: dict[str, Any],
    workbench: TMMWorkbench,
) -> dict[str, Any]:
    try:
        task = simulation_task_from_dict(candidate["simulation"])
        certified = certify_simulation(
            workbench,
            task,
            AcceptanceSettings(),
        )
        return certified.certificate
    except Exception as exc:
        failure = failure_from_exception(exc).to_dict()
        return {
            "accepted": False,
            "status": "challenge_execution_failed",
            "failures": [failure],
        }


def minimize_challenge_case(
    candidate: dict[str, Any],
    cert: dict[str, Any],
    *,
    spec: ChallengeSpec | None = None,
    workbench: TMMWorkbench | None = None,
) -> dict[str, Any]:
    """Apply deterministic layer shrinking while preserving the objective.

    If no evaluator is supplied, returning a deep copy is intentional and keeps
    this helper useful for callers that only want a stable canonical payload.
    """

    minimized = copy.deepcopy(candidate)
    if spec is None or workbench is None:
        return minimized
    baseline = _challenge_score(cert, spec.objective)
    layers = list(minimized["simulation"]["stack"]["layers"])
    while len(layers) > spec.num_layers[0]:
        trial = copy.deepcopy(minimized)
        trial_layers = list(trial["simulation"]["stack"]["layers"])
        trial_layers.pop()
        trial["simulation"]["stack"]["layers"] = trial_layers
        trial_cert = _evaluate_candidate(trial, workbench)
        trial_score = _challenge_score(trial_cert, spec.objective)
        preserves = (
            trial_score <= baseline
            if spec.objective == ChallengeObjective.MIN_MARGIN
            else trial_score >= baseline
        )
        if not preserves:
            break
        minimized = trial
        layers = trial_layers
        baseline = trial_score
    return minimized


def run_challenge_search(
    spec: ChallengeSpec,
    *,
    workbench: TMMWorkbench | None = None,
    registry: MaterialRegistry | None = None,
) -> ChallengeResult:
    """Search exactly ``spec.budget`` candidates through full verification."""

    registry = registry or MaterialRegistry()
    workbench = workbench or TMMWorkbench(registry)
    best_score: float | None = None
    worst_candidate: dict[str, Any] | None = None
    worst_cert: dict[str, Any] | None = None
    trajectory: list[dict[str, Any]] = []
    failure_count = 0
    accepted_candidate_found = False
    for iteration in range(int(spec.budget)):
        candidate = generate_challenge_candidate(spec, iteration)
        certificate = _evaluate_candidate(candidate, workbench)
        if not certificate.get("accepted"):
            failure_count += 1
        if spec.objective == ChallengeObjective.MIN_MARGIN:
            if certificate.get("accepted") and not accepted_candidate_found:
                # Prefer the worst case that still carries an accepted
                # certificate; rejected candidates remain visible in the
                # trajectory and failure_count for active-falsification use.
                accepted_candidate_found = True
                best_score = None
                worst_candidate = None
                worst_cert = None
            if accepted_candidate_found and not certificate.get("accepted"):
                continue
        score = _challenge_score(certificate, spec.objective)
        trajectory.append(
            {
                "iteration": iteration,
                "candidate_sha256": stable_payload_sha256(candidate),
                "score": score,
                "accepted": bool(certificate.get("accepted")),
                "certificate_id": certificate.get("certificate_id"),
            }
        )
        if _is_better(score, best_score, spec.objective):
            best_score = score
            worst_candidate = copy.deepcopy(candidate)
            worst_cert = copy.deepcopy(certificate)

    minimized = None
    if spec.minimize_on_find and worst_candidate is not None and worst_cert is not None:
        minimized = minimize_challenge_case(
            worst_candidate,
            worst_cert,
            spec=spec,
            workbench=workbench,
        )
    worst_margin = None
    worst_disagreement = None
    if worst_cert is not None:
        margin = (worst_cert.get("tightest_margin") or {}).get("normalized_margin")
        disagreement = (worst_cert.get("independent_solver_check") or {}).get(
            "maximum_absolute_difference"
        )
        worst_margin = float(margin) if isinstance(margin, (int, float)) else None
        worst_disagreement = (
            float(disagreement) if isinstance(disagreement, (int, float)) else None
        )
    return ChallengeResult(
        spec=spec,
        candidates_evaluated=int(spec.budget),
        worst_candidate=worst_candidate,
        worst_margin=worst_margin,
        worst_disagreement=worst_disagreement,
        minimized_candidate=minimized,
        certificate=worst_cert,
        accepted=bool(worst_cert and worst_cert.get("accepted")),
        failure_count=failure_count,
        trajectory_sha256=stable_payload_sha256(trajectory),
    )


__all__ = [
    "ChallengeObjective",
    "ChallengeResult",
    "ChallengeSpec",
    "generate_challenge_candidate",
    "minimize_challenge_case",
    "run_challenge_search",
]
