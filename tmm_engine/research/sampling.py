"""Deterministic indexed sampling plans for research design spaces.

The Sobol implementation is a compact 32-bit digitally shifted engine for at
most 16 dimensions.  It uses published primitive-polynomial direction-number
parameters and never substitutes pseudorandom samples when the requested
dimension is unsupported.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from typing import Annotated, Any, Literal, TypeAlias

import numpy as np
from pydantic import Field, JsonValue, StrictInt, StrictStr, model_validator

from .contracts import DesignCandidate, ResearchModel, content_id
from .design_space import DesignSpace

SAMPLING_PLAN_SCHEMA_VERSION = "veritmm-sampling-plan-v1"
MAX_SAMPLE_COUNT = 1_000_000
MAX_GRID_POINTS = 1_000_000
SOBOL_MAX_DIMENSION = 16
SOBOL_BITS = 32

SamplingStrategy: TypeAlias = Literal[
    "random", "grid", "latin_hypercube", "sobol"
]


class SamplingPlan(ResearchModel):
    """Versioned immutable sampling request with a stable content identity."""

    schema_version: Literal[SAMPLING_PLAN_SCHEMA_VERSION] = (
        SAMPLING_PLAN_SCHEMA_VERSION
    )
    plan_id: StrictStr = ""
    strategy: SamplingStrategy
    sample_count: Annotated[StrictInt, Field(ge=1, le=MAX_SAMPLE_COUNT)]
    seed: StrictInt = 0
    grid_levels: StrictInt | dict[StrictStr, StrictInt] | None = None
    options: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _valid_plan(self) -> "SamplingPlan":
        if self.strategy == "grid":
            if self.grid_levels is None:
                raise ValueError("grid strategy requires finite grid_levels")
            levels = (
                self.grid_levels.values()
                if isinstance(self.grid_levels, dict)
                else (self.grid_levels,)
            )
            if any(
                isinstance(level, bool) or not isinstance(level, int) or level < 1
                for level in levels
            ):
                raise ValueError("grid levels must be positive integers")
            if self.options:
                raise ValueError("grid strategy does not define additional options")
        elif self.grid_levels is not None:
            raise ValueError("grid_levels is only valid for grid strategy")

        allowed_options = {
            "random": set(),
            "grid": set(),
            "latin_hypercube": {"centered"},
            "sobol": {"skip", "fallback_policy"},
        }[self.strategy]
        unknown = set(self.options) - allowed_options
        if unknown:
            raise ValueError(
                f"unsupported {self.strategy} sampling options: {sorted(unknown)}"
            )
        if "centered" in self.options and not isinstance(
            self.options["centered"], bool
        ):
            raise ValueError("latin_hypercube centered option must be boolean")
        if "fallback_policy" in self.options:
            fp = self.options["fallback_policy"]
            if fp not in ("fail", "lhs"):
                raise ValueError(
                    "sobol fallback_policy must be 'fail' or 'lhs'"
                )
        if "skip" in self.options:
            skip = self.options["skip"]
            if isinstance(skip, bool) or not isinstance(skip, int) or skip < 0:
                raise ValueError("sobol skip option must be a non-negative integer")
            if skip + self.sample_count > 2**SOBOL_BITS:
                raise ValueError("sobol skip plus sample_count exceeds the 32-bit engine")

        expected_id = content_id(
            "sampling_plan",
            self.model_dump(mode="json", exclude={"plan_id"}),
        )
        if self.plan_id and self.plan_id != expected_id:
            raise ValueError("plan_id does not match sampling plan content")
        object.__setattr__(self, "plan_id", expected_id)
        return self


def sample_candidates(
    design_space: DesignSpace, plan: SamplingPlan
) -> tuple[DesignCandidate, ...]:
    """Generate one stable ordered candidate tuple for ``plan``."""

    if not isinstance(design_space, DesignSpace):
        raise TypeError("design_space must be a DesignSpace")
    if not isinstance(plan, SamplingPlan):
        raise TypeError("plan must be a SamplingPlan")
    if plan.strategy == "random":
        raw = design_space.sample_indices(range(plan.sample_count), seed=plan.seed)
        candidates = tuple(
            design_space.candidate(
                candidate.values,
                sample_index=index,
                sampler="random",
                seed=plan.seed,
                metadata={"sampling_plan_id": plan.plan_id},
            )
            for index, candidate in enumerate(raw)
        )
    else:
        normalized, effective_strategy = _normalized_samples_impl(design_space, plan)
        candidates = tuple(
            _candidate_from_normalized(
                design_space,
                plan,
                index,
                row,
                effective_strategy=effective_strategy,
            )
            for index, row in enumerate(normalized)
        )
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError(
            "sampling plan produced duplicate candidate IDs; reduce sample_count "
            "or increase the finite design-space cardinality"
        )
    return candidates


def normalized_samples(design_space: DesignSpace, plan: SamplingPlan) -> np.ndarray:
    """Return the bounded normalized matrix for non-random strategies."""
    samples, _ = _normalized_samples_impl(design_space, plan)
    return samples


def _normalized_samples_impl(
    design_space: DesignSpace, plan: SamplingPlan
) -> tuple[np.ndarray, str]:
    """Return ``(matrix, effective_strategy)`` — may differ from plan.strategy."""
    dimension = len(design_space.variables)
    if plan.strategy == "grid":
        return _grid_samples(design_space, plan), "grid"
    if plan.strategy == "latin_hypercube":
        return (
            _latin_hypercube_samples(
                design_space,
                plan,
                centered=bool(plan.options.get("centered", False)),
            ),
            "latin_hypercube",
        )
    if plan.strategy == "sobol":
        if dimension > SOBOL_MAX_DIMENSION:
            fallback_policy = str(plan.options.get("fallback_policy", "fail"))
            if fallback_policy == "lhs":
                return (
                    _latin_hypercube_samples(design_space, plan, centered=False),
                    "latin_hypercube",
                )
            raise ValueError(
                f"Sobol sampler supports at most {SOBOL_MAX_DIMENSION} dimensions "
                f"(got {dimension}); set options={{'fallback_policy': 'lhs'}} to "
                f"use Latin-hypercube sampling instead"
            )
        skip = int(plan.options.get("skip", 0))
        return _sobol_samples(design_space, plan, skip=skip), "sobol"
    raise ValueError("normalized_samples is not used for random strategy")


def _candidate_from_normalized(
    design_space: DesignSpace,
    plan: SamplingPlan,
    sample_index: int,
    row: np.ndarray,
    *,
    effective_strategy: str | None = None,
) -> DesignCandidate:
    decoded = design_space.candidate_from_normalized(tuple(float(value) for value in row))
    metadata: dict[str, Any] = {"sampling_plan_id": plan.plan_id}
    if effective_strategy is not None and effective_strategy != plan.strategy:
        metadata["effective_strategy"] = effective_strategy
        metadata["declared_strategy"] = plan.strategy
        metadata["fallback_reason"] = "sobol_dimension_limit"
    return design_space.candidate(
        decoded.values,
        sample_index=sample_index,
        sampler=effective_strategy if effective_strategy is not None else plan.strategy,
        seed=plan.seed,
        metadata=metadata,
    )


def _grid_samples(design_space: DesignSpace, plan: SamplingPlan) -> np.ndarray:
    variables = design_space.variables
    if isinstance(plan.grid_levels, dict):
        expected = [variable.name for variable in variables]
        missing = [name for name in expected if name not in plan.grid_levels]
        extra = sorted(set(plan.grid_levels) - set(expected))
        if missing or extra:
            raise ValueError(
                f"grid level variable mismatch: missing={missing}, extra={extra}"
            )
        levels = tuple(int(plan.grid_levels[name]) for name in expected)
    elif isinstance(plan.grid_levels, int) and not isinstance(plan.grid_levels, bool):
        levels = (int(plan.grid_levels),) * len(variables)
    else:  # pragma: no cover - SamplingPlan invariant
        raise ValueError("grid strategy requires grid_levels")

    from .contracts import DiscreteThicknessVariable, MaterialChoiceVariable

    for variable, level in zip(variables, levels):
        cardinality: int | None = None
        if isinstance(variable, DiscreteThicknessVariable):
            cardinality = len(variable.values_nm)
        elif isinstance(variable, MaterialChoiceVariable):
            cardinality = len(variable.options)
        if cardinality is not None and level > cardinality:
            raise ValueError(
                f"grid level for finite variable {variable.name!r} exceeds "
                f"its cardinality {cardinality}"
            )
    total = math.prod(levels)
    if total > MAX_GRID_POINTS:
        raise ValueError(
            f"declared Cartesian grid has {total} points; maximum is {MAX_GRID_POINTS}"
        )
    if total < plan.sample_count:
        raise ValueError(
            "declared Cartesian grid has fewer points than requested sample_count"
        )
    axes = [
        np.asarray([0.5], dtype=np.float64)
        if level == 1
        else np.linspace(0.0, 1.0, level, dtype=np.float64)
        for level in levels
    ]
    rows = itertools.islice(itertools.product(*axes), plan.sample_count)
    return np.asarray(tuple(rows), dtype=np.float64)


def _latin_hypercube_samples(
    design_space: DesignSpace,
    plan: SamplingPlan,
    *,
    centered: bool,
) -> np.ndarray:
    count = plan.sample_count
    dimension = len(design_space.variables)
    matrix = np.empty((count, dimension), dtype=np.float64)
    for column in range(dimension):
        rng = np.random.Generator(
            np.random.PCG64(
                _derived_seed(
                    "latin_hypercube",
                    design_space.design_space_id,
                    plan.seed,
                    column,
                )
            )
        )
        permutation = rng.permutation(count)
        offsets = np.full(count, 0.5) if centered else rng.random(count)
        matrix[:, column] = (permutation + offsets) / count
    return matrix


# Primitive-polynomial parameters (degree, coefficient bits, initial m values)
# for Sobol dimensions 2 through 16. Dimension 1 uses powers of two directly.
_SOBOL_PARAMETERS: tuple[tuple[int, int, tuple[int, ...]], ...] = (
    (1, 0, (1,)),
    (2, 1, (1, 3)),
    (3, 1, (1, 3, 1)),
    (3, 2, (1, 1, 1)),
    (4, 1, (1, 3, 5, 13)),
    (4, 4, (1, 1, 5, 5)),
    (5, 2, (1, 3, 3, 9, 7)),
    (5, 4, (1, 1, 5, 11, 27)),
    (5, 7, (1, 1, 7, 13, 3)),
    (5, 11, (1, 1, 5, 1, 15)),
    (5, 13, (1, 1, 1, 3, 29)),
    (5, 14, (1, 3, 5, 5, 21)),
    (6, 1, (1, 3, 3, 9, 7, 49)),
    (6, 13, (1, 1, 1, 15, 21, 21)),
    (6, 16, (1, 3, 1, 13, 27, 49)),
)


def _sobol_samples(
    design_space: DesignSpace,
    plan: SamplingPlan,
    *,
    skip: int,
) -> np.ndarray:
    dimension = len(design_space.variables)
    directions = _sobol_direction_numbers(dimension)
    shifts = np.asarray(
        [
            _derived_seed(
                "sobol_digital_shift",
                design_space.design_space_id,
                plan.seed,
                column,
            )
            & 0xFFFFFFFF
            for column in range(dimension)
        ],
        dtype=np.uint32,
    )
    matrix = np.empty((plan.sample_count, dimension), dtype=np.float64)
    denominator = float(2**SOBOL_BITS)
    for row, index in enumerate(range(skip, skip + plan.sample_count)):
        gray = index ^ (index >> 1)
        for column in range(dimension):
            value = 0
            bits = gray
            bit = 0
            while bits:
                if bits & 1:
                    value ^= int(directions[column, bit])
                bits >>= 1
                bit += 1
            value ^= int(shifts[column])
            matrix[row, column] = value / denominator
    return matrix


def _sobol_direction_numbers(dimension: int) -> np.ndarray:
    if not 1 <= dimension <= SOBOL_MAX_DIMENSION:
        raise ValueError(
            f"Sobol dimension must be between 1 and {SOBOL_MAX_DIMENSION}"
        )
    directions = np.zeros((dimension, SOBOL_BITS), dtype=np.uint32)
    for bit in range(SOBOL_BITS):
        directions[0, bit] = np.uint32(1 << (SOBOL_BITS - bit - 1))
    for column in range(1, dimension):
        degree, coefficients, initial = _SOBOL_PARAMETERS[column - 1]
        for bit in range(degree):
            directions[column, bit] = np.uint32(
                initial[bit] << (SOBOL_BITS - bit - 1)
            )
        for bit in range(degree, SOBOL_BITS):
            value = int(directions[column, bit - degree])
            value ^= value >> degree
            for coefficient_index in range(1, degree):
                if (coefficients >> (degree - 1 - coefficient_index)) & 1:
                    value ^= int(directions[column, bit - coefficient_index])
            directions[column, bit] = np.uint32(value)
    return directions


def _derived_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


__all__ = [
    "MAX_GRID_POINTS",
    "MAX_SAMPLE_COUNT",
    "SAMPLING_PLAN_SCHEMA_VERSION",
    "SOBOL_MAX_DIMENSION",
    "SamplingPlan",
    "SamplingStrategy",
    "normalized_samples",
    "sample_candidates",
]
