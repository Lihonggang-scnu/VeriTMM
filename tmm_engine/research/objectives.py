"""Serializable objective and constraint contracts for research algorithms.

These contracts define optimization bookkeeping, not spectrum evaluation.
Neither a high optimizer score nor satisfied constraints implies physics
validity; callers must obtain that status from VeriTMM's physics validation and
certificate mechanisms.
"""

from __future__ import annotations

import math
from typing import Any, Literal, TypeAlias

from pydantic import (
    Field,
    JsonValue,
    StrictBool,
    StrictFloat,
    StrictStr,
    field_validator,
    model_validator,
)

from .contracts import ResearchModel, content_id

OBJECTIVE_SET_SCHEMA_VERSION = "veritmm-objective-set-v1"
OBJECTIVE_RESULT_SCHEMA_VERSION = "veritmm-objective-result-v1"

Observable: TypeAlias = Literal["R", "T", "A"]
Polarization: TypeAlias = Literal["s", "p", "unpolarized"]
Aggregation: TypeAlias = Literal["mean", "min", "max"]


def _finite(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


def _nonempty(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


class SpectralQuantitySpec(ResearchModel):
    """Shared observable, band, channel, and aggregation selection."""

    name: StrictStr
    observable: Observable
    wavelength_min_nm: StrictFloat
    wavelength_max_nm: StrictFloat
    angle_deg: StrictFloat = 0.0
    polarization: Polarization = "unpolarized"
    aggregation: Aggregation = "mean"
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        return _nonempty(value, "name")

    @model_validator(mode="after")
    def _valid_spectral_selection(self) -> "SpectralQuantitySpec":
        lower = _finite(self.wavelength_min_nm, "wavelength_min_nm")
        upper = _finite(self.wavelength_max_nm, "wavelength_max_nm")
        angle = _finite(self.angle_deg, "angle_deg")
        if lower <= 0 or upper <= lower:
            raise ValueError("wavelength band requires 0 < minimum < maximum")
        if not 0 <= angle < 90:
            raise ValueError("angle_deg must satisfy 0 <= angle_deg < 90")
        return self


class ObjectiveSpec(SpectralQuantitySpec):
    """One weighted maximize, minimize, or target objective."""

    direction: Literal["maximize", "minimize", "target"]
    weight: StrictFloat = 1.0
    target: StrictFloat | None = None

    @model_validator(mode="after")
    def _valid_objective(self) -> "ObjectiveSpec":
        weight = _finite(self.weight, "weight")
        if weight <= 0:
            raise ValueError("objective weight must be positive")
        if self.direction == "target":
            if self.target is None:
                raise ValueError("target direction requires target")
            target = _finite(self.target, "target")
            if not 0 <= target <= 1:
                raise ValueError("target must be in [0, 1]")
        elif self.target is not None:
            raise ValueError("target is only valid for target objectives")
        return self


class ConstraintSpec(SpectralQuantitySpec):
    """One threshold relation over an aggregated observable."""

    relation: Literal["at_least", "at_most"]
    threshold: StrictFloat
    tolerance: StrictFloat = 0.0

    @model_validator(mode="after")
    def _valid_constraint(self) -> "ConstraintSpec":
        threshold = _finite(self.threshold, "threshold")
        tolerance = _finite(self.tolerance, "tolerance")
        if not 0 <= threshold <= 1:
            raise ValueError("constraint threshold must be in [0, 1]")
        if tolerance < 0:
            raise ValueError("constraint tolerance must be non-negative")
        return self


class ObjectiveSet(ResearchModel):
    """Versioned weighted multi-objective and constraint definition."""

    schema_version: Literal[OBJECTIVE_SET_SCHEMA_VERSION] = OBJECTIVE_SET_SCHEMA_VERSION
    objective_set_id: StrictStr = ""
    objectives: tuple[ObjectiveSpec, ...] = Field(min_length=1)
    constraints: tuple[ConstraintSpec, ...] = ()
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _valid_set(self) -> "ObjectiveSet":
        names = [item.name for item in (*self.objectives, *self.constraints)]
        if len(names) != len(set(names)):
            raise ValueError("objective and constraint names must be unique")
        expected_id = content_id(
            "objective_set",
            self.model_dump(mode="json", exclude={"objective_set_id"}),
        )
        if self.objective_set_id and self.objective_set_id != expected_id:
            raise ValueError("objective_set_id does not match the contract content")
        object.__setattr__(self, "objective_set_id", expected_id)
        return self


class ObjectiveValue(ResearchModel):
    """Aggregated observable value produced externally by an evaluator."""

    objective_name: StrictStr
    value: StrictFloat

    @model_validator(mode="after")
    def _finite_value(self) -> "ObjectiveValue":
        _nonempty(self.objective_name, "objective_name")
        _finite(self.value, "value")
        return self


class ObjectiveScore(ResearchModel):
    """Deterministic optimizer score component, never a validity signal."""

    objective_name: StrictStr
    value: StrictFloat
    score: StrictFloat
    weighted_score: StrictFloat

    @model_validator(mode="after")
    def _finite_score(self) -> "ObjectiveScore":
        _nonempty(self.objective_name, "objective_name")
        _finite(self.value, "value")
        _finite(self.score, "score")
        _finite(self.weighted_score, "weighted_score")
        return self


class ConstraintStatus(ResearchModel):
    """Externally evaluated threshold status with explicit comparison inputs."""

    constraint_name: StrictStr
    relation: Literal["at_least", "at_most"]
    value: StrictFloat
    threshold: StrictFloat
    tolerance: StrictFloat
    satisfied: StrictBool

    @model_validator(mode="after")
    def _finite_status(self) -> "ConstraintStatus":
        _nonempty(self.constraint_name, "constraint_name")
        _finite(self.value, "value")
        _finite(self.threshold, "threshold")
        if _finite(self.tolerance, "tolerance") < 0:
            raise ValueError("constraint status tolerance must be non-negative")
        return self


class ObjectiveSetResult(ResearchModel):
    """Evaluator output whose score explicitly does not certify physics validity."""

    schema_version: Literal[OBJECTIVE_RESULT_SCHEMA_VERSION] = OBJECTIVE_RESULT_SCHEMA_VERSION
    objective_set_id: StrictStr
    candidate_id: StrictStr
    values: tuple[ObjectiveValue, ...]
    scores: tuple[ObjectiveScore, ...]
    total_score: StrictFloat
    constraints: tuple[ConstraintStatus, ...] = ()
    feasible: StrictBool
    physics_validity: Literal["not_assessed"] = "not_assessed"
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _valid_result(self) -> "ObjectiveSetResult":
        _nonempty(self.objective_set_id, "objective_set_id")
        _nonempty(self.candidate_id, "candidate_id")
        _finite(self.total_score, "total_score")
        value_names = [item.objective_name for item in self.values]
        score_names = [item.objective_name for item in self.scores]
        if len(value_names) != len(set(value_names)):
            raise ValueError("objective result values must have unique names")
        if len(score_names) != len(set(score_names)):
            raise ValueError("objective result scores must have unique names")
        if set(value_names) != set(score_names):
            raise ValueError("objective values and scores must cover the same names")
        if self.feasible != all(status.satisfied for status in self.constraints):
            raise ValueError("feasible must equal the conjunction of constraint statuses")
        return self


__all__ = [
    "Aggregation",
    "ConstraintSpec",
    "ConstraintStatus",
    "OBJECTIVE_RESULT_SCHEMA_VERSION",
    "OBJECTIVE_SET_SCHEMA_VERSION",
    "ObjectiveScore",
    "ObjectiveSet",
    "ObjectiveSetResult",
    "ObjectiveSpec",
    "ObjectiveValue",
    "Observable",
    "Polarization",
]
