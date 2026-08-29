"""Scientific contract for diverse inverse-design result sets."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

_MetricName = Literal[
    "pareto_nondominated",
    "spectral_distance",
    "structural_distance",
    "material_distance",
]


class DiversityMetric(BaseModel):
    """One measured diversity dimension."""

    model_config = ConfigDict(extra="forbid")

    metric: _MetricName
    value: float
    threshold: float | None = None


class CandidateSetProvenance(BaseModel):
    """How and why candidates were selected."""

    model_config = ConfigDict(extra="forbid")

    source_method: str = Field(
        description="For example, pareto_archive, niching, or cluster_centroids."
    )
    selection_criteria: list[str]
    deduplicated: bool
    deduplication_tolerance: float | None = None


class CandidateSet(BaseModel):
    """Diverse candidates with documented selection and distance evidence."""

    model_config = ConfigDict(extra="forbid")

    candidates: list[dict[str, Any]] = Field(default_factory=list)
    diversity_metrics: list[DiversityMetric] = Field(default_factory=list)
    provenance: CandidateSetProvenance = Field(
        default_factory=lambda: CandidateSetProvenance(
            source_method="explicit",
            selection_criteria=["caller_supplied"],
            deduplicated=False,
        )
    )
    pareto_front_indices: list[int] | None = None

    @classmethod
    def deduplicate(
        cls,
        candidates: Sequence[Mapping[str, Any]],
        tolerance: float = 1e-3,
        distance_fn: Callable[[Mapping[str, Any], Mapping[str, Any]], float] | None = None,
    ) -> "CandidateSet":
        """Greedily retain the first representative of each tolerance-near group."""

        if not math.isfinite(float(tolerance)) or tolerance < 0:
            raise ValueError("deduplication tolerance must be finite and non-negative")
        normalized = _normalize_candidates(candidates)
        distance = distance_fn or structural_distance
        kept: list[dict[str, Any]] = []
        merged = 0
        for candidate in normalized:
            if any(float(distance(candidate, representative)) <= tolerance for representative in kept):
                merged += 1
            else:
                kept.append(candidate)
        metrics = [
            DiversityMetric(
                metric="structural_distance",
                value=_minimum_positive_distance(kept, structural_distance),
                threshold=float(tolerance),
            )
        ]
        if merged:
            metrics.append(
                DiversityMetric(
                    metric="material_distance",
                    value=float(merged / max(1, len(normalized))),
                    threshold=None,
                )
            )
        return cls(
            candidates=kept,
            diversity_metrics=metrics,
            provenance=CandidateSetProvenance(
                source_method="deduplicate",
                selection_criteria=[
                    "retain first representative",
                    "pairwise structural distance above tolerance",
                ],
                deduplicated=True,
                deduplication_tolerance=float(tolerance),
            ),
        )

    @classmethod
    def pareto_filter(
        cls,
        candidates: Sequence[Mapping[str, Any]],
        objectives: Mapping[str, str] | Sequence[str | Mapping[str, str]],
    ) -> "CandidateSet":
        """Return the non-dominated archive under explicit objective directions."""

        normalized = _normalize_candidates(candidates)
        names, directions = _objective_spec(objectives)
        values = [_objective_values(candidate, names) for candidate in normalized]
        front: list[int] = []
        for index, value in enumerate(values):
            dominated = any(
                other_index != index
                and _dominates(values[other_index], value, directions)
                for other_index in range(len(values))
            )
            if not dominated:
                front.append(index)
        retained = [normalized[index] for index in front]
        return cls(
            candidates=retained,
            diversity_metrics=[
                DiversityMetric(
                    metric="pareto_nondominated",
                    value=float(len(front)),
                    threshold=1.0,
                )
            ],
            provenance=CandidateSetProvenance(
                source_method="pareto_archive",
                selection_criteria=[
                    f"{name}:{directions[name]}" for name in names
                ],
                deduplicated=False,
            ),
            pareto_front_indices=front,
        )

    @staticmethod
    def spectral_distance_matrix(candidates: Sequence[Mapping[str, Any]]) -> np.ndarray:
        """Return a symmetric Euclidean distance matrix over stored spectra."""

        return spectral_distance_matrix(candidates)

    @staticmethod
    def structural_distance_matrix(candidates: Sequence[Mapping[str, Any]]) -> np.ndarray:
        """Return a symmetric distance matrix over normalized design structure."""

        return _distance_matrix(candidates, structural_distance)

    @staticmethod
    def material_distance_matrix(candidates: Sequence[Mapping[str, Any]]) -> np.ndarray:
        """Return a symmetric categorical distance matrix over material sequences."""

        return _distance_matrix(candidates, material_distance)


def _normalize_candidates(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise TypeError("candidates must be a sequence of mappings")
    normalized: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            if hasattr(candidate, "model_dump"):
                candidate = candidate.model_dump(mode="json")
            else:
                raise TypeError("each candidate must be a mapping")
        normalized.append(dict(candidate))
    return normalized


def _distance_matrix(
    candidates: Sequence[Mapping[str, Any]],
    distance_fn: Callable[[Mapping[str, Any], Mapping[str, Any]], float],
) -> np.ndarray:
    normalized = _normalize_candidates(candidates)
    matrix = np.zeros((len(normalized), len(normalized)), dtype=np.float64)
    for row in range(len(normalized)):
        for column in range(row):
            value = float(distance_fn(normalized[row], normalized[column]))
            if not math.isfinite(value) or value < 0:
                raise ValueError("distance function must return finite non-negative values")
            matrix[row, column] = value
            matrix[column, row] = value
    return matrix


def _numeric_vector(candidate: Mapping[str, Any], keys: Sequence[str]) -> np.ndarray | None:
    for key in keys:
        value = candidate.get(key)
        if value is None:
            continue
        if isinstance(value, Mapping):
            flattened: list[float] = []
            for nested_key in sorted(value):
                nested = value[nested_key]
                if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
                    flattened.extend(float(item) for item in nested)
                elif isinstance(nested, (int, float)) and not isinstance(nested, bool):
                    flattened.append(float(nested))
            value = flattened
        if isinstance(value, np.ndarray) or (
            isinstance(value, Sequence) and not isinstance(value, (str, bytes))
        ):
            array = np.asarray(value, dtype=np.float64)
            if array.ndim == 1 and array.size and np.all(np.isfinite(array)):
                return array
    return None


def spectral_distance_matrix(candidates: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Compute pairwise Euclidean distances from candidate spectral vectors."""

    normalized = _normalize_candidates(candidates)
    vectors = [
        _numeric_vector(item, ("spectrum", "spectra", "spectrum_values", "response", "R"))
        for item in normalized
    ]
    if any(vector is None for vector in vectors):
        raise ValueError("spectral distance requires a finite spectrum vector per candidate")
    sizes = {int(vector.size) for vector in vectors if vector is not None}
    if len(sizes) != 1:
        raise ValueError("all candidate spectrum vectors must have the same length")
    matrix = np.zeros((len(vectors), len(vectors)), dtype=np.float64)
    for row in range(len(vectors)):
        for column in range(row):
            value = float(np.linalg.norm(vectors[row] - vectors[column]))
            matrix[row, column] = value
            matrix[column, row] = value
    return matrix


def structural_distance(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> float:
    """Compare normalized design vectors, with categorical fallback support."""

    first_vector = _numeric_vector(first, ("normalized_design", "normalized", "design"))
    second_vector = _numeric_vector(second, ("normalized_design", "normalized", "design"))
    if first_vector is not None and second_vector is not None:
        if first_vector.size != second_vector.size:
            raise ValueError("normalized design vectors must have the same length")
        return float(np.linalg.norm(first_vector - second_vector))
    first_values = first.get("values", first)
    second_values = second.get("values", second)
    if not isinstance(first_values, Mapping) or not isinstance(second_values, Mapping):
        raise ValueError("candidate has no structural representation")
    keys = sorted(set(first_values) | set(second_values))
    distance = 0.0
    for key in keys:
        left = first_values.get(key)
        right = second_values.get(key)
        if isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(
            right, (int, float)
        ) and not isinstance(right, bool):
            distance += (float(left) - float(right)) ** 2
        else:
            distance += float(left != right)
    return float(math.sqrt(distance))


def material_distance(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    """Return normalized Hamming distance between material sequences."""

    def sequence(candidate: Mapping[str, Any]) -> list[Any]:
        for key in ("material_sequence", "materials", "material_names"):
            value = candidate.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return list(value)
        values = candidate.get("values")
        if isinstance(values, Mapping):
            return [
                values[key]
                for key in sorted(values)
                if "material" in str(key).casefold()
            ]
        return []

    left, right = sequence(first), sequence(second)
    length = max(len(left), len(right))
    if length == 0:
        return 0.0
    mismatches = sum(
        left[index] != right[index]
        for index in range(length)
        if index >= len(left) or index >= len(right) or left[index] != right[index]
    )
    return float(mismatches / length)


def _minimum_positive_distance(
    candidates: Sequence[Mapping[str, Any]],
    distance_fn: Callable[[Mapping[str, Any], Mapping[str, Any]], float],
) -> float:
    values = [
        float(distance_fn(candidates[row], candidates[column]))
        for row in range(len(candidates))
        for column in range(row)
    ]
    return min(values) if values else 0.0


def _objective_spec(
    objectives: Mapping[str, str] | Sequence[str | Mapping[str, str]],
) -> tuple[list[str], dict[str, Literal["maximize", "minimize"]]]:
    if hasattr(objectives, "objectives"):
        objectives = tuple(getattr(objectives, "objectives"))
    if isinstance(objectives, Mapping):
        raw = list(objectives.items())
    else:
        raw = []
        for item in objectives:
            if isinstance(item, Mapping):
                raw.extend(item.items())
            elif hasattr(item, "name") and hasattr(item, "direction"):
                raw.append((str(item.name), str(item.direction)))
            else:
                raw.append((str(item), "maximize"))
    if not raw:
        raise ValueError("at least one Pareto objective is required")
    names: list[str] = []
    directions: dict[str, Literal["maximize", "minimize"]] = {}
    for name, direction in raw:
        if str(direction) not in {"maximize", "minimize"}:
            raise ValueError("Pareto objective directions must be maximize or minimize")
        if str(name) in directions:
            raise ValueError("Pareto objective names must be unique")
        names.append(str(name))
        directions[str(name)] = str(direction)  # type: ignore[assignment]
    return names, directions


def _objective_values(candidate: Mapping[str, Any], names: Sequence[str]) -> dict[str, float]:
    raw = candidate.get("objectives", candidate.get("objective_values", candidate.get("metrics")))
    if isinstance(raw, Mapping):
        values = {name: raw.get(name) for name in names}
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        if all(isinstance(item, Mapping) for item in raw):
            values = {
                name: next(
                    (
                        item.get("value")
                        for item in raw
                        if item.get("objective_name", item.get("name")) == name
                    ),
                    None,
                )
                for name in names
            }
        else:
            if len(raw) != len(names):
                raise ValueError("objective vector length does not match objective specification")
            values = dict(zip(names, raw, strict=True))
    else:
        values = {name: candidate.get(name) for name in names}
    result: dict[str, float] = {}
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"candidate objective {name!r} must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"candidate objective {name!r} must be finite")
        result[name] = number
    return result


def _dominates(
    first: Mapping[str, float],
    second: Mapping[str, float],
    directions: Mapping[str, Literal["maximize", "minimize"]],
) -> bool:
    first_values = [
        value if directions[name] == "maximize" else -value
        for name, value in first.items()
    ]
    second_values = [
        value if directions[name] == "maximize" else -value
        for name, value in second.items()
    ]
    return all(left >= right for left, right in zip(first_values, second_values)) and any(
        left > right for left, right in zip(first_values, second_values)
    )


def deduplicate(
    candidates: Sequence[Mapping[str, Any]],
    tolerance: float = 1e-3,
    distance_fn: Callable[[Mapping[str, Any], Mapping[str, Any]], float] | None = None,
) -> CandidateSet:
    """Functional wrapper for :meth:`CandidateSet.deduplicate`."""

    return CandidateSet.deduplicate(candidates, tolerance=tolerance, distance_fn=distance_fn)


def pareto_filter(
    candidates: Sequence[Mapping[str, Any]],
    objectives: Mapping[str, str] | Sequence[str | Mapping[str, str]],
) -> CandidateSet:
    """Functional wrapper for :meth:`CandidateSet.pareto_filter`."""

    return CandidateSet.pareto_filter(candidates, objectives)


def structural_distance_matrix(candidates: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Return a structural distance matrix without constructing a CandidateSet."""

    return _distance_matrix(candidates, structural_distance)


__all__ = [
    "CandidateSet",
    "CandidateSetProvenance",
    "DiversityMetric",
    "deduplicate",
    "material_distance",
    "pareto_filter",
    "spectral_distance_matrix",
    "structural_distance",
    "structural_distance_matrix",
]
