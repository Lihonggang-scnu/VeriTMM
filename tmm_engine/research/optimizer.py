"""Algorithm-neutral optimizer contracts and deterministic random search.

Optimizer ranking consumes verified evaluation metadata only.  It never
creates, mutates, or upgrades a physics acceptance certificate.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import StrictBool, StrictFloat, StrictInt, StrictStr, model_validator

from .contracts import DesignCandidate, ResearchModel
from .design_space import DesignSpace
from .evaluator import EvaluationRecord

RANDOM_SEARCH_STATE_SCHEMA_VERSION = "veritmm-random-search-state-v1"
DEFAULT_MAX_OBSERVATIONS = 10_000
DEFAULT_MAX_PENDING = 1_024
HARD_MAX_OBSERVATIONS = 10_000
HARD_MAX_PENDING = 1_024


class OptimizerObservation(ResearchModel):
    """Bounded ranking metadata copied from one ``EvaluationRecord``."""

    candidate_id: StrictStr
    candidate: DesignCandidate
    status: Literal["completed", "failed"]
    physics_accepted: StrictBool
    certificate_id: StrictStr | None
    run_id: StrictStr | None
    task_sha256: StrictStr | None
    feasible: StrictBool | None
    total_score: StrictFloat | None

    @model_validator(mode="after")
    def _valid_candidate_binding(self) -> "OptimizerObservation":
        if self.candidate.candidate_id != self.candidate_id:
            raise ValueError("optimizer observation candidate identity mismatch")
        if self.total_score is not None and not math.isfinite(self.total_score):
            raise ValueError("optimizer observation total_score must be finite")
        return self

    @property
    def rankable(self) -> bool:
        return (
            self.status == "completed"
            and self.physics_accepted
            and self.certificate_id is not None
            and self.feasible is not None
            and self.total_score is not None
            and math.isfinite(self.total_score)
        )


class RandomSearchState(ResearchModel):
    """Versioned bounded persistence schema for ``RandomSearchAdapter``."""

    schema_version: Literal[RANDOM_SEARCH_STATE_SCHEMA_VERSION] = (
        RANDOM_SEARCH_STATE_SCHEMA_VERSION
    )
    adapter: Literal["random_search"] = "random_search"
    design_space_id: StrictStr
    seed: StrictInt
    cursor: StrictInt
    max_observations: StrictInt
    max_pending: StrictInt
    pending: tuple[DesignCandidate, ...] = ()
    observations: tuple[OptimizerObservation, ...] = ()
    best_candidate_id: StrictStr | None = None

    @model_validator(mode="after")
    def _valid_state(self) -> "RandomSearchState":
        if self.cursor < 0:
            raise ValueError("optimizer cursor must be non-negative")
        if self.max_observations < 1 or self.max_pending < 1:
            raise ValueError("optimizer state limits must be positive")
        if self.max_observations > HARD_MAX_OBSERVATIONS:
            raise ValueError("optimizer observation limit exceeds its hard bound")
        if self.max_pending > HARD_MAX_PENDING:
            raise ValueError("optimizer pending limit exceeds its hard bound")
        if len(self.pending) > self.max_pending:
            raise ValueError("optimizer pending state exceeds its configured bound")
        if len(self.observations) > self.max_observations:
            raise ValueError("optimizer observations exceed their configured bound")
        pending_ids = [item.candidate_id for item in self.pending]
        observation_ids = [item.candidate_id for item in self.observations]
        if len(pending_ids) != len(set(pending_ids)):
            raise ValueError("optimizer state contains duplicate pending candidates")
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("optimizer state contains duplicate observations")
        if set(pending_ids) & set(observation_ids):
            raise ValueError("optimizer candidate cannot be pending and observed")
        return self


@runtime_checkable
class OptimizerAdapter(Protocol):
    """Algorithm-neutral ask/tell optimizer surface."""

    def ask(self, count: int) -> tuple[DesignCandidate, ...]:
        """Return new pending candidates."""

    def tell(self, records: Iterable[EvaluationRecord]) -> None:
        """Record evaluations for previously asked candidates."""

    def best(self) -> OptimizerObservation | None:
        """Return the best verified compact observation, if one exists."""

    def state_dict(self) -> dict[str, Any]:
        """Return versioned JSON-safe state."""

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore state after strict identity and integrity validation."""


class RandomSearchAdapter:
    """Reference deterministic random search with bounded persistent state."""

    def __init__(
        self,
        design_space: DesignSpace,
        *,
        seed: int,
        max_observations: int = DEFAULT_MAX_OBSERVATIONS,
        max_pending: int = DEFAULT_MAX_PENDING,
    ) -> None:
        if not isinstance(design_space, DesignSpace):
            raise TypeError("design_space must be a DesignSpace")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if (
            isinstance(max_observations, bool)
            or not isinstance(max_observations, int)
            or not 1 <= max_observations <= HARD_MAX_OBSERVATIONS
        ):
            raise ValueError(
                f"max_observations must be between 1 and {HARD_MAX_OBSERVATIONS}"
            )
        if (
            isinstance(max_pending, bool)
            or not isinstance(max_pending, int)
            or not 1 <= max_pending <= HARD_MAX_PENDING
        ):
            raise ValueError(f"max_pending must be between 1 and {HARD_MAX_PENDING}")
        self.design_space = design_space
        self.seed = seed
        self.max_observations = max_observations
        self.max_pending = max_pending
        self._cursor = 0
        self._pending: dict[str, DesignCandidate] = {}
        self._observations: dict[str, OptimizerObservation] = {}
        self._best_candidate_id: str | None = None

    def ask(self, count: int) -> tuple[DesignCandidate, ...]:
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("ask count must be a positive integer")
        if len(self._pending) + count > self.max_pending:
            raise ValueError("ask would exceed the bounded pending-candidate limit")
        unavailable = set(self._pending) | set(self._observations)
        candidates: list[DesignCandidate] = []
        attempts = 0
        cursor = self._cursor
        max_attempts = max(10_000, count * 1_000)
        while len(candidates) < count:
            if attempts >= max_attempts:
                raise ValueError(
                    "random search could not find enough unique candidates; "
                    "the finite design space may be exhausted"
                )
            sample_index = cursor
            cursor += 1
            attempts += 1
            sampled = self.design_space.sample_indices(
                (sample_index,), seed=self.seed
            )[0]
            candidate = self.design_space.candidate(
                sampled.values,
                sample_index=sample_index,
                sampler="random_search",
                seed=self.seed,
                metadata={"optimizer_adapter": "random_search"},
            )
            if candidate.candidate_id in unavailable:
                continue
            unavailable.add(candidate.candidate_id)
            candidates.append(candidate)
        self._cursor = cursor
        self._pending.update((item.candidate_id, item) for item in candidates)
        return tuple(candidates)

    def tell(self, records: Iterable[EvaluationRecord]) -> None:
        record_list = tuple(records)
        if not record_list:
            raise ValueError("tell requires at least one EvaluationRecord")
        if len(self._observations) + len(record_list) > self.max_observations:
            raise ValueError("tell would exceed the bounded observation limit")
        seen: set[str] = set()
        for record in record_list:
            if not isinstance(record, EvaluationRecord):
                raise TypeError("tell accepts EvaluationRecord instances only")
            candidate_id = record.candidate_id
            if candidate_id in seen:
                raise ValueError("tell contains a duplicate candidate")
            seen.add(candidate_id)
            if candidate_id in self._observations:
                raise ValueError("candidate has already been told")
            if candidate_id not in self._pending:
                raise ValueError("tell candidate was not returned by ask")
            if record.design_space_id != self.design_space.design_space_id:
                raise ValueError("evaluation belongs to a different design space")

        for record in record_list:
            observation = OptimizerObservation(
                candidate_id=record.candidate_id,
                candidate=self._pending[record.candidate_id],
                status=record.status,
                physics_accepted=record.physics_accepted,
                certificate_id=record.certificate_id,
                run_id=record.run_id,
                task_sha256=record.task_sha256,
                feasible=record.feasible,
                total_score=record.total_score,
            )
            self._observations[record.candidate_id] = observation
            self._pending.pop(record.candidate_id)
        self._best_candidate_id = self._compute_best_candidate_id()

    def best(self) -> OptimizerObservation | None:
        if self._best_candidate_id is None:
            return None
        return self._observations[self._best_candidate_id]

    def state_dict(self) -> dict[str, Any]:
        state = RandomSearchState(
            design_space_id=self.design_space.design_space_id,
            seed=self.seed,
            cursor=self._cursor,
            max_observations=self.max_observations,
            max_pending=self.max_pending,
            pending=tuple(self._pending.values()),
            observations=tuple(self._observations.values()),
            best_candidate_id=self._best_candidate_id,
        )
        return state.model_dump(mode="json")

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("optimizer state must be a mapping")
        try:
            restored = RandomSearchState.model_validate_json(
                json.dumps(dict(state), ensure_ascii=False, allow_nan=False)
            )
        except Exception as exc:
            raise ValueError(f"random-search state is corrupt: {exc}") from exc
        if restored.design_space_id != self.design_space.design_space_id:
            raise ValueError("random-search state design-space binding mismatch")
        if restored.seed != self.seed:
            raise ValueError("random-search state seed mismatch")
        if restored.max_observations != self.max_observations:
            raise ValueError("random-search observation limit mismatch")
        if restored.max_pending != self.max_pending:
            raise ValueError("random-search pending limit mismatch")

        bound_candidates = (
            *restored.pending,
            *(item.candidate for item in restored.observations),
        )
        for candidate in bound_candidates:
            validated = self.design_space.validate_candidate(candidate)
            if (
                validated.sample_index is None
                or validated.sample_index >= restored.cursor
                or validated.seed != self.seed
                or validated.sampler != "random_search"
            ):
                raise ValueError("random-search pending provenance is invalid")
            sampled = self.design_space.sample_indices(
                (validated.sample_index,), seed=self.seed
            )[0]
            expected = self.design_space.candidate(
                sampled.values,
                sample_index=validated.sample_index,
                sampler="random_search",
                seed=self.seed,
                metadata={"optimizer_adapter": "random_search"},
            )
            if expected != validated:
                raise ValueError("random-search pending candidate content is invalid")

        observations = {
            item.candidate_id: item for item in restored.observations
        }
        computed_best = self._rank_observations(observations.values())
        if restored.best_candidate_id != computed_best:
            raise ValueError("random-search best identity is inconsistent")
        self._cursor = restored.cursor
        self._pending = {item.candidate_id: item for item in restored.pending}
        self._observations = observations
        self._best_candidate_id = restored.best_candidate_id

    def _compute_best_candidate_id(self) -> str | None:
        return self._rank_observations(self._observations.values())

    @staticmethod
    def _rank_observations(
        observations: Iterable[OptimizerObservation],
    ) -> str | None:
        rankable = [item for item in observations if item.rankable]
        if not rankable:
            return None
        return min(
            rankable,
            key=lambda item: (
                0 if item.feasible else 1,
                -float(item.total_score),
                item.candidate_id,
            ),
        ).candidate_id


__all__ = [
    "DEFAULT_MAX_OBSERVATIONS",
    "DEFAULT_MAX_PENDING",
    "HARD_MAX_OBSERVATIONS",
    "HARD_MAX_PENDING",
    "RANDOM_SEARCH_STATE_SCHEMA_VERSION",
    "OptimizerAdapter",
    "OptimizerObservation",
    "RandomSearchAdapter",
    "RandomSearchState",
]
