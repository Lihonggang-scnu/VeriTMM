"""Optional ML and environment adapters over verified research contracts.

This module imports neither PyTorch nor Gymnasium at module import time.  It
contains no model, learning algorithm, solver, or certificate implementation.
"""

from __future__ import annotations

import importlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, TypeAlias, runtime_checkable

from pydantic import (
    Field,
    JsonValue,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    TypeAdapter,
    model_validator,
)

from .contracts import ResearchModel, content_id
from .dataset import DATASET_INDEX_SCHEMA_VERSION, DatasetRecord
from .design_space import DesignSpace
from .evaluator import ResearchEvaluator

RESEARCH_ACTION_SCHEMA_VERSION = "veritmm-research-action-v1"
ENVIRONMENT_STATE_SCHEMA_VERSION = "veritmm-design-environment-state-v1"
ENVIRONMENT_STEP_SCHEMA_VERSION = "veritmm-design-environment-step-v1"


class OptionalDependencyError(ImportError):
    """Raised when an optional adapter dependency is unavailable at use time."""


TargetValue: TypeAlias = float | int | Sequence[float | int]


class VerifiedTorchDataset:
    """Lazy PyTorch dataset exposing verified normalized designs and targets.

    Targets are supplied explicitly by candidate ID and labelled as an
    objective value or objective score.  The adapter never reads run spectra.
    """

    def __init__(
        self,
        source: Sequence[DatasetRecord] | str | Path,
        *,
        targets: Mapping[str, TargetValue],
        target_name: str,
        target_kind: Literal["objective_value", "objective_score"],
    ) -> None:
        if not isinstance(target_name, str) or not target_name.strip():
            raise ValueError("target_name must be a non-empty string")
        records = _load_dataset_records(source)
        source_ids = {record.candidate_id for record in records}
        unknown_targets = set(targets) - source_ids
        if unknown_targets:
            raise ValueError(
                f"targets contain unknown candidate IDs: {sorted(unknown_targets)}"
            )
        verified = tuple(
            record
            for record in records
            if (
                record.verification_status == "accepted"
                and record.physics_accepted
                and record.certificate_id is not None
                and record.run_id is not None
                and record.task_sha256 is not None
            )
        )
        if not verified:
            raise ValueError("VerifiedTorchDataset requires at least one accepted record")
        missing = [record.candidate_id for record in verified if record.candidate_id not in targets]
        if missing:
            raise ValueError(f"verified records are missing explicit targets: {missing}")
        features = [tuple(float(value) for value in record.normalized_design) for record in verified]
        if len({len(row) for row in features}) != 1:
            raise ValueError("verified feature vectors must have a consistent dimension")
        target_rows = [_target_row(targets[record.candidate_id]) for record in verified]
        if len({len(row) for row in target_rows}) != 1:
            raise ValueError("explicit targets must have a consistent dimension")
        try:
            torch = importlib.import_module("torch")
        except ImportError as exc:
            raise OptionalDependencyError(
                "VerifiedTorchDataset requires the optional 'torch' dependency"
            ) from exc
        self.features = torch.tensor(features, dtype=torch.float64)
        self.targets = torch.tensor(target_rows, dtype=torch.float64)
        self.candidate_ids = tuple(record.candidate_id for record in verified)
        self.run_ids = tuple(str(record.run_id) for record in verified)
        self.task_sha256s = tuple(str(record.task_sha256) for record in verified)
        self.certificate_ids = tuple(str(record.certificate_id) for record in verified)
        self.target_name = target_name
        self.target_kind = target_kind

    def __len__(self) -> int:
        return len(self.candidate_ids)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        return self.features[index], self.targets[index]


class ResearchActionBase(ResearchModel):
    schema_version: Literal[RESEARCH_ACTION_SCHEMA_VERSION] = (
        RESEARCH_ACTION_SCHEMA_VERSION
    )


class ChooseMaterialAction(ResearchActionBase):
    action: Literal["choose_material"] = "choose_material"
    variable: StrictStr
    choice: StrictStr


class ChooseThicknessAction(ResearchActionBase):
    action: Literal["choose_thickness"] = "choose_thickness"
    variable: StrictStr
    value_nm: StrictFloat


class AddLayerAction(ResearchActionBase):
    action: Literal["add_layer"] = "add_layer"


class RemoveLayerAction(ResearchActionBase):
    action: Literal["remove_layer"] = "remove_layer"
    layer_index: StrictInt | None = None


class StopAction(ResearchActionBase):
    action: Literal["stop"] = "stop"


ResearchAction: TypeAlias = Annotated[
    ChooseMaterialAction
    | ChooseThicknessAction
    | AddLayerAction
    | RemoveLayerAction
    | StopAction,
    Field(discriminator="action"),
]
_ACTION_ADAPTER = TypeAdapter(ResearchAction)


class EnvironmentIssue(ResearchModel):
    """Typed unsupported or invalid action explanation."""

    code: StrictStr
    message: StrictStr
    recoverable: StrictBool


class EnvironmentState(ResearchModel):
    """Serializable fixed-layer assignment state without a trajectory."""

    schema_version: Literal[ENVIRONMENT_STATE_SCHEMA_VERSION] = (
        ENVIRONMENT_STATE_SCHEMA_VERSION
    )
    state_id: StrictStr = ""
    episode_id: StrictStr
    design_space_id: StrictStr
    seed: StrictInt
    assignments: dict[StrictStr, StrictInt | StrictFloat | StrictStr] = Field(
        default_factory=dict
    )
    assigned_variables: tuple[StrictStr, ...] = ()
    complete: StrictBool = False
    terminated: StrictBool = False
    step_count: StrictInt = 0

    @model_validator(mode="after")
    def _valid_state(self) -> "EnvironmentState":
        if self.step_count < 0:
            raise ValueError("environment step_count must be non-negative")
        if set(self.assignments) != set(self.assigned_variables):
            raise ValueError("environment assigned-variable metadata is inconsistent")
        expected = content_id(
            "environment_state",
            self.model_dump(mode="json", exclude={"state_id"}),
        )
        if self.state_id and self.state_id != expected:
            raise ValueError("environment state_id does not match state content")
        object.__setattr__(self, "state_id", expected)
        return self


class EnvironmentStep(ResearchModel):
    """Bounded transition result; reward is never a physics-validity signal."""

    schema_version: Literal[ENVIRONMENT_STEP_SCHEMA_VERSION] = (
        ENVIRONMENT_STEP_SCHEMA_VERSION
    )
    status: Literal["assigned", "evaluated", "unsupported", "invalid"]
    state: EnvironmentState
    reward: StrictFloat | None = None
    terminated: StrictBool
    issue: EnvironmentIssue | None = None
    info: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _valid_step(self) -> "EnvironmentStep":
        if self.terminated != self.state.terminated:
            raise ValueError("environment transition termination is inconsistent")
        if self.status in {"unsupported", "invalid"} and self.issue is None:
            raise ValueError("unsupported or invalid transition requires a typed issue")
        if self.status == "evaluated" and self.issue is not None:
            raise ValueError("evaluated transition cannot contain an action issue")
        return self


@runtime_checkable
class ResearchEnvironment(Protocol):
    """Dependency-free environment protocol for fixed-layer research spaces."""

    def reset(self, *, seed: int | None = None) -> EnvironmentState:
        """Reset to an empty deterministic assignment state."""

    def step(self, action: ResearchAction | Mapping[str, Any]) -> EnvironmentStep:
        """Apply one versioned action."""

    def state(self) -> EnvironmentState:
        """Return the current immutable serializable state."""


class DesignSpaceEnvironment:
    """Minimal fixed-layer environment whose stop action uses ResearchEvaluator."""

    def __init__(
        self,
        design_space: DesignSpace,
        evaluator: ResearchEvaluator,
        *,
        seed: int = 0,
    ) -> None:
        if not isinstance(design_space, DesignSpace):
            raise TypeError("design_space must be a DesignSpace")
        if not isinstance(evaluator, ResearchEvaluator):
            raise TypeError("evaluator must be a ResearchEvaluator")
        if design_space.design_space_id != evaluator.design_space.design_space_id:
            raise ValueError("environment design space does not match evaluator")
        self.design_space = design_space
        self.evaluator = evaluator
        self._initial_seed = self._validate_seed(seed)
        self._state = self._empty_state(self._initial_seed)

    def reset(self, *, seed: int | None = None) -> EnvironmentState:
        active_seed = self._initial_seed if seed is None else self._validate_seed(seed)
        self._state = self._empty_state(active_seed)
        return self._state

    def state(self) -> EnvironmentState:
        return self._state

    def state_dict(self) -> dict[str, Any]:
        return self._state.model_dump(mode="json")

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("environment state must be a mapping")
        try:
            restored = EnvironmentState.model_validate_json(
                json.dumps(dict(state), ensure_ascii=False, allow_nan=False)
            )
        except Exception as exc:
            raise ValueError(f"environment state is corrupt: {exc}") from exc
        if restored.design_space_id != self.design_space.design_space_id:
            raise ValueError("environment state design-space binding mismatch")
        if restored.episode_id != self._episode_id(restored.seed):
            raise ValueError("environment episode identity is invalid")
        expected_order = tuple(
            variable.name
            for variable in self.design_space.variables
            if variable.name in restored.assignments
        )
        if restored.assigned_variables != expected_order:
            raise ValueError("environment assignment ordering is invalid")
        self._validate_assignments(restored.assignments, complete=restored.complete)
        if restored.complete != (
            len(restored.assignments) == len(self.design_space.variables)
        ):
            raise ValueError("environment completeness metadata is invalid")
        self._state = restored

    def step(self, action: ResearchAction | Mapping[str, Any]) -> EnvironmentStep:
        try:
            parsed = (
                action
                if isinstance(action, ResearchActionBase)
                else _ACTION_ADAPTER.validate_python(action)
            )
        except Exception as exc:
            return self._issue_step(
                "invalid",
                code="invalid_action_schema",
                message=str(exc),
                recoverable=True,
            )
        if self._state.terminated:
            return self._issue_step(
                "invalid",
                code="episode_terminated",
                message="reset is required before another action",
                recoverable=True,
            )
        if isinstance(parsed, (AddLayerAction, RemoveLayerAction)):
            return self._issue_step(
                "unsupported",
                code="variable_layer_count_unsupported",
                message="Fixed-layer design spaces do not support add/remove layer",
                recoverable=False,
            )
        if isinstance(parsed, StopAction):
            return self._stop()
        return self._assign(parsed)

    def _assign(
        self, action: ChooseMaterialAction | ChooseThicknessAction
    ) -> EnvironmentStep:
        variable = next(
            (
                item
                for item in self.design_space.variables
                if item.name == action.variable
            ),
            None,
        )
        if variable is None:
            return self._issue_step(
                "invalid",
                code="unknown_variable",
                message=f"unknown design variable: {action.variable}",
                recoverable=True,
            )
        if variable.name in self._state.assignments:
            return self._issue_step(
                "invalid",
                code="variable_already_assigned",
                message=f"design variable is already assigned: {variable.name}",
                recoverable=True,
            )
        from .contracts import (
            ContinuousThicknessVariable,
            DiscreteThicknessVariable,
            MaterialChoiceVariable,
        )

        value: int | float | str
        if isinstance(action, ChooseMaterialAction):
            if not isinstance(variable, MaterialChoiceVariable):
                return self._issue_step(
                    "invalid",
                    code="action_variable_type_mismatch",
                    message="choose_material requires a material-choice variable",
                    recoverable=True,
                )
            choices = {option.name for option in variable.options}
            if action.choice not in choices:
                return self._issue_step(
                    "invalid",
                    code="invalid_material_choice",
                    message=f"material choice is not allowed: {action.choice}",
                    recoverable=True,
                )
            value = action.choice
        else:
            if not isinstance(
                variable, (ContinuousThicknessVariable, DiscreteThicknessVariable)
            ):
                return self._issue_step(
                    "invalid",
                    code="action_variable_type_mismatch",
                    message="choose_thickness requires a thickness variable",
                    recoverable=True,
                )
            value = float(action.value_nm)
            try:
                _validate_variable_value(variable, value)
            except Exception as exc:
                return self._issue_step(
                    "invalid",
                    code="invalid_thickness_choice",
                    message=str(exc),
                    recoverable=True,
                )
        assignments = dict(self._state.assignments)
        assignments[variable.name] = value
        self._state = self._make_state(
            assignments=assignments,
            terminated=False,
            step_count=self._state.step_count + 1,
        )
        return EnvironmentStep(
            status="assigned",
            state=self._state,
            terminated=False,
            info={
                "complete": self._state.complete,
                "reward_is_not_physics_validity": True,
            },
        )

    def _stop(self) -> EnvironmentStep:
        if not self._state.complete:
            missing = [
                variable.name
                for variable in self.design_space.variables
                if variable.name not in self._state.assignments
            ]
            return self._issue_step(
                "invalid",
                code="incomplete_design",
                message=f"stop requires assignments for: {missing}",
                recoverable=True,
            )
        candidate = self.design_space.candidate(
            self._state.assignments,
            sampler="environment",
            seed=self._state.seed,
            metadata={"episode_id": self._state.episode_id},
        )
        record = self.evaluator.evaluate(
            candidate,
            experiment_metadata={"environment_episode_id": self._state.episode_id},
        )
        verified = (
            record.status == "completed"
            and record.physics_accepted
            and record.certificate_id is not None
            and record.total_score is not None
        )
        reward = float(record.total_score) if verified else 0.0
        self._state = self._make_state(
            assignments=self._state.assignments,
            terminated=True,
            step_count=self._state.step_count + 1,
        )
        return EnvironmentStep(
            status="evaluated",
            state=self._state,
            reward=reward,
            terminated=True,
            info={
                "candidate_id": candidate.candidate_id,
                "status": record.status,
                "evaluation_status": record.status,
                "physics_accepted": record.physics_accepted,
                "certificate_id": record.certificate_id,
                "run_id": record.run_id,
                "task_sha256": record.task_sha256,
                "feasible": record.feasible,
                "reward_is_not_physics_validity": True,
            },
        )

    def _issue_step(
        self,
        status: Literal["unsupported", "invalid"],
        *,
        code: str,
        message: str,
        recoverable: bool,
    ) -> EnvironmentStep:
        terminated = self._state.terminated
        self._state = self._make_state(
            assignments=self._state.assignments,
            terminated=terminated,
            step_count=self._state.step_count + 1,
        )
        return EnvironmentStep(
            status=status,
            state=self._state,
            terminated=terminated,
            issue=EnvironmentIssue(
                code=code,
                message=_bounded_message(message),
                recoverable=recoverable,
            ),
            info={"reward_is_not_physics_validity": True},
        )

    def _empty_state(self, seed: int) -> EnvironmentState:
        return EnvironmentState(
            episode_id=self._episode_id(seed),
            design_space_id=self.design_space.design_space_id,
            seed=seed,
        )

    def _make_state(
        self,
        *,
        assignments: Mapping[str, Any],
        terminated: bool,
        step_count: int,
    ) -> EnvironmentState:
        ordered_names = tuple(
            variable.name
            for variable in self.design_space.variables
            if variable.name in assignments
        )
        return EnvironmentState(
            episode_id=self._state.episode_id,
            design_space_id=self.design_space.design_space_id,
            seed=self._state.seed,
            assignments=dict(assignments),
            assigned_variables=ordered_names,
            complete=len(assignments) == len(self.design_space.variables),
            terminated=terminated,
            step_count=step_count,
        )

    def _validate_assignments(
        self, assignments: Mapping[str, Any], *, complete: bool
    ) -> None:
        expected = {variable.name for variable in self.design_space.variables}
        if not set(assignments) <= expected:
            raise ValueError("environment state contains unknown variables")
        if complete:
            self.design_space.candidate(assignments)
            return
        for variable in self.design_space.variables:
            if variable.name in assignments:
                _validate_variable_value(variable, assignments[variable.name])

    def _episode_id(self, seed: int) -> str:
        return content_id(
            "environment_episode",
            {"design_space_id": self.design_space.design_space_id, "seed": seed},
        )

    @staticmethod
    def _validate_seed(seed: Any) -> int:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("environment seed must be an integer")
        return seed


def _load_dataset_records(
    source: Sequence[DatasetRecord] | str | Path,
) -> tuple[DatasetRecord, ...]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        records: list[DatasetRecord] = []
        indices: set[int] = set()
        dataset_id: str | None = None
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ValueError(f"dataset index is unreadable: {exc}") from exc
        for line_number, line in enumerate(lines, start=1):
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError("entry must be an object")
                if payload.get("schema_version") != DATASET_INDEX_SCHEMA_VERSION:
                    raise ValueError("schema_version is invalid")
                index = payload.get("index")
                if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                    raise ValueError("sample index is invalid")
                if not isinstance(payload.get("dataset_id"), str):
                    raise ValueError("dataset identity is invalid")
                if not isinstance(payload.get("candidate_id"), str):
                    raise ValueError("candidate identity is invalid")
                record = DatasetRecord.model_validate_json(
                    json.dumps(payload["record"], ensure_ascii=False, allow_nan=False)
                )
                if (
                    record.dataset_id != payload["dataset_id"]
                    or record.sample_index != index
                    or record.candidate_id != payload["candidate_id"]
                ):
                    raise ValueError("record binding is invalid")
                if index in indices:
                    raise ValueError("sample index is duplicated")
                if dataset_id is not None and record.dataset_id != dataset_id:
                    raise ValueError("records belong to different datasets")
            except Exception as exc:
                raise ValueError(f"dataset index line {line_number} is invalid") from exc
            records.append(record)
            indices.add(index)
            dataset_id = record.dataset_id
        result = tuple(records)
    else:
        result = tuple(source)
        if any(not isinstance(record, DatasetRecord) for record in result):
            raise TypeError("VerifiedTorchDataset source must contain DatasetRecord values")
    ids = [record.candidate_id for record in result]
    if len(ids) != len(set(ids)):
        raise ValueError("dataset source contains duplicate candidate IDs")
    return result


def _target_row(value: TargetValue) -> tuple[float, ...]:
    raw = (value,) if isinstance(value, (int, float)) and not isinstance(value, bool) else value
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise TypeError("target must be a number or finite numeric sequence")
    row = tuple(float(item) for item in raw)
    if not row or not all(math.isfinite(item) for item in row):
        raise ValueError("target must be a non-empty finite numeric sequence")
    return row


def _validate_variable_value(variable: Any, value: Any) -> None:
    from .contracts import (
        ContinuousThicknessVariable,
        DiscreteThicknessVariable,
        MaterialChoiceVariable,
    )

    if isinstance(variable, ContinuousThicknessVariable):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("continuous thickness action must be numeric")
        number = float(value)
        if not math.isfinite(number) or not variable.lower_nm <= number <= variable.upper_nm:
            raise ValueError("continuous thickness action is outside its finite bounds")
        return
    if isinstance(variable, DiscreteThicknessVariable):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("discrete thickness action must be numeric")
        number = float(value)
        if not math.isfinite(number) or number not in variable.values_nm:
            raise ValueError("discrete thickness action is not an allowed value")
        return
    if isinstance(variable, MaterialChoiceVariable):
        if not isinstance(value, str) or value not in {
            option.name for option in variable.options
        }:
            raise ValueError("material action is not an allowed choice")
        return
    raise TypeError("unsupported design variable contract")


def _bounded_message(value: str, limit: int = 512) -> str:
    text = str(value)
    return text if len(text) <= limit else f"{text[: limit - 14]}...[truncated]"


__all__ = [
    "ENVIRONMENT_STATE_SCHEMA_VERSION",
    "ENVIRONMENT_STEP_SCHEMA_VERSION",
    "RESEARCH_ACTION_SCHEMA_VERSION",
    "AddLayerAction",
    "ChooseMaterialAction",
    "ChooseThicknessAction",
    "DesignSpaceEnvironment",
    "EnvironmentIssue",
    "EnvironmentState",
    "EnvironmentStep",
    "OptionalDependencyError",
    "RemoveLayerAction",
    "ResearchAction",
    "ResearchEnvironment",
    "StopAction",
    "VerifiedTorchDataset",
]
