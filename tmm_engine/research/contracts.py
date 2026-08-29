"""Versioned contracts for deterministic optical design spaces.

These models describe research inputs and candidate identities only.  Candidate
scores never establish that a task or result is physically valid; physics
validity remains the responsibility of VeriTMM's existing validation and
certificate layers.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from ..protocol.models import SimulationTaskPayload
from ..schemas import SimulationTask, dataclass_to_dict

DESIGN_SPACE_SCHEMA_VERSION = "veritmm-design-space-v1"
CANDIDATE_SCHEMA_VERSION = "veritmm-design-candidate-v1"


def canonical_json(value: Any) -> str:
    """Return the canonical JSON representation used for stable identities."""

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def content_id(prefix: str, value: Any) -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest}"


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _nonempty(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


class ResearchModel(BaseModel):
    """Strict, immutable base model with a canonical JSON helper."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    def canonical_json(self) -> str:
        return canonical_json(self)


class VariableBase(ResearchModel):
    name: StrictStr
    layer_index: Annotated[StrictInt, Field(ge=0)]
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        return _nonempty(value, "variable name")


class ContinuousThicknessVariable(VariableBase):
    """Inclusive continuous thickness interval in nanometres."""

    kind: Literal["continuous_thickness"] = "continuous_thickness"
    lower_nm: StrictFloat
    upper_nm: StrictFloat

    @model_validator(mode="after")
    def _valid_bounds(self) -> "ContinuousThicknessVariable":
        lower = _finite_number(self.lower_nm, "lower_nm")
        upper = _finite_number(self.upper_nm, "upper_nm")
        if lower <= 0 or upper <= lower:
            raise ValueError("continuous thickness requires 0 < lower_nm < upper_nm")
        return self


class DiscreteThicknessVariable(VariableBase):
    """Finite ordered set of allowed thicknesses in nanometres."""

    kind: Literal["discrete_thickness"] = "discrete_thickness"
    values_nm: tuple[StrictFloat, ...] = Field(min_length=1)

    @field_validator("values_nm")
    @classmethod
    def _valid_values(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        normalized = tuple(_finite_number(value, "values_nm") for value in values)
        if any(value <= 0 for value in normalized):
            raise ValueError("discrete thickness values must be positive")
        if len(set(normalized)) != len(normalized):
            raise ValueError("discrete thickness values must be unique")
        return values


class MaterialOption(ResearchModel):
    """Named material selector preserved when constructing a ``LayerSpec``."""

    name: StrictStr
    material: StrictStr | None = None
    provider: StrictStr | None = None
    dataset_id: StrictStr | StrictInt | None = None
    constant_n: StrictFloat | None = None
    constant_k: StrictFloat = 0.0
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        return _nonempty(value, "material option name")

    @model_validator(mode="after")
    def _valid_selector(self) -> "MaterialOption":
        has_material = self.material is not None
        has_constant = self.constant_n is not None
        if has_material == has_constant:
            raise ValueError("material option requires exactly one of material or constant_n")
        k = _finite_number(self.constant_k, "constant_k")
        if k < 0:
            raise ValueError("constant_k must be non-negative")
        if has_material:
            _nonempty(self.material or "", "material")
            if k != 0:
                raise ValueError("constant_k is only valid with constant_n")
        else:
            n = _finite_number(self.constant_n, "constant_n")
            if n <= 0:
                raise ValueError("constant_n must be positive")
            if self.provider is not None or self.dataset_id is not None:
                raise ValueError("provider and dataset_id require a material selector")
        return self


class MaterialChoiceVariable(VariableBase):
    """Finite ordered choice of material selectors for one layer."""

    kind: Literal["material_choice"] = "material_choice"
    options: tuple[MaterialOption, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_options(self) -> "MaterialChoiceVariable":
        names = [option.name for option in self.options]
        if len(names) != len(set(names)):
            raise ValueError("material option names must be unique within a variable")
        return self


DesignVariable: TypeAlias = Annotated[
    ContinuousThicknessVariable | DiscreteThicknessVariable | MaterialChoiceVariable,
    Field(discriminator="kind"),
]


class DesignSpaceCapabilities(ResearchModel):
    """Explicit fixed-layer capability boundary for forward-compatible consumers."""

    variable_kinds: tuple[
        Literal["continuous_thickness", "discrete_thickness", "material_choice"], ...
    ] = (
        "continuous_thickness",
        "discrete_thickness",
        "material_choice",
    )
    samplers: tuple[Literal["random"], ...] = ("random",)
    variable_layer_count: Literal[False] = False


class DesignSpaceContract(ResearchModel):
    """Serializable definition of variables over an existing simulation task."""

    schema_version: Literal[DESIGN_SPACE_SCHEMA_VERSION] = DESIGN_SPACE_SCHEMA_VERSION
    design_space_id: StrictStr = ""
    base_task: SimulationTaskPayload
    variables: tuple[DesignVariable, ...] = Field(min_length=1)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    capabilities: DesignSpaceCapabilities = Field(default_factory=DesignSpaceCapabilities)

    @field_validator("base_task", mode="before")
    @classmethod
    def _serialize_base_task(cls, value: Any) -> Any:
        if isinstance(value, SimulationTask):
            value.validate()
            return dataclass_to_dict(value)
        return value

    @model_validator(mode="after")
    def _validate_space(self) -> "DesignSpaceContract":
        from ..task_io import simulation_task_from_dict

        base_task = simulation_task_from_dict(self.base_task.model_dump(mode="python"))
        names = [variable.name for variable in self.variables]
        if len(names) != len(set(names)):
            raise ValueError("design variable names must be unique")

        targets: set[tuple[int, str]] = set()
        layer_count = len(base_task.stack.layers)
        for variable in self.variables:
            if variable.layer_index >= layer_count:
                raise ValueError(
                    f"variable {variable.name!r} targets missing layer {variable.layer_index}"
                )
            property_name = (
                "material" if isinstance(variable, MaterialChoiceVariable) else "thickness_nm"
            )
            target = (variable.layer_index, property_name)
            if target in targets:
                raise ValueError(
                    "multiple variables cannot target the same layer property: "
                    f"layer {variable.layer_index} {property_name}"
                )
            targets.add(target)

            layer = base_task.stack.layers[variable.layer_index]
            if isinstance(variable, ContinuousThicknessVariable):
                self._check_declared_thickness_bounds(
                    layer.min_thickness_nm,
                    layer.max_thickness_nm,
                    variable.lower_nm,
                    variable.upper_nm,
                    variable.name,
                )
            elif isinstance(variable, DiscreteThicknessVariable):
                self._check_declared_thickness_bounds(
                    layer.min_thickness_nm,
                    layer.max_thickness_nm,
                    min(variable.values_nm),
                    max(variable.values_nm),
                    variable.name,
                )

        identity_payload = self.model_dump(mode="json", exclude={"design_space_id"})
        expected_id = content_id("design_space", identity_payload)
        if self.design_space_id and self.design_space_id != expected_id:
            raise ValueError("design_space_id does not match the contract content")
        object.__setattr__(self, "design_space_id", expected_id)
        return self

    @staticmethod
    def _check_declared_thickness_bounds(
        layer_min: float | None,
        layer_max: float | None,
        declared_min: float,
        declared_max: float,
        variable_name: str,
    ) -> None:
        if layer_min is not None and declared_min < layer_min:
            raise ValueError(
                f"variable {variable_name!r} falls below the layer min_thickness_nm"
            )
        if layer_max is not None and declared_max > layer_max:
            raise ValueError(
                f"variable {variable_name!r} exceeds the layer max_thickness_nm"
            )

    def to_simulation_task(self) -> SimulationTask:
        """Reconstruct and validate the existing public ``SimulationTask``."""

        from ..task_io import simulation_task_from_dict

        return simulation_task_from_dict(self.base_task.model_dump(mode="python"))


CandidateScalar: TypeAlias = StrictInt | StrictFloat | StrictStr


class DesignCandidate(ResearchModel):
    """Validated assignment and content-derived identity for a design space."""

    schema_version: Literal[CANDIDATE_SCHEMA_VERSION] = CANDIDATE_SCHEMA_VERSION
    design_space_id: StrictStr
    candidate_id: StrictStr
    values: dict[StrictStr, CandidateScalar]
    normalized_design: tuple[StrictFloat, ...]
    sample_index: StrictInt | None = None
    sampler: StrictStr | None = None
    seed: StrictInt | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("design_space_id", "candidate_id")
    @classmethod
    def _nonempty_id(cls, value: str) -> str:
        return _nonempty(value, "identity")

    @field_validator("normalized_design")
    @classmethod
    def _valid_normalized(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        for value in values:
            number = _finite_number(value, "normalized_design")
            if not 0 <= number <= 1:
                raise ValueError("normalized_design values must be in [0, 1]")
        return values

    @field_validator("sample_index")
    @classmethod
    def _valid_sample_index(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("sample_index must be non-negative")
        return value


__all__ = [
    "CANDIDATE_SCHEMA_VERSION",
    "DESIGN_SPACE_SCHEMA_VERSION",
    "ContinuousThicknessVariable",
    "DesignCandidate",
    "DesignSpaceCapabilities",
    "DesignSpaceContract",
    "DesignVariable",
    "DiscreteThicknessVariable",
    "MaterialChoiceVariable",
    "MaterialOption",
]
