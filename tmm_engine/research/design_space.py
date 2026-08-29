"""Deterministic runtime operations over research design-space contracts."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import replace
from typing import Any, Iterable, Mapping, Protocol, runtime_checkable

from ..schemas import LayerSpec, SimulationTask
from .contracts import (
    ContinuousThicknessVariable,
    DesignCandidate,
    DesignSpaceContract,
    DesignVariable,
    DiscreteThicknessVariable,
    MaterialChoiceVariable,
    MaterialOption,
    canonical_json,
    content_id,
)


@runtime_checkable
class DesignSampler(Protocol):
    """Extension hook for deterministic indexed samplers.

    Implementations must make each output a pure function of the supplied
    design-space identity, seed, sample index, and variable index.
    """

    name: str

    def value_for(
        self,
        variable: DesignVariable,
        *,
        design_space_id: str,
        seed: int,
        sample_index: int,
        variable_index: int,
    ) -> int | float | str:
        """Return one legal value for ``variable`` at an indexed sample."""


class RandomSampler:
    """Seeded random sampler with no mutable sequence state."""

    name = "random"

    @staticmethod
    def _rng(
        design_space_id: str,
        seed: int,
        sample_index: int,
        variable_index: int,
    ) -> random.Random:
        payload = canonical_json(
            {
                "design_space_id": design_space_id,
                "seed": seed,
                "sample_index": sample_index,
                "variable_index": variable_index,
                "sampler": RandomSampler.name,
            }
        )
        digest = hashlib.sha256(payload.encode("utf-8")).digest()
        return random.Random(int.from_bytes(digest, "big"))

    def value_for(
        self,
        variable: DesignVariable,
        *,
        design_space_id: str,
        seed: int,
        sample_index: int,
        variable_index: int,
    ) -> int | float | str:
        rng = self._rng(design_space_id, seed, sample_index, variable_index)
        if isinstance(variable, ContinuousThicknessVariable):
            return variable.lower_nm + rng.random() * (
                variable.upper_nm - variable.lower_nm
            )
        if isinstance(variable, DiscreteThicknessVariable):
            return variable.values_nm[rng.randrange(len(variable.values_nm))]
        return variable.options[rng.randrange(len(variable.options))].name


class DesignSpace:
    """Validated, deterministic operations for a ``DesignSpaceContract``.

    This class only creates simulation tasks and research identities.  It does
    not run a solver or issue a physics-validity certificate.
    """

    def __init__(self, contract: DesignSpaceContract | Mapping[str, Any]) -> None:
        self.contract = (
            contract
            if isinstance(contract, DesignSpaceContract)
            else DesignSpaceContract.model_validate(contract)
        )

    @property
    def design_space_id(self) -> str:
        return self.contract.design_space_id

    @property
    def variables(self) -> tuple[DesignVariable, ...]:
        return self.contract.variables

    def candidate(
        self,
        values: Mapping[str, Any],
        *,
        sample_index: int | None = None,
        sampler: str | None = None,
        seed: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> DesignCandidate:
        """Validate an assignment and derive its stable content identity."""

        if not isinstance(values, Mapping):
            raise TypeError("candidate values must be a mapping")
        expected = [variable.name for variable in self.variables]
        provided = set(values)
        missing = [name for name in expected if name not in provided]
        extra = sorted(provided - set(expected))
        if missing or extra:
            raise ValueError(f"candidate variable mismatch: missing={missing}, extra={extra}")
        if sample_index is not None:
            self._validate_index(sample_index, "sample_index")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise TypeError("seed must be an integer")

        normalized_values: dict[str, int | float | str] = {}
        normalized_design: list[float] = []
        for variable in self.variables:
            value, normalized = self._normalize_value(variable, values[variable.name])
            normalized_values[variable.name] = value
            normalized_design.append(normalized)

        identity_payload = {
            "design_space_id": self.design_space_id,
            "normalized_assignment": [
                [variable.name, normalized_design[index]]
                for index, variable in enumerate(self.variables)
            ],
        }
        return DesignCandidate(
            design_space_id=self.design_space_id,
            candidate_id=content_id("candidate", identity_payload),
            values=normalized_values,
            normalized_design=tuple(normalized_design),
            sample_index=sample_index,
            sampler=sampler,
            seed=seed,
            metadata={} if metadata is None else dict(metadata),
        )

    def validate_candidate(self, candidate: DesignCandidate | Mapping[str, Any]) -> DesignCandidate:
        """Validate candidate values and reject identities forged for another space."""

        supplied = (
            candidate
            if isinstance(candidate, DesignCandidate)
            else DesignCandidate.model_validate(candidate)
        )
        if supplied.design_space_id != self.design_space_id:
            raise ValueError("candidate belongs to a different design space")
        expected = self.candidate(
            supplied.values,
            sample_index=supplied.sample_index,
            sampler=supplied.sampler,
            seed=supplied.seed,
            metadata=supplied.metadata,
        )
        if supplied.candidate_id != expected.candidate_id:
            raise ValueError("candidate_id does not match candidate content")
        if supplied.normalized_design != expected.normalized_design:
            raise ValueError("normalized_design does not match candidate values")
        return supplied

    def to_simulation_task(
        self, candidate: DesignCandidate | Mapping[str, Any]
    ) -> SimulationTask:
        """Apply a candidate to a copy of the existing ``SimulationTask``."""

        if isinstance(candidate, DesignCandidate) or "candidate_id" in candidate:
            validated = self.validate_candidate(candidate)
        else:
            validated = self.candidate(candidate)

        base_task = self.contract.to_simulation_task()
        layers = list(base_task.stack.layers)
        for variable in self.variables:
            value = validated.values[variable.name]
            layer = layers[variable.layer_index]
            if isinstance(variable, (ContinuousThicknessVariable, DiscreteThicknessVariable)):
                layers[variable.layer_index] = replace(layer, thickness_nm=float(value))
            else:
                option = self._material_option(variable, value)
                layers[variable.layer_index] = self._apply_material_option(layer, option)

        task = replace(base_task, stack=replace(base_task.stack, layers=tuple(layers)))
        task.validate()
        return task

    def candidate_from_normalized(
        self,
        normalized_design: Iterable[float],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> DesignCandidate:
        """Decode an ordered unit-vector into a legal candidate assignment."""

        vector = tuple(normalized_design)
        if len(vector) != len(self.variables):
            raise ValueError("normalized design length must match the variable count")
        values: dict[str, int | float | str] = {}
        for variable, raw in zip(self.variables, vector):
            normalized = self._unit_value(raw)
            if isinstance(variable, ContinuousThicknessVariable):
                values[variable.name] = variable.lower_nm + normalized * (
                    variable.upper_nm - variable.lower_nm
                )
            elif isinstance(variable, DiscreteThicknessVariable):
                index = self._decode_index(normalized, len(variable.values_nm))
                values[variable.name] = variable.values_nm[index]
            else:
                index = self._decode_index(normalized, len(variable.options))
                values[variable.name] = variable.options[index].name
        return self.candidate(values, metadata=metadata)

    def sample_indices(
        self,
        indices: Iterable[int],
        *,
        seed: int,
        sampler: DesignSampler | None = None,
    ) -> tuple[DesignCandidate, ...]:
        """Sample explicit indices reproducibly, independent of prior calls."""

        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        active_sampler = RandomSampler() if sampler is None else sampler
        if not isinstance(active_sampler, DesignSampler):
            raise TypeError("sampler must implement the DesignSampler protocol")
        sampler_name = getattr(active_sampler, "name", "")
        if not isinstance(sampler_name, str) or not sampler_name.strip():
            raise ValueError("sampler name must be a non-empty string")

        candidates: list[DesignCandidate] = []
        for sample_index in indices:
            self._validate_index(sample_index, "sample index")
            values = {
                variable.name: active_sampler.value_for(
                    variable,
                    design_space_id=self.design_space_id,
                    seed=seed,
                    sample_index=sample_index,
                    variable_index=variable_index,
                )
                for variable_index, variable in enumerate(self.variables)
            }
            candidates.append(
                self.candidate(
                    values,
                    sample_index=sample_index,
                    sampler=sampler_name,
                    seed=seed,
                )
            )
        return tuple(candidates)

    def sample(
        self,
        count: int,
        *,
        seed: int,
        start_index: int = 0,
        sampler: DesignSampler | None = None,
    ) -> tuple[DesignCandidate, ...]:
        """Sample ``count`` consecutive deterministic sample indices."""

        self._validate_index(count, "count")
        self._validate_index(start_index, "start_index")
        return self.sample_indices(
            range(start_index, start_index + count), seed=seed, sampler=sampler
        )

    @staticmethod
    def _validate_index(value: Any, field_name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field_name} must be a non-negative integer")

    @classmethod
    def _normalize_value(
        cls, variable: DesignVariable, value: Any
    ) -> tuple[int | float | str, float]:
        if isinstance(variable, ContinuousThicknessVariable):
            number = cls._numeric_value(value, variable.name)
            if not variable.lower_nm <= number <= variable.upper_nm:
                raise ValueError(f"{variable.name!r} is outside its inclusive bounds")
            normalized = (number - variable.lower_nm) / (
                variable.upper_nm - variable.lower_nm
            )
            return number, cls._clamp_unit(normalized)

        if isinstance(variable, DiscreteThicknessVariable):
            number = cls._numeric_value(value, variable.name)
            try:
                index = variable.values_nm.index(number)
            except ValueError as exc:
                raise ValueError(
                    f"{variable.name!r} is not an allowed discrete thickness"
                ) from exc
            return variable.values_nm[index], cls._encode_index(
                index, len(variable.values_nm)
            )

        if not isinstance(value, str):
            raise TypeError(f"{variable.name!r} must be a material option name")
        names = [option.name for option in variable.options]
        try:
            index = names.index(value)
        except ValueError as exc:
            raise ValueError(f"{variable.name!r} is not an allowed material choice") from exc
        return value, cls._encode_index(index, len(names))

    @staticmethod
    def _numeric_value(value: Any, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name!r} must be a numeric thickness")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name!r} must be finite")
        return number

    @staticmethod
    def _encode_index(index: int, count: int) -> float:
        return 0.0 if count == 1 else index / (count - 1)

    @staticmethod
    def _decode_index(value: float, count: int) -> int:
        if count == 1:
            return 0
        return min(count - 1, math.floor(value * (count - 1) + 0.5))

    @staticmethod
    def _unit_value(value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("normalized design values must be numeric")
        number = float(value)
        if not math.isfinite(number) or not 0 <= number <= 1:
            raise ValueError("normalized design values must be finite and in [0, 1]")
        return number

    @staticmethod
    def _clamp_unit(value: float) -> float:
        return min(1.0, max(0.0, value))

    @staticmethod
    def _material_option(
        variable: MaterialChoiceVariable, value: int | float | str
    ) -> MaterialOption:
        return next(option for option in variable.options if option.name == value)

    @staticmethod
    def _apply_material_option(layer: LayerSpec, option: MaterialOption) -> LayerSpec:
        return replace(
            layer,
            material=option.material,
            provider=option.provider,
            dataset_id=option.dataset_id,
            constant_n=option.constant_n,
            constant_k=option.constant_k,
        )


__all__ = ["DesignSampler", "DesignSpace", "RandomSampler"]
