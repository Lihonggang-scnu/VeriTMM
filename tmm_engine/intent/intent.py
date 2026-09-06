"""ScientificIntentSpec: the semantic IR between intent and TMM tasks.

The intent layer is the controlled semantic seam between natural-language or
upstream-agent requests and concrete TMM tasks.  ``compile_intent`` turns a
validated :class:`IntentSpec` plus a declared stack into a legal
SimulationTask (measurement intent) or OptimizationTask (design intent) using
the existing constructors and validators — never bypassing them — and issues a
:meth:`CompilationEquivalenceCertificate` recording every mapping, unit
conversion, inserted default, and ambiguity.

Fail-closed rules:
- unknown quantities / reducers / relations / units are typed rejections
  (``intent_rejected``), never guesses;
- a filled-in default (for example an unspecified reducer) is inserted AND
  reported: every default appears in both ``defaults_inserted`` and
  ``ambiguities``;
- hard constraints that the task language cannot express are downgraded only
  with an explicit ambiguity entry; structural hard constraints are enforced
  at compile time through the design-problem compiler.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

from ..schemas import (
    IlluminationSpec,
    LayerSpec,
    OptimizationTask,
    OptimizerSpec,
    SimulationTask,
    SpectralGrid,
    SpectralTarget,
    StackSpec,
)

INTENT_SCHEMA_VERSION = "veritmm-intent-spec-v1"
EQUIVALENCE_SCHEMA_VERSION = "compilation-equivalence.v1"

_UNIT_FACTORS_TO_NM = {
    "nm": 1.0,
    "um": 1000.0,
    "µm": 1000.0,  # micro sign U+00B5
    "μm": 1000.0,  # greek small mu U+03BC
}


class IntentCompileError(ValueError):
    """Typed rejection of an intent compilation."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class IntentQuantity(BaseModel):
    """A length value with an explicit unit (nm/um/µm/μm)."""

    model_config = ConfigDict(extra="forbid")

    value: float
    unit: Literal["nm", "um", "µm", "μm"] = "nm"

    def to_nm(self) -> float:
        try:
            factor = _UNIT_FACTORS_TO_NM[self.unit]
        except KeyError:
            raise IntentCompileError(
                "unit_unsupported", f"unsupported length unit {self.unit!r}"
            )
        return self.value * factor


class IntentDomain(BaseModel):
    """Observable domain: wavelength window, incidence, polarization."""

    model_config = ConfigDict(extra="forbid")

    wavelength_min: Optional[IntentQuantity] = None
    wavelength_max: Optional[IntentQuantity] = None
    angle_deg: float = 0.0
    polarization: Literal["s", "p", "unpolarized"] = "unpolarized"


class IntentObservable(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quantity: str
    domain: IntentDomain = Field(default_factory=IntentDomain)
    reducer: Optional[Literal["mean", "min", "max", "band_worst_case"]] = None
    relation: Optional[Literal[">=", "<=", "=="]] = None
    target: Optional[float] = None
    constraint_role: Literal["hard", "soft"] = "soft"
    weight: float = 1.0


class IntentDesignVariable(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["layer_thickness"] = "layer_thickness"
    layer_selector: Union[str, List[int]]
    bounds_min: IntentQuantity
    bounds_max: IntentQuantity


class IntentHardConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal[
        "layer_count_max",
        "total_thickness_max",
        "forbidden_materials",
        "substrate",
    ]
    value: float | None = None
    materials: List[str] = []
    substrate: Optional[str] = None


class IntentVerificationRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    energy: Literal["independent", "closure"] = "independent"
    cross_solver: bool = True
    reciprocity: bool = False


class IntentSpec(BaseModel):
    """Semantic IR: what is being asked, independent of any solver."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["veritmm-intent-spec-v1"] = INTENT_SCHEMA_VERSION
    intent_id: str
    source_clause: str = ""
    observables: List[IntentObservable] = Field(min_length=1)
    design_variables: List[IntentDesignVariable] = []
    hard_constraints: List[IntentHardConstraint] = []
    uncertainty: Dict[str, Any] = {}
    verification_requirement: IntentVerificationRequirement = Field(
        default_factory=IntentVerificationRequirement
    )


class CompilationEquivalenceCertificate(BaseModel):
    """Machine-readable record of one compilation: mappings, conversions,
    defaults, ambiguities, and the semantic verdict."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["compilation-equivalence.v1"] = (
        EQUIVALENCE_SCHEMA_VERSION
    )
    intent_id: str
    source_clause: str
    source_clause_hash: str
    intent_fields: Dict[str, Any]
    compiled_task_fields: List[Dict[str, str]]
    unit_conversions: List[Dict[str, Any]]
    defaults_inserted: List[Dict[str, str]]
    ambiguities: List[Dict[str, str]]
    verification_requirement: Dict[str, Any]
    notes: List[str] = []
    semantic_status: Literal["equivalent", "ambiguous", "rejected"] = "equivalent"
    rejection: Dict[str, str] | None = None


class CompiledTask(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_kind: Literal["simulation", "optimization"]
    simulation: Optional[SimulationTask] = None
    optimization: Optional[OptimizationTask] = None
    targets: List[SpectralTarget] = []
    """The intent-level SpectralTargets, inspectable regardless of task kind."""
    certificate: CompilationEquivalenceCertificate


class _Compilation:
    """Accumulates mappings/conversions/defaults/ambiguities during compile."""

    def __init__(self, spec: IntentSpec) -> None:
        self.spec = spec
        self.notes: List[str] = []
        self.compiled_task_fields: List[Dict[str, str]] = []
        self.unit_conversions: List[Dict[str, Any]] = []
        self.defaults_inserted: List[Dict[str, str]] = []
        self.ambiguities: List[Dict[str, str]] = []

    def record_mapping(self, intent_path: str, task_path: str, value: Any) -> None:
        self.compiled_task_fields.append(
            {
                "intent_path": intent_path,
                "task_path": task_path,
                "value": str(value),
            }
        )

    def record_conversion(self, intent_path: str, value: float, unit: str) -> float:
        nm = value * _UNIT_FACTORS_TO_NM[unit]
        if unit != "nm":
            self.unit_conversions.append(
                {
                    "intent_path": intent_path,
                    "from_unit": unit,
                    "to_unit": "nm",
                    "from_value": value,
                    "to_value": nm,
                    "factor": _UNIT_FACTORS_TO_NM[unit],
                }
            )
        return nm

    def insert_default(self, intent_path: str, field_name: str, value: Any, reason: str) -> Any:
        self.defaults_inserted.append(
            {"intent_path": intent_path, "field": field_name, "value": str(value)}
        )
        self.ambiguities.append(
            {
                "intent_path": intent_path,
                "field": field_name,
                "issue": reason,
                "resolution": f"default {value!r} inserted",
            }
        )
        return value

    def ambiguity(self, intent_path: str, field_name: str, issue: str, resolution: str) -> None:
        self.ambiguities.append(
            {"intent_path": intent_path, "field": field_name, "issue": issue, "resolution": resolution}
        )


def _domain_bounds(domain: IntentDomain, index: int, compilation: _Compilation) -> Tuple[Optional[float], Optional[float]]:
    lo = hi = None
    if domain.wavelength_min is not None:
        lo = compilation.record_conversion(
            f"observables[{index}].domain.wavelength_min",
            domain.wavelength_min.value,
            domain.wavelength_min.unit,
        )
    if domain.wavelength_max is not None:
        hi = compilation.record_conversion(
            f"observables[{index}].domain.wavelength_max",
            domain.wavelength_max.value,
            domain.wavelength_max.unit,
        )
    if domain.wavelength_min is not None and domain.wavelength_max is None:
        # an open-ended band closes into a narrow 2% window above the edge
        hi = compilation.insert_default(
            f"observables[{index}].domain.wavelength_max", "wavelength_max", lo * 1.02, "open-ended band closed 2% above the min wavelength"
        )
    if domain.wavelength_max is not None and domain.wavelength_min is None:
        lo = compilation.insert_default(
            f"observables[{index}].domain.wavelength_min", "wavelength_min", hi / 1.02, "open-ended band closed 2% below the max wavelength"
        )
    return lo, hi


def _reduce_reducer(
    compilation: _Compilation, index: int, observable: IntentObservable
) -> Tuple[str, str]:
    reducer = observable.reducer
    relation = observable.relation
    if reducer is None:
        reducer = compilation.insert_default(
            f"observables[{index}].reducer", "reducer", "mean", "reducer unspecified; band mean inserted"
        )
    if relation is None:
        relation = compilation.insert_default(
            f"observables[{index}].relation", "relation", "==", "relation unspecified; equality inserted"
        )
    if reducer in ("min", "max"):
        if (reducer == "min" and relation == ">=") or (reducer == "max" and relation == "<="):
            return "worst_case", reducer
        compilation.ambiguity(
            f"observables[{index}].reducer",
            "reducer",
            f"reducer {reducer!r} is not directly expressible as a target aggregation",
            "downgraded to worst_case aggregation",
        )
        return "worst_case", reducer
    return ("mean" if reducer == "mean" else "worst_case"), reducer


def _observable_target(
    compilation: _Compilation, index: int, observable: IntentObservable
) -> Tuple[float, str]:
    relation = observable.relation or "=="
    target = observable.target
    if target is None:
        if relation == ">=":
            target = compilation.insert_default(
                f"observables[{index}].target", "target", 1.0, "at-least relation without target saturates at 1.0"
            )
        elif relation == "<=":
            target = compilation.insert_default(
                f"observables[{index}].target", "target", 0.0, "at-most relation without target bottoms at 0.0"
            )
        else:
            target = compilation.insert_default(
                f"observables[{index}].target", "target", 0.0, "equality without target"
            )
    if relation == ">=":
        constraint = "at_least"
    elif relation == "<=":
        constraint = "at_most"
    else:
        constraint = "match"
    if observable.constraint_role == "hard":
        compilation.ambiguity(
            f"observables[{index}].constraint_role",
            "constraint_role",
            "hard observable targets are enforced as weighted loss terms by the task language",
            "compiled as a soft target; the hard role is recorded here",
        )
    return float(target), constraint


def compile_intent(
    spec: IntentSpec, stack: StackSpec, spectrum: Optional[SpectralGrid] = None
) -> CompiledTask:
    """Compile an intent plus a declared stack into a task + certificate."""

    compilation = _Compilation(spec)
    try:
        return _compile(spec, stack, spectrum, compilation)
    except IntentCompileError as exc:
        certificate = CompilationEquivalenceCertificate(
            intent_id=spec.intent_id,
            source_clause=spec.source_clause,
            source_clause_hash=stable_hash(spec.source_clause),
            intent_fields=spec.model_dump(mode="json"),
            compiled_task_fields=compilation.compiled_task_fields,
            unit_conversions=compilation.unit_conversions,
            defaults_inserted=compilation.defaults_inserted,
            ambiguities=compilation.ambiguities,
            verification_requirement=spec.verification_requirement.model_dump(),
            notes=list(compilation.notes),
            semantic_status="rejected",
            rejection={"code": exc.code, "message": exc.message[:400]},
        )
        return CompiledTask(
            task_kind="simulation",
            simulation=None,
            optimization=None,
            targets=[],
            certificate=certificate,
        )
    except Exception as exc:
        certificate = CompilationEquivalenceCertificate(
            intent_id=spec.intent_id,
            source_clause=spec.source_clause,
            source_clause_hash=stable_hash(spec.source_clause),
            intent_fields=spec.model_dump(mode="json"),
            compiled_task_fields=compilation.compiled_task_fields,
            unit_conversions=compilation.unit_conversions,
            defaults_inserted=compilation.defaults_inserted,
            ambiguities=compilation.ambiguities,
            verification_requirement=spec.verification_requirement.model_dump(),
            notes=list(compilation.notes),
            semantic_status="rejected",
            rejection={"code": "intent_rejected", "message": str(exc)[:400]},
        )
        return CompiledTask(
            task_kind="simulation",
            simulation=None,
            optimization=None,
            targets=[],
            certificate=certificate,
        )


def stable_hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _check_hard_constraints(
    spec: IntentSpec, stack: StackSpec, compilation: _Compilation
) -> None:
    """Structural hard constraints are compile-time boundaries on the
    declared stack (both measurement and design intents)."""

    for constraint in spec.hard_constraints:
        if constraint.kind == "layer_count_max":
            if len(stack.layers) > int(constraint.value):
                raise IntentCompileError(
                    "constraint_violated",
                    "layer count %d exceeds layer_count_max %s"
                    % (len(stack.layers), constraint.value),
                )
            compilation.record_mapping(
                "hard_constraints.layer_count_max", "stack.layers", "<=%s" % constraint.value
            )
        elif constraint.kind == "total_thickness_max":
            total = sum(layer.thickness_nm for layer in stack.layers)
            if total > float(constraint.value) + 1e-9:
                raise IntentCompileError(
                    "constraint_violated",
                    "total thickness %.6f nm exceeds total_thickness_max %s"
                    % (total, constraint.value),
                )
            compilation.record_mapping(
                "hard_constraints.total_thickness_max", "stack.layers", "<=%s" % constraint.value
            )
        elif constraint.kind == "forbidden_materials":
            forbidden = {name.lower() for name in constraint.materials}
            offenders = sorted(
                {
                    layer.material.lower()
                    for layer in stack.layers
                    if layer.material and layer.material.lower() in forbidden
                }
            )
            if offenders:
                raise IntentCompileError(
                    "constraint_violated",
                    "stack uses forbidden materials %s" % offenders,
                )
            compilation.record_mapping(
                "hard_constraints.forbidden_materials", "stack.layers", "not in %s" % sorted(forbidden)
            )
        elif constraint.kind == "substrate":
            wanted = (constraint.substrate or "").strip()
            exit_medium = stack.exit
            if wanted.lower().startswith("constant:"):
                wanted_value = float(wanted.split(":", 1)[1])
                actual = exit_medium.constant_n
                matched = actual is not None and abs(actual - wanted_value) <= 1e-9
            else:
                wanted_name = wanted.lower()
                actual = exit_medium.material
                matched = actual is not None and actual.lower() == wanted_name
            if not wanted or not matched:
                raise IntentCompileError(
                    "constraint_violated",
                    "substrate constraint requires exit medium %r, got material=%r constant_n=%r"
                    % (constraint.substrate, exit_medium.material, exit_medium.constant_n),
                )
            compilation.record_mapping(
                "hard_constraints.substrate", "stack.exit", constraint.substrate
            )


def _compile(
    spec: IntentSpec,
    stack: StackSpec,
    spectrum: Optional[SpectralGrid],
    compilation: _Compilation,
) -> CompiledTask:
    _check_hard_constraints(spec, stack, compilation)

    targets: List[SpectralTarget] = []
    _SUPPORTED_QUANTITIES = ("R", "T", "A", "layer_absorption")
    for index, observable in enumerate(spec.observables):
        if observable.quantity not in _SUPPORTED_QUANTITIES:
            raise IntentCompileError(
                "quantity_unsupported",
                f"unknown quantity {observable.quantity!r}; supported: "
                f"{list(_SUPPORTED_QUANTITIES)}",
            )
        if observable.quantity == "layer_absorption":
            raise IntentCompileError(
                "quantity_unsupported",
                "layer_absorption is not yet a compilable objective quantity",
            )
        if observable.quantity == "A":
            compilation.notes.append(
                f"observables[{index}]: mean_A follows the closure convention "
                "A = 1 - R - T; the independent absorption ledger is not "
                "differentiable in this release"
            )
        lo, hi = _domain_bounds(observable.domain, index, compilation)
        aggregation, _ = _reduce_reducer(compilation, index, observable)
        target_value, constraint = _observable_target(compilation, index, observable)
        intent_path = f"observables[{index}]"
        target = SpectralTarget(
            observable=observable.quantity,
            target=target_value,
            wavelength_min_nm=float(lo),
            wavelength_max_nm=float(hi),
            weight=float(observable.weight),
            angle_deg=observable.domain.angle_deg,
            polarization=observable.domain.polarization,
            constraint=constraint,
            aggregation=aggregation,
            name=f"intent[{index}]:{observable.quantity}:{lo:g}..{hi:g}",
        )
        target.validate()
        targets.append(target)
        compilation.record_mapping(
            intent_path,
            f"targets[{len(targets) - 1}]",
            f"{observable.quantity} {constraint} {target_value} over [{lo:g}, {hi:g}] nm",
        )

    bands = [(target.wavelength_min_nm, target.wavelength_max_nm) for target in targets]
    grid = spectrum or _derive_grid(bands)
    angles = sorted({target.angle_deg for target in targets})
    polarizations = sorted({str(target.polarization) for target in targets})
    simulation = SimulationTask(
        stack=stack,
        spectrum=grid,
        illumination=IlluminationSpec(tuple(angles), tuple(polarizations)),
        requested_outputs=("R", "T", "A"),
    )

    if not spec.design_variables:
        simulation.validate()
        certificate = _certificate(spec, compilation, "simulation")
        return CompiledTask(
            task_kind="simulation",
            simulation=simulation,
            targets=targets,
            certificate=certificate,
        )

    # design intent: parameters select optimizable layers; hard structural
    # constraints are enforced through the existing LayerSpec/Task validators
    optimizable = set()
    bounds_by_layer: Dict[int, Tuple[float, float]] = {}
    compiled_layers: List[LayerSpec] = []
    for index, layer in enumerate(stack.layers):
        variable = _variable_for_layer(spec, index, compilation)
        if variable is None:
            compiled_layers.append(
                LayerSpec(
                    layer.material,
                    layer.thickness_nm,
                    coherence=layer.coherence,
                    constant_n=layer.constant_n,
                    constant_k=layer.constant_k,
                    optimizable=False,
                    label=layer.label,
                )
            )
            continue
        lo = compilation.record_conversion(
            f"design_variables[{index}].bounds_min", variable.bounds_min.value, variable.bounds_min.unit
        )
        hi = compilation.record_conversion(
            f"design_variables[{index}].bounds_max", variable.bounds_max.value, variable.bounds_max.unit
        )
        start = min(max(layer.thickness_nm, lo), hi)
        compiled_layers.append(
            LayerSpec(
                layer.material,
                start,
                coherence=layer.coherence,
                constant_n=layer.constant_n,
                constant_k=layer.constant_k,
                optimizable=True,
                min_thickness_nm=lo,
                max_thickness_nm=hi,
                label=layer.label,
            )
        )
        optimizable.add(index)
        bounds_by_layer[index] = (lo, hi)
    if not optimizable:
        raise IntentCompileError(
            "no_compilable_variables",
            "design variables selected no layers within the declared stack",
        )
    compiled_stack = StackSpec(
        layers=tuple(compiled_layers), incident=stack.incident, exit=stack.exit
    )

    task = OptimizationTask(
        simulation=replace(simulation, stack=compiled_stack),
        targets=tuple(targets),
        optimizer=OptimizerSpec(),
    )
    task.validate()
    certificate = _certificate(spec, compilation, "optimization")
    return CompiledTask(
        task_kind="optimization",
        optimization=task,
        targets=targets,
        certificate=certificate,
    )


def _variable_for_layer(spec: IntentSpec, index: int, compilation: _Compilation) -> Optional[IntentDesignVariable]:
    for position, variable in enumerate(spec.design_variables):
        selector = variable.layer_selector
        if isinstance(selector, str):
            raise IntentCompileError(
                "selector_unsupported",
                "symbolic layer selectors must be expanded before compilation",
            )
        if index in selector:
            return variable
    return None


def _derive_grid(bands: Sequence) -> SpectralGrid:
    lo = min(band[0] for band in bands)
    hi = max(band[1] for band in bands)
    margin = max((hi - lo) * 0.10, lo * 0.02, 1.0)
    return SpectralGrid(max(1.0, lo - margin), hi + margin, 101)


def _certificate(
    spec: IntentSpec, compilation: _Compilation, task_kind: str
) -> CompilationEquivalenceCertificate:
    status = "ambiguous" if compilation.ambiguities else "equivalent"
    return CompilationEquivalenceCertificate(
        intent_id=spec.intent_id,
        source_clause=spec.source_clause,
        source_clause_hash=stable_hash(spec.source_clause),
        intent_fields=spec.model_dump(mode="json"),
        compiled_task_fields=compilation.compiled_task_fields,
        unit_conversions=compilation.unit_conversions,
        defaults_inserted=compilation.defaults_inserted,
        ambiguities=compilation.ambiguities,
        verification_requirement=spec.verification_requirement.model_dump(),
        notes=list(compilation.notes),
        semantic_status=status,
    )


__all__ = [
    "CompiledTask",
    "CompilationEquivalenceCertificate",
    "IntentCompileError",
    "IntentDesignVariable",
    "IntentDomain",
    "IntentHardConstraint",
    "IntentObservable",
    "IntentQuantity",
    "IntentSpec",
    "IntentVerificationRequirement",
    "compile_intent",
]
