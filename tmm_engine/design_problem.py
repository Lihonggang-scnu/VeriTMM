"""Declarative OptimizationProblem: a machine-understandable design space.

This module is a *declaration layer plus compiler*, not an optimizer.  An
agent declares a problem — stack skeleton, band objectives, constraints,
tunable parameters — and :func:`compile_design_problem` translates it into an
ordinary :class:`~tmm_engine.schemas.OptimizationTask` that runs the existing
differentiable optimization chain and its independent verification.  The
existing OptimizationTask behavior is untouched.

Semantics (verifier-first: constraints are boundaries, never penalties):

- Band objectives map to :class:`SpectralTarget` entries.
  ``BandContrast`` expands into a *dual-band composite*: the objective weight
  is split evenly between the two bands — ``R(in-band)`` with ``at_least 1``
  and ``R(contrast band)`` with ``match 0`` (inverted for ``minimize``).
  Minimizing the resulting weighted sum of ``(1-R_A)^2 + R_B^2`` is the
  standard monotone surrogate for maximizing the contrast
  ``mean R(A) - mean R(B)`` on [0, 1] observables, and the even split keeps
  the two bands equally weighted in the surrogate.
- ``StopbandRipple`` maps to the ``worst_case`` aggregation: maximizing the
  worst in-band R (or minimizing the worst R for ``minimize``) is exactly
  in-band ripple suppression.
- Thickness constraints are compiled into the per-layer optimization box
  (``min/max_thickness_nm``), so every point the optimizer can reach satisfies
  them — ``total_thickness_max`` is encoded by capping each layer's upper
  bound at ``value - sum(other lower bounds)`` and is rejected as infeasible
  when no sound box exists.  Structural constraints (layer count, material
  count, forbidden materials) are compile-time validations: violations are
  typed rejections at compilation, never soft penalties.
"""

from __future__ import annotations

from typing import List, Literal, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .schemas import (
    IlluminationSpec,
    LayerSpec,
    MediumSpec,
    OptimizationTask,
    OptimizerSpec,
    RobustnessSpec,
    SimulationTask,
    SpectralGrid,
    SpectralTarget,
    StackSpec,
)

DESIGN_PROBLEM_SCHEMA_VERSION = "veritmm-design-problem-v1"

_DEFAULT_GRID_POINTS = 101
_DEFAULT_BAND_MARGIN_FRACTION = 0.10


class DesignProblemError(ValueError):
    """Typed compile-time rejection of a declarative problem.

    Constraint violations are boundaries, not penalties: a problem that
    violates a structural constraint or admits no sound thickness box is
    rejected here, before any optimization runs.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class ProblemMedium(BaseModel):
    """Incident or exit medium of the declared stack skeleton."""

    model_config = ConfigDict(extra="forbid")

    material: str | None = None
    constant_n: float | None = None
    constant_k: float = 0.0

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "ProblemMedium":
        if bool(self.material) == (self.constant_n is not None):
            raise ValueError("medium requires exactly one of material or constant_n")
        return self


class ProblemLayer(BaseModel):
    """One film of the declared stack skeleton (starting thickness included)."""

    model_config = ConfigDict(extra="forbid")

    material: str | None = None
    thickness_nm: float
    constant_n: float | None = None
    constant_k: float = 0.0
    coherence: Literal["coherent", "incoherent"] = "coherent"
    label: str | None = None

    @model_validator(mode="after")
    def _validate_layer(self) -> "ProblemLayer":
        if bool(self.material) == (self.constant_n is not None):
            raise ValueError("layer requires exactly one of material or constant_n")
        if not self.thickness_nm > 0.0:
            raise ValueError("layer thickness_nm must be positive")
        if self.constant_k < 0.0:
            raise ValueError("constant_k must be non-negative for passive media")
        return self


class ProblemStack(BaseModel):
    """Stack skeleton: materials, layer order, and starting thicknesses."""

    model_config = ConfigDict(extra="forbid")

    layers: List[ProblemLayer] = Field(min_length=1)
    incident: ProblemMedium
    exit: ProblemMedium
    name: str = "declarative_stack"


class ProblemObjective(BaseModel):
    """A band objective over the declared stack."""

    model_config = ConfigDict(extra="forbid")

    metric: Literal[
        "MeanReflectance",
        "MeanTransmittance",
        "MeanAbsorption",
        "BandContrast",
        "StopbandRipple",
        "PolarizationExtinction",
    ]
    band_nm: Tuple[float, float]
    contrast_band_nm: Tuple[float, float] | None = None
    angle_deg: float = 0.0
    pol: Literal["s", "p", "unpolarized"] = "unpolarized"
    direction: Literal["maximize", "minimize"] = "maximize"
    weight: float = 1.0

    @model_validator(mode="after")
    def _validate_objective(self) -> "ProblemObjective":
        lo, hi = self.band_nm
        if not (lo > 0.0 and hi >= lo):
            raise ValueError("band_nm must be positive and ordered")
        if not self.weight > 0.0:
            raise ValueError("weight must be positive")
        if self.metric == "BandContrast":
            if self.contrast_band_nm is None:
                raise ValueError(
                    "BandContrast requires contrast_band_nm (the reference band)"
                )
            lo2, hi2 = self.contrast_band_nm
            if not (lo2 > 0.0 and hi2 >= lo2):
                raise ValueError("contrast_band_nm must be positive and ordered")
        if self.metric == "PolarizationExtinction" and self.pol == "unpolarized":
            raise ValueError(
                "PolarizationExtinction requires s or p polarization (the "
                "orthogonal channel is compiled automatically)"
            )
        return self

    @property
    def observable(self) -> str:
        return {
            "MeanReflectance": "R",
            "MeanTransmittance": "T",
            "MeanAbsorption": "A",
            "BandContrast": "R",
            "StopbandRipple": "R",
            "PolarizationExtinction": "T",
        }[self.metric]


class ProblemConstraint(BaseModel):
    """A structural or thickness constraint.  Boundaries, not penalties."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal[
        "total_thickness_max",
        "layer_thickness_min",
        "layer_thickness_max",
        "layer_count_max",
        "material_count_max",
        "forbidden_materials",
    ]
    value: float | None = None
    forbidden_materials: List[str] = []

    @model_validator(mode="after")
    def _validate_constraint(self) -> "ProblemConstraint":
        if self.kind == "forbidden_materials":
            if not self.forbidden_materials:
                raise ValueError("forbidden_materials must name at least one material")
            return self
        if self.value is None:
            raise ValueError(f"constraint {self.kind!r} requires value")
        if self.kind in ("layer_count_max", "material_count_max"):
            if isinstance(self.value, bool) or self.value != int(self.value) or self.value < 1:
                raise ValueError(f"{self.kind} requires a positive integer value")
        elif self.value <= 0.0:
            raise ValueError(f"constraint {self.kind!r} requires a positive value")
        return self


class ProblemParameter(BaseModel):
    """Which layers are tunable and within which thickness bounds."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["layer_thickness"] = "layer_thickness"
    layer_selector: Union[Literal["alternating", "cavity", "all"], List[int]]
    bounds_nm: Tuple[float, float]

    @model_validator(mode="after")
    def _validate_parameter(self) -> "ProblemParameter":
        lo, hi = self.bounds_nm
        if not (lo > 0.0 and hi >= lo):
            raise ValueError("bounds_nm must be positive and ordered")
        if isinstance(self.layer_selector, list):
            if not self.layer_selector:
                raise ValueError("layer_selector indices must be non-empty")
            if any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in self.layer_selector
            ):
                raise ValueError("layer_selector indices must be non-negative integers")
        return self


class OptimizationProblemModel(BaseModel):
    """Declarative optimization problem (compile with compile_design_problem)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["veritmm-design-problem-v1"] = DESIGN_PROBLEM_SCHEMA_VERSION
    stack: ProblemStack
    objectives: List[ProblemObjective] = Field(min_length=1)
    constraints: List[ProblemConstraint] = []
    parameters: List[ProblemParameter] = []
    solver: dict = Field(default_factory=dict)
    robustness: dict | None = None
    spectrum: Tuple[float, float, int] | None = None
    """(start_nm, stop_nm, points); derived deterministically from the
    objective bands when omitted."""


def _selector_indices(selector: Union[str, List[int]], layer_count: int) -> List[int]:
    if isinstance(selector, list):
        indices = sorted(set(int(item) for item in selector))
        for index in indices:
            if not 0 <= index < layer_count:
                raise DesignProblemError(
                    "unknown_variable",
                    f"layer_selector index {index} is outside the stack",
                )
        return indices
    if selector == "all":
        return list(range(layer_count))
    if selector == "alternating":
        return list(range(0, layer_count, 2))
    if selector == "cavity":
        if layer_count % 2 == 0:
            raise DesignProblemError(
                "no_cavity_layer",
                "selector 'cavity' requires an odd number of layers",
            )
        return [layer_count // 2]
    raise DesignProblemError(
        "unknown_variable", f"unsupported layer_selector {selector!r}"
    )


def _resolve_parameter_bounds(
    problem: OptimizationProblemModel,
) -> Tuple[dict, dict]:
    """Per-layer bounds (selected layers only) and the default bounds for all."""

    layer_count = len(problem.stack.layers)
    defaults = {
        index: (layer.thickness_nm / 2.0, layer.thickness_nm * 2.0)
        for index, layer in enumerate(problem.stack.layers)
    }
    selected: dict = {}
    for parameter in problem.parameters:
        if parameter.kind != "layer_thickness":
            raise DesignProblemError(
                "parameter_unsupported",
                f"unsupported parameter kind {parameter.kind!r}",
            )
        indices = _selector_indices(parameter.layer_selector, layer_count)
        lo, hi = parameter.bounds_nm
        for index in indices:
            existing = selected.get(index, defaults[index])
            selected[index] = (max(existing[0], lo), min(existing[1], hi))
    for index, (lo, hi) in list(selected.items()):
        if hi < lo:
            raise DesignProblemError(
                "constraint_infeasible",
                f"layer {index} has conflicting parameter bounds [{lo}, {hi}]",
            )
    return selected, defaults


def _apply_thickness_constraints(problem: OptimizationProblemModel, selected: dict) -> None:
    """Fold thickness constraints into the per-layer optimization box.

    ``total_thickness_max`` is encoded by proportionally shrinking the free
    span (hi - lo) of every selected layer until ``sum(hi) <= value``: the
    box's worst case (all layers at max) then satisfies the constraint, so
    any point the optimizer can reach is feasible by construction.
    """

    for constraint in problem.constraints:
        if constraint.kind == "layer_thickness_min":
            for index in selected:
                lo, hi = selected[index]
                selected[index] = (max(lo, float(constraint.value)), hi)
        elif constraint.kind == "layer_thickness_max":
            for index in selected:
                lo, hi = selected[index]
                selected[index] = (lo, min(hi, float(constraint.value)))

    total_max_constraints = [
        float(constraint.value)
        for constraint in problem.constraints
        if constraint.kind == "total_thickness_max"
    ]
    if total_max_constraints:
        total_max = min(total_max_constraints)
        lower_sum = sum(bounds[0] for bounds in selected.values())
        upper_sum = sum(bounds[1] for bounds in selected.values())
        if lower_sum > total_max + 1e-12:
            raise DesignProblemError(
                "constraint_infeasible",
                "no sound thickness box exists: sum of lower bounds "
                f"{lower_sum:.6f} nm exceeds total_thickness_max {total_max:.6f} nm",
            )
        if upper_sum > total_max:
            free_span = upper_sum - lower_sum
            shrink = (total_max - lower_sum) / free_span
            for index in selected:
                lo, hi = selected[index]
                selected[index] = (lo, lo + (hi - lo) * shrink)

    for index, (lo, hi) in selected.items():
        if hi < lo - 1e-12:
            raise DesignProblemError(
                "constraint_infeasible",
                f"layer {index} thickness bounds collapsed: [{lo}, {hi}] after constraints",
            )


def _structural_constraints(problem: OptimizationProblemModel) -> None:
    layer_count = len(problem.stack.layers)
    for constraint in problem.constraints:
        if constraint.kind == "layer_count_max":
            if layer_count > int(constraint.value):
                raise DesignProblemError(
                    "constraint_violated",
                    f"stack has {layer_count} layers; layer_count_max is "
                    f"{int(constraint.value)}",
                )
        elif constraint.kind == "material_count_max":
            # Material identity: a named dataset, or a constant-index medium
            # identified by its (n, k) pair — two distinct constant indices
            # are two distinct materials.
            materials = {
                (
                    layer.material.lower()
                    if layer.material
                    else f"constant_n={layer.constant_n!r};k={layer.constant_k!r}"
                )
                for layer in problem.stack.layers
            }
            if len(materials) > int(constraint.value):
                raise DesignProblemError(
                    "constraint_violated",
                    f"stack uses {len(materials)} distinct materials; "
                    f"material_count_max is {int(constraint.value)}",
                )
        elif constraint.kind == "forbidden_materials":
            forbidden = {name.lower() for name in constraint.forbidden_materials}
            offenders = sorted(
                {
                    layer.material
                    for layer in problem.stack.layers
                    if layer.material and layer.material.lower() in forbidden
                }
            )
            if offenders:
                raise DesignProblemError(
                    "constraint_violated",
                    f"stack uses forbidden materials: {offenders} "
                    f"(forbidden: {sorted(forbidden)})",
                )


def _derive_spectrum(problem: OptimizationProblemModel) -> SpectralGrid:
    if problem.spectrum is not None:
        start, stop, points = problem.spectrum
        return SpectralGrid(float(start), float(stop), int(points))
    spans = []
    for objective in problem.objectives:
        spans.append((float(objective.band_nm[0]), float(objective.band_nm[1])))
        if objective.contrast_band_nm is not None:
            spans.append(
                (float(objective.contrast_band_nm[0]), float(objective.contrast_band_nm[1]))
            )
    lo = min(span[0] for span in spans)
    hi = max(span[1] for span in spans)
    margin = (hi - lo) * _DEFAULT_BAND_MARGIN_FRACTION
    return SpectralGrid(
        max(1.0, lo - margin), hi + margin, _DEFAULT_GRID_POINTS
    )


def compile_design_problem(
    problem: OptimizationProblemModel, *, validate_only: bool = False
) -> OptimizationTask:
    """Compile a declarative problem into an existing-chain OptimizationTask."""

    _structural_constraints(problem)
    selected, defaults = _resolve_parameter_bounds(problem)
    _apply_thickness_constraints(problem, selected)

    try:
        optimizer_spec = OptimizerSpec(**problem.solver)
        optimizer_spec.validate()
    except (TypeError, ValueError) as exc:
        raise DesignProblemError("solver_invalid", f"invalid solver spec: {exc}") from exc
    robustness_spec = None
    if problem.robustness is not None:
        try:
            robustness_spec = RobustnessSpec(**problem.robustness)
            robustness_spec.validate()
        except (TypeError, ValueError) as exc:
            raise DesignProblemError(
                "robustness_invalid", f"invalid robustness spec: {exc}"
            ) from exc

    layers: List[LayerSpec] = []
    for index, layer in enumerate(problem.stack.layers):
        if index in selected:
            lo, hi = selected[index]
            start = min(max(layer.thickness_nm, lo), hi)
            layers.append(
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
        else:
            layers.append(
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

    def _medium(medium: ProblemMedium) -> MediumSpec:
        if medium.material:
            return MediumSpec(
                material=medium.material, provider=None, dataset_id=None
            )
        return MediumSpec(
            constant_n=medium.constant_n, constant_k=medium.constant_k
        )

    stack = StackSpec(
        layers=tuple(layers),
        incident=_medium(problem.stack.incident),
        exit=_medium(problem.stack.exit),
        name=problem.stack.name,
    )

    targets: List[SpectralTarget] = []
    for objective in problem.objectives:
        observable = objective.observable
        lo, hi = float(objective.band_nm[0]), float(objective.band_nm[1])
        if objective.metric == "BandContrast":
            # Dual-band composite with an even weight split: the surrogate
            # (1 - R_A)^2 + R_B^2 is the monotone contrast surrogate for
            # maximize; `minimize` swaps the bands' roles.
            c_lo, c_hi = (
                float(objective.contrast_band_nm[0]),
                float(objective.contrast_band_nm[1]),
            )
            half = float(objective.weight) / 2.0
            primary_constraint = (
                "at_least" if objective.direction == "maximize" else "match"
            )
            primary_target = 1.0 if objective.direction == "maximize" else 0.0
            contrast_constraint = (
                "match" if objective.direction == "maximize" else "at_least"
            )
            contrast_target = 0.0 if objective.direction == "maximize" else 1.0
            targets.append(
                SpectralTarget(
                    observable=observable,
                    target=primary_target,
                    wavelength_min_nm=lo,
                    wavelength_max_nm=hi,
                    weight=half,
                    angle_deg=objective.angle_deg,
                    polarization=objective.pol,
                    constraint=primary_constraint,
                    aggregation="mean",
                    name=f"{objective.metric}:in_band",
                )
            )
            targets.append(
                SpectralTarget(
                    observable=observable,
                    target=contrast_target,
                    wavelength_min_nm=c_lo,
                    wavelength_max_nm=c_hi,
                    weight=half,
                    angle_deg=objective.angle_deg,
                    polarization=objective.pol,
                    constraint=contrast_constraint,
                    aggregation="mean",
                    name=f"{objective.metric}:contrast_band",
                )
            )
            continue
        if objective.metric == "PolarizationExtinction":
            # Dual-polarization composite with an even weight split: the
            # wanted polarization is driven up (at_least 1) and the orthogonal
            # one down (match 0); the weighted surrogate maximizes the
            # s/p transmittance extinction ratio within the band.
            half = float(objective.weight) / 2.0
            wanted = "p" if objective.pol == "s" else "s"
            wanted_constraint = (
                "at_least" if objective.direction == "maximize" else "match"
            )
            wanted_target = 1.0 if objective.direction == "maximize" else 0.0
            other_constraint = (
                "match" if objective.direction == "maximize" else "at_least"
            )
            other_target = 0.0 if objective.direction == "maximize" else 1.0
            targets.append(
                SpectralTarget(
                    observable="T",
                    target=wanted_target,
                    wavelength_min_nm=lo,
                    wavelength_max_nm=hi,
                    weight=half,
                    angle_deg=objective.angle_deg,
                    polarization=objective.pol,
                    constraint=wanted_constraint,
                    aggregation="mean",
                    name="PolarizationExtinction:wanted",
                )
            )
            targets.append(
                SpectralTarget(
                    observable="T",
                    target=other_target,
                    wavelength_min_nm=lo,
                    wavelength_max_nm=hi,
                    weight=half,
                    angle_deg=objective.angle_deg,
                    polarization=wanted,
                    constraint=other_constraint,
                    aggregation="mean",
                    name="PolarizationExtinction:orthogonal",
                )
            )
            continue
        constraint = "at_least" if objective.direction == "maximize" else "match"
        target = 1.0 if objective.direction == "maximize" else 0.0
        aggregation = (
            "worst_case" if objective.metric == "StopbandRipple" else "mean"
        )
        targets.append(
            SpectralTarget(
                observable=observable,
                target=target,
                wavelength_min_nm=lo,
                wavelength_max_nm=hi,
                weight=float(objective.weight),
                angle_deg=objective.angle_deg,
                polarization=objective.pol,
                constraint=constraint,
                aggregation=aggregation,
                name=f"{objective.metric}:{objective.band_nm[0]:g}..{objective.band_nm[1]:g}",
            )
        )

    angles = sorted({float(objective.angle_deg) for objective in problem.objectives})
    polarizations = set()
    for objective in problem.objectives:
        polarizations.add(str(objective.pol))
        if objective.metric == "PolarizationExtinction":
            # the compiled pair spans both polarizations in the band
            polarizations.add("p" if objective.pol == "s" else "s")
    polarizations = sorted(polarizations)
    simulation = SimulationTask(
        stack=stack,
        spectrum=_derive_spectrum(problem),
        illumination=IlluminationSpec(tuple(angles), tuple(polarizations)),
        requested_outputs=("R", "T", "A"),
    )
    task = OptimizationTask(
        simulation=simulation,
        targets=tuple(targets),
        optimizer=optimizer_spec,
        robustness=robustness_spec,
    )
    task.validate()
    return task


def problem_summary(problem: OptimizationProblemModel, task: OptimizationTask) -> dict:
    """Bounded first-read summary of a compiled problem (for the CLI)."""

    return {
        "schema_version": DESIGN_PROBLEM_SCHEMA_VERSION,
        "stack": {
            "layer_count": len(problem.stack.layers),
            "name": problem.stack.name,
        },
        "objective_count": len(problem.objectives),
        "constraint_count": len(problem.constraints),
        "parameter_count": len(problem.parameters),
        "compiled": {
            "targets": [
                {
                    "name": target.name,
                    "observable": target.observable,
                    "target": target.target,
                    "constraint": target.constraint,
                    "aggregation": target.aggregation,
                    "weight": target.weight,
                    "band_nm": [target.wavelength_min_nm, target.wavelength_max_nm],
                }
                for target in task.targets
            ],
            "optimizable_layers": [
                index
                for index, layer in enumerate(task.simulation.stack.layers)
                if layer.optimizable
            ],
            "spectrum_nm": [
                task.simulation.spectrum.wavelengths_nm()[0],
                task.simulation.spectrum.wavelengths_nm()[-1],
            ],
        },
    }


__all__ = [
    "DESIGN_PROBLEM_SCHEMA_VERSION",
    "DesignProblemError",
    "OptimizationProblemModel",
    "ProblemConstraint",
    "ProblemLayer",
    "ProblemMedium",
    "ProblemObjective",
    "ProblemParameter",
    "ProblemStack",
    "compile_design_problem",
    "problem_summary",
]
