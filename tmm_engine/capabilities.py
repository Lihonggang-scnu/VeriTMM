"""Machine-readable capability and failure contracts for optical solvers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from .schemas import SimulationTask


class FailureCode(str, Enum):
    INVALID_TASK = "invalid_task"
    UNSUPPORTED_GEOMETRY = "unsupported_geometry"
    UNSUPPORTED_MATERIAL_MODEL = "unsupported_material_model"
    UNSUPPORTED_EXCITATION = "unsupported_excitation"
    TIME_DOMAIN_REQUIRED = "time_domain_required"
    UNSUPPORTED_OUTPUT_COMBINATION = "unsupported_output_combination"
    REQUESTED_OUTPUT_MISSING = "requested_output_missing"
    MATERIAL_NOT_FOUND = "material_not_found"
    MATERIAL_AMBIGUITY = "material_ambiguity"
    MATERIAL_RANGE_ERROR = "material_range_error"
    OPTIONAL_DEPENDENCY_MISSING = "optional_dependency_missing"
    NUMERICAL_NONFINITE = "numerical_nonfinite"
    PASSIVITY_VIOLATION = "passivity_violation"
    ENERGY_CONSERVATION_FAILURE = "energy_conservation_failure"
    SPECTRAL_CONVERGENCE_FAILURE = "spectral_convergence_failure"
    SOLVER_DISAGREEMENT = "solver_disagreement"
    OPTIMIZER_FAILURE = "optimizer_failure"
    BUDGET_EXHAUSTED = "budget_exhausted"
    PROVENANCE_CONFLICT = "provenance_conflict"
    INSUFFICIENT_VALID_SAMPLES = "insufficient_valid_samples"


ActionSafety = Literal[
    "safe", "requires_scientific_judgment", "requires_user_input"
]


@dataclass(frozen=True)
class FailureAction:
    """A machine-readable next step attached to a typed failure.

    A patch is present only when changing the task is mechanically safe.  The
    engine deliberately does not encode scientific choices (for example a new
    material or target wavelength) as automatic JSON patches.
    """

    action_id: str
    action_type: str
    description: str
    safety: ActionSafety
    patch: tuple[Dict[str, Any], ...] = ()
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["patch"] = list(self.patch)
        return payload


@dataclass(frozen=True)
class FailureRecord:
    code: FailureCode
    message: str
    recoverable: bool
    suggested_solver_family: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    severity: Literal["warning", "error", "fatal"] = "error"
    requires_user_choice: bool = False
    actions: tuple[FailureAction, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["code"] = self.code.value
        payload["actions"] = [item.to_dict() for item in self.actions]
        return payload


def enrich_failure_actions(failure: FailureRecord) -> FailureRecord:
    """Attach conservative navigation advice without making scientific choices."""

    if failure.actions or not failure.recoverable:
        return failure

    code = failure.code
    action: FailureAction | None = None
    requires_user_choice = failure.requires_user_choice
    if code in {
        FailureCode.UNSUPPORTED_GEOMETRY,
        FailureCode.UNSUPPORTED_MATERIAL_MODEL,
        FailureCode.UNSUPPORTED_EXCITATION,
        FailureCode.TIME_DOMAIN_REQUIRED,
    }:
        action = FailureAction(
            action_id="route_to_supported_solver_family",
            action_type="solver_family_guidance",
            safety="requires_scientific_judgment",
            description=(
                "Keep this task unchanged and route it to the suggested solver family; "
                "VeriTMM will not execute that handoff."
            ),
            context={"suggested_solver_family": failure.suggested_solver_family},
        )
    elif code == FailureCode.UNSUPPORTED_OUTPUT_COMBINATION:
        action = FailureAction(
            action_id="split_or_revise_requested_outputs",
            action_type="task_revision_guidance",
            safety="requires_scientific_judgment",
            description=(
                "Inspect the output semantics and split incompatible observables into "
                "separate scientifically valid tasks if appropriate."
            ),
        )
    elif code == FailureCode.REQUESTED_OUTPUT_MISSING:
        action = FailureAction(
            action_id="inspect_backend_output_contract",
            action_type="diagnostic_review",
            safety="requires_scientific_judgment",
            description=(
                "Do not use the incomplete result. Inspect backend routing and replay the "
                "unchanged task only after the requested-output contract is satisfied."
            ),
            context=dict(failure.context),
        )
    elif code == FailureCode.MATERIAL_NOT_FOUND:
        requires_user_choice = True
        action = FailureAction(
            action_id="select_available_material_dataset",
            action_type="material_selection_required",
            safety="requires_user_input",
            description=(
                "Choose an available material and dataset explicitly; the engine will "
                "not substitute a different material automatically."
            ),
            context=dict(failure.context),
        )
    elif code == FailureCode.MATERIAL_AMBIGUITY:
        action = FailureAction(
            action_id="select_explicit_dataset_id",
            action_type="material_selection_required",
            safety="requires_scientific_judgment",
            description=(
                "Select a provider and dataset_id from the reported candidates before rerunning."
            ),
            context=dict(failure.context),
        )
    elif code == FailureCode.MATERIAL_RANGE_ERROR:
        action = FailureAction(
            action_id="resolve_material_wavelength_coverage",
            action_type="task_revision_guidance",
            safety="requires_scientific_judgment",
            description=(
                "Choose a dataset covering the requested wavelengths or explicitly revise "
                "the scientific spectrum; extrapolation is never enabled automatically."
            ),
            context=dict(failure.context),
        )
    elif code == FailureCode.OPTIONAL_DEPENDENCY_MISSING:
        requires_user_choice = True
        action = FailureAction(
            action_id="install_required_optional_dependency",
            action_type="environment_change",
            safety="requires_user_input",
            description="Install the missing optional dependency, then replay the unchanged task.",
        )
    elif code == FailureCode.SPECTRAL_CONVERGENCE_FAILURE:
        action = FailureAction(
            action_id="increase_convergence_budget",
            action_type="execution_setting_change",
            safety="safe",
            description=(
                "Increase spectral refinement budget or use a denser grid without changing "
                "the requested physical target."
            ),
        )
    elif code in {
        FailureCode.SOLVER_DISAGREEMENT,
        FailureCode.OPTIMIZER_FAILURE,
        FailureCode.BUDGET_EXHAUSTED,
    }:
        action = FailureAction(
            action_id="inspect_numerics_before_retry",
            action_type="diagnostic_review",
            safety="requires_scientific_judgment",
            description=(
                "Inspect the numerical diagnostics and execution budget before deciding "
                "whether and how to retry."
            ),
            context=dict(failure.context),
        )
    elif code == FailureCode.PROVENANCE_CONFLICT:
        action = FailureAction(
            action_id="use_fresh_run_identity",
            action_type="provenance_safe_retry",
            safety="safe",
            description=(
                "Keep the existing ledger row and artifacts unchanged. Retry only with "
                "a fresh run_id and an empty destination directory."
            ),
            context=dict(failure.context),
        )
    elif code == FailureCode.INSUFFICIENT_VALID_SAMPLES:
        action = FailureAction(
            action_id="inspect_sample_failure_taxonomy",
            action_type="diagnostic_review",
            safety="requires_scientific_judgment",
            description=(
                "Inspect the sample failure taxonomy and numerical/material diagnostics "
                "before deciding whether a scientifically unchanged retry is justified."
            ),
            context=dict(failure.context),
        )

    if action is None:
        return failure
    return replace(
        failure,
        requires_user_choice=requires_user_choice,
        actions=(action,),
    )


class PhysicsEngineError(RuntimeError):
    def __init__(self, failure: FailureRecord) -> None:
        super().__init__(failure.message)
        self.failure = failure


def failure_from_exception(exc: Exception) -> FailureRecord:
    """Translate known execution exceptions into the stable failure contract."""

    if isinstance(exc, PhysicsEngineError):
        return enrich_failure_actions(exc.failure)

    # Imported lazily so the capability vocabulary remains usable without
    # constructing a material registry.
    from .experiment_store import RunLedgerConflictError
    from .material_registry import (
        MaterialAmbiguityError,
        MaterialNotFoundError,
        MaterialRangeError,
    )

    if isinstance(exc, RunLedgerConflictError):
        failure = FailureRecord(
            FailureCode.PROVENANCE_CONFLICT,
            str(exc),
            True,
            context={"run_id": exc.run_id},
        )
    elif isinstance(exc, MaterialRangeError):
        failure = FailureRecord(
            FailureCode.MATERIAL_RANGE_ERROR,
            str(exc),
            True,
            context={
                "material": exc.material,
                "requested_range": exc.requested_range,
                "available_range": exc.available_range,
            },
        )
    elif isinstance(exc, MaterialAmbiguityError):
        failure = FailureRecord(
            FailureCode.MATERIAL_AMBIGUITY,
            str(exc),
            True,
            context={
                "material": exc.material,
                "candidates": [
                    {
                        "provider": item.provider,
                        "dataset_id": item.dataset_id,
                        "book": item.book,
                        "page": item.page,
                        "range_um": item.range,
                    }
                    for item in exc.candidates
                ],
            },
        )
    elif isinstance(exc, MaterialNotFoundError):
        failure = FailureRecord(
            FailureCode.MATERIAL_NOT_FOUND,
            str(exc),
            True,
            context={
                "material": exc.material,
                "provider": exc.provider,
                "dataset_id": exc.dataset_id,
            },
        )
    elif isinstance(exc, ImportError):
        failure = FailureRecord(
            FailureCode.OPTIONAL_DEPENDENCY_MISSING,
            str(exc),
            True,
        )
    else:
        failure = FailureRecord(
            FailureCode.INVALID_TASK,
            f"{type(exc).__name__}: {exc}",
            False,
        )
    return enrich_failure_actions(failure)


@dataclass(frozen=True)
class CapabilityAssessment:
    engine_id: str
    supported: bool
    resolved_solver: Optional[str]
    failures: tuple[FailureRecord, ...] = ()
    warnings: tuple[str, ...] = ()
    capability_version: str = "tmm-isotropic-planar-v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "capability_version": self.capability_version,
            "supported": self.supported,
            "resolved_solver": self.resolved_solver,
            "failures": [enrich_failure_actions(item).to_dict() for item in self.failures],
            "warnings": list(self.warnings),
        }


def assess_tmm_capability(task: SimulationTask) -> CapabilityAssessment:
    """Decide whether the current scalar TMM environment may execute a task."""

    failures: List[FailureRecord] = []
    warnings: List[str] = []
    try:
        task.validate()
    except Exception as exc:
        failures.append(FailureRecord(FailureCode.INVALID_TASK, str(exc), False))
        return CapabilityAssessment("veritmm", False, None, tuple(failures))

    profile = task.physics
    if profile.geometry_class != "layered_planar":
        recommended = "rcwa" if profile.geometry_class == "lateral_periodic" else "fdtd_or_fem"
        failures.append(
            FailureRecord(
                FailureCode.UNSUPPORTED_GEOMETRY,
                "Scalar TMM supports variation only along the layer normal; lateral or arbitrary geometry requires another solver.",
                True,
                recommended,
                {"geometry_class": profile.geometry_class},
            )
        )
    if profile.material_class != "isotropic":
        recommended = "berreman_4x4" if profile.material_class == "anisotropic" else "fdtd_or_fem"
        failures.append(
            FailureRecord(
                FailureCode.UNSUPPORTED_MATERIAL_MODEL,
                "The current TMM engine accepts passive isotropic scalar optical constants only.",
                True,
                recommended,
                {"material_class": profile.material_class},
            )
        )
    if profile.excitation_class != "plane_wave":
        failures.append(
            FailureRecord(
                FailureCode.UNSUPPORTED_EXCITATION,
                "The current TMM engine supports plane-wave excitation only.",
                True,
                "fourier_optics_or_full_wave",
                {"excitation_class": profile.excitation_class},
            )
        )
    if profile.time_domain_required:
        failures.append(
            FailureRecord(
                FailureCode.TIME_DOMAIN_REQUIRED,
                "A time-domain response was requested; frequency-domain TMM is not sufficient.",
                True,
                "fdtd",
            )
        )

    outputs = set(task.requested_outputs)
    mixed = task.stack.has_incoherent_layers
    if mixed and "ellipsometry" in outputs:
        failures.append(
            FailureRecord(
                FailureCode.UNSUPPORTED_OUTPUT_COMBINATION,
                "Ellipsometric phase is not available for a mixed coherent/incoherent stack.",
                False,
            )
        )
    if mixed and "amplitudes" in outputs:
        failures.append(
            FailureRecord(
                FailureCode.UNSUPPORTED_OUTPUT_COMBINATION,
                "A single coherent complex amplitude is undefined for a mixed coherent/incoherent stack.",
                False,
            )
        )
    if mixed and "phase_dispersion" in outputs:
        failures.append(
            FailureRecord(
                FailureCode.UNSUPPORTED_OUTPUT_COMBINATION,
                "A coherent phase/group-delay response is undefined for a mixed coherent/incoherent stack.",
                False,
            )
        )
    if task.solver == "byrnes" and "phase_dispersion" in outputs:
        failures.append(
            FailureRecord(
                FailureCode.UNSUPPORTED_OUTPUT_COMBINATION,
                "Use the internal coherent S-matrix backend for phase-dispersion output; Byrnes remains the independent R/T/A oracle.",
                True,
            )
        )
    if task.solver == "byrnes" and "amplitudes" in outputs:
        failures.append(
            FailureRecord(
                FailureCode.UNSUPPORTED_OUTPUT_COMBINATION,
                "The Byrnes backend does not expose the coherent complex amplitudes in VeriTMM; use the internal S-matrix backend for this output.",
                True,
                context={
                    "requested_solver": task.solver,
                    "requested_output": "amplitudes",
                    "compatible_solver": "smatrix",
                },
            )
        )
    wavelength_count = int(task.spectrum.wavelengths_nm().size)
    if "phase_dispersion" in outputs and wavelength_count < 3:
        failures.append(
            FailureRecord(
                FailureCode.UNSUPPORTED_OUTPUT_COMBINATION,
                "Phase dispersion requires at least three strictly increasing wavelength samples.",
                True,
                context={
                    "requested_output": "phase_dispersion",
                    "wavelength_points": wavelength_count,
                    "minimum_wavelength_points": 3,
                },
            )
        )
    if ("amplitudes" in outputs or "phase_dispersion" in outputs) and (
        "ellipsometry" in outputs or "layer_absorption" in outputs
    ):
        failures.append(
            FailureRecord(
                FailureCode.UNSUPPORTED_OUTPUT_COMBINATION,
                "Request complex amplitudes separately from Byrnes-only layer absorption or ellipsometry outputs.",
                True,
            )
        )

    resolved_solver = task.solver
    if not failures and (mixed or "ellipsometry" in outputs or "layer_absorption" in outputs):
        resolved_solver = "byrnes"
        if task.solver != "byrnes":
            warnings.append("The task was routed to the Byrnes backend to satisfy its requested outputs.")
    return CapabilityAssessment(
        "veritmm",
        not failures,
        resolved_solver if not failures else None,
        tuple(failures),
        tuple(warnings),
    )


__all__ = [
    "ActionSafety",
    "CapabilityAssessment",
    "FailureAction",
    "FailureCode",
    "FailureRecord",
    "PhysicsEngineError",
    "assess_tmm_capability",
    "enrich_failure_actions",
    "failure_from_exception",
]
