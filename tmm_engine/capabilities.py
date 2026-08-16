"""Machine-readable capability and failure contracts for optical solvers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .schemas import SimulationTask


class FailureCode(str, Enum):
    INVALID_TASK = "invalid_task"
    UNSUPPORTED_GEOMETRY = "unsupported_geometry"
    UNSUPPORTED_MATERIAL_MODEL = "unsupported_material_model"
    UNSUPPORTED_EXCITATION = "unsupported_excitation"
    TIME_DOMAIN_REQUIRED = "time_domain_required"
    UNSUPPORTED_OUTPUT_COMBINATION = "unsupported_output_combination"
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


@dataclass(frozen=True)
class FailureRecord:
    code: FailureCode
    message: str
    recoverable: bool
    suggested_solver_family: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["code"] = self.code.value
        return payload


class PhysicsEngineError(RuntimeError):
    def __init__(self, failure: FailureRecord) -> None:
        super().__init__(failure.message)
        self.failure = failure


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
            "failures": [item.to_dict() for item in self.failures],
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
    "CapabilityAssessment",
    "FailureCode",
    "FailureRecord",
    "PhysicsEngineError",
    "assess_tmm_capability",
]
