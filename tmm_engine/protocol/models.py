"""Pydantic v2 models for VeriTMM's AI-facing public protocol.

The protocol models describe the shape of a task and its machine-readable
execution envelope.  They intentionally do not duplicate the numerical
validation performed by :mod:`tmm_engine.schemas` and the runtime preflight
checks.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..capabilities import FailureCode as _RuntimeFailureCode
from .evidence import EvidenceCoverage
from .uncertainty_budget import UncertaintyBudget

PROTOCOL_VERSION = "veritmm-agent-v1"

Mode: TypeAlias = Literal["simulate", "optimize", "sweep", "sensitivity", "tolerance"]
ResponseProfile: TypeAlias = Literal["compact", "standard", "full"]
Solver: TypeAlias = Literal["smatrix", "characteristic", "byrnes"]
GeometryClass: TypeAlias = Literal[
    "layered_planar", "lateral_periodic", "arbitrary_2d", "arbitrary_3d"
]
MaterialClass: TypeAlias = Literal[
    "isotropic", "anisotropic", "magneto_optic", "nonlinear"
]
ExcitationClass: TypeAlias = Literal[
    "plane_wave", "finite_beam", "dipole", "mode_source"
]
Polarization: TypeAlias = Literal["s", "p", "unpolarized"]
Coherence: TypeAlias = Literal["coherent", "incoherent"]
Observable: TypeAlias = Literal["R", "T", "A"]
RequestedOutput: TypeAlias = Literal[
    "R",
    "T",
    "A",
    "amplitudes",
    "ellipsometry",
    "layer_absorption",
    "system_emissivity",
    "phase_dispersion",
]

# Re-export the runtime enum under the protocol namespace.  Keeping one source
# for failure codes prevents the public envelope from drifting from runtime
# failure records.
FailureCode = _RuntimeFailureCode
PositiveFloat = Annotated[float, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0)]
AngleDegrees = Annotated[float, Field(ge=0, lt=90)]


class ProtocolModel(BaseModel):
    """Common strict configuration for public protocol models."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class FailureAction(ProtocolModel):
    """Machine-readable next step emitted by the runtime failure contract."""

    action_id: str
    action_type: str
    description: str
    safety: Literal["safe", "requires_scientific_judgment", "requires_user_input"]
    patch: list[dict[str, Any]] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)


class MediumContract(ProtocolModel):
    """A semi-infinite medium selector or a constant scalar refractive index."""

    material: str | None = None
    constant_n: PositiveFloat | None = None
    constant_k: NonNegativeFloat = 0.0
    provider: str | None = None
    dataset_id: str | int | None = None


class LayerContract(ProtocolModel):
    """One finite scalar-nk layer in a planar stack."""

    material: str | None = None
    thickness_nm: PositiveFloat
    coherence: Coherence = "coherent"
    provider: str | None = None
    dataset_id: str | int | None = None
    optimizable: bool = True
    min_thickness_nm: PositiveFloat | None = None
    max_thickness_nm: PositiveFloat | None = None
    label: str | None = None
    constant_n: PositiveFloat | None = None
    constant_k: NonNegativeFloat = 0.0


class StackContract(ProtocolModel):
    """Planar stack data shared by simulation and optimization tasks."""

    layers: list[LayerContract] = Field(min_length=1)
    incident: MediumContract = Field(
        default_factory=lambda: MediumContract(constant_n=1.0)
    )
    exit: MediumContract = Field(default_factory=lambda: MediumContract(material="sio2"))
    name: str = "multilayer_stack"


class SpectralGridContract(ProtocolModel):
    """A wavelength grid in nanometres.

    The mutually exclusive grid forms and monotonicity are runtime concerns;
    this model only describes their JSON shape.
    """

    start_nm: PositiveFloat | None = None
    stop_nm: PositiveFloat | None = None
    points: Annotated[int, Field(ge=2)] | None = None
    values_nm: list[PositiveFloat] | None = None


class IlluminationContract(ProtocolModel):
    """Plane-wave angles and polarization channels."""

    angles_deg: list[AngleDegrees] = Field(default_factory=lambda: [0.0], min_length=1)
    polarizations: list[Polarization] = Field(
        default_factory=lambda: ["unpolarized"], min_length=1
    )


class PhysicsRequirementsContract(ProtocolModel):
    """Requested physics classes, including values that preflight may reject."""

    geometry_class: GeometryClass = "layered_planar"
    material_class: MaterialClass = "isotropic"
    excitation_class: ExcitationClass = "plane_wave"
    time_domain_required: bool = False


class SimulationTaskPayload(ProtocolModel):
    """Unwrapped simulation fields used inside ``simulation``."""

    stack: StackContract
    spectrum: SpectralGridContract
    illumination: IlluminationContract = Field(default_factory=IlluminationContract)
    solver: Solver = "smatrix"
    allow_material_extrapolation: bool = False
    requested_outputs: list[RequestedOutput] = Field(default_factory=lambda: ["R", "T", "A"])
    physics: PhysicsRequirementsContract = Field(default_factory=PhysicsRequirementsContract)


class SpectralTargetContract(ProtocolModel):
    """One optimization target over an observable and wavelength band."""

    observable: Observable
    target: Annotated[float, Field(ge=0, le=1)]
    wavelength_min_nm: PositiveFloat
    wavelength_max_nm: PositiveFloat
    weight: PositiveFloat = 1.0
    angle_deg: AngleDegrees = 0.0
    polarization: Polarization = "unpolarized"
    tolerance: NonNegativeFloat | None = None
    name: str | None = None
    constraint: Literal["match", "at_least", "at_most"] = "match"
    aggregation: Literal["mean", "worst_case"] = "mean"


class OptimizerContract(ProtocolModel):
    """Differentiable thickness-optimization settings."""

    method: Literal["adam", "adam_lbfgs"] = "adam_lbfgs"
    max_steps: Annotated[int, Field(ge=1)] = 120
    learning_rate: PositiveFloat = 0.05
    starts: Annotated[int, Field(ge=1)] = 4
    seed: int = 42
    early_stop_patience: Annotated[int, Field(ge=1)] = 20
    improvement_tolerance: NonNegativeFloat = 1e-7
    gradient_clip_norm: PositiveFloat = 10.0
    thickness_window_nm: PositiveFloat = 150.0
    quantization_nm: PositiveFloat | None = None


class RobustnessContract(ProtocolModel):
    """Stochastic objective settings; never a physics-validity declaration."""

    enabled: bool = True
    distribution: Literal["normal"] = "normal"
    objective: Literal["expected_loss", "worst_case_loss", "mean_plus_k_sigma", "cvar"] = (
        "expected_loss"
    )
    samples_per_step: Annotated[int, Field(ge=2, le=256)] = 8
    final_samples: Annotated[int, Field(ge=8, le=100_000)] = 128
    seed: int = 42
    thickness_sigma_nm: PositiveFloat = 2.0
    k_sigma: NonNegativeFloat = 2.0
    cvar_alpha: Annotated[float, Field(gt=0, lt=1)] | None = None
    boundary_policy: Literal["truncate"] = "truncate"
    min_thickness_physical_nm: PositiveFloat = 0.1

    @model_validator(mode="after")
    def _validate_cvar(self) -> "RobustnessContract":
        if self.objective == "cvar" and self.cvar_alpha is None:
            raise ValueError("cvar objective requires cvar_alpha")
        return self


class OptimizationTaskPayload(ProtocolModel):
    """Unwrapped optimization fields used inside ``optimization``."""

    simulation: SimulationTaskPayload
    targets: list[SpectralTargetContract]
    optimizer: OptimizerContract = Field(default_factory=OptimizerContract)
    robustness: RobustnessContract | None = None


class SimulationTaskContract(ProtocolModel):
    """Version-compatible simulation request wrapper.

    The canonical form is ``{"mode": "simulate", "simulation": {...}}``.
    For callers migrating from an unwrapped v0.1 payload, the inner simulation
    mapping is also accepted and normalized to that wrapper.
    """

    mode: Literal["simulate"] = "simulate"
    simulation: SimulationTaskPayload

    @model_validator(mode="before")
    @classmethod
    def _wrap_unwrapped_payload(cls, value: Any) -> Any:
        if isinstance(value, Mapping) and "simulation" not in value and "stack" in value:
            inner = dict(value)
            inner.pop("mode", None)
            return {"mode": "simulate", "simulation": inner}
        return value

    @model_validator(mode="after")
    def _match_runtime_contract(self) -> "SimulationTaskContract":
        from ..task_io import simulation_task_from_dict

        simulation_task_from_dict(self.simulation.model_dump(mode="python"))
        return self


class OptimizationTaskContract(ProtocolModel):
    """Version-compatible optimization request wrapper."""

    mode: Literal["optimize"] = "optimize"
    optimization: OptimizationTaskPayload

    @model_validator(mode="before")
    @classmethod
    def _wrap_unwrapped_payload(cls, value: Any) -> Any:
        if (
            isinstance(value, Mapping)
            and "optimization" not in value
            and "simulation" in value
            and "targets" in value
        ):
            inner = dict(value)
            inner.pop("mode", None)
            return {"mode": "optimize", "optimization": inner}
        return value

    @model_validator(mode="after")
    def _match_runtime_contract(self) -> "OptimizationTaskContract":
        from ..task_io import optimization_task_from_dict

        optimization_task_from_dict(self.optimization.model_dump(mode="python"))
        return self


class MetricContract(ProtocolModel):
    """A deterministic scalar reduction over one simulated channel."""

    name: str
    observable: Literal["R", "T", "A", "E_system"]
    wavelength_min_nm: PositiveFloat | None = None
    wavelength_max_nm: PositiveFloat | None = None
    wavelength_nm: PositiveFloat | None = None
    aggregation: Literal[
        "mean", "min", "max", "worst_case", "value_at_wavelength", "threshold_band_width"
    ] = "mean"
    angle_deg: AngleDegrees = 0.0
    polarization: Polarization = "unpolarized"
    threshold: Annotated[float, Field(ge=0, le=1)] | None = None
    threshold_direction: Literal["at_least", "at_most"] = "at_least"

    @model_validator(mode="after")
    def _validate_metric_shape(self) -> "MetricContract":
        if self.aggregation == "value_at_wavelength" and self.wavelength_nm is None:
            raise ValueError("value_at_wavelength requires wavelength_nm")
        if self.aggregation == "threshold_band_width" and self.threshold is None:
            raise ValueError("threshold_band_width requires threshold")
        if (self.wavelength_min_nm is None) != (self.wavelength_max_nm is None):
            raise ValueError("metric wavelength band requires both minimum and maximum")
        if (
            self.wavelength_min_nm is not None
            and self.wavelength_max_nm is not None
            and self.wavelength_max_nm < self.wavelength_min_nm
        ):
            raise ValueError("metric wavelength_max_nm must be >= wavelength_min_nm")
        return self


_ALLOWED_SWEEP_PATHS = (
    "/stack/layers/",
    "/illumination/angles_deg/",
    "/spectrum/start_nm",
    "/spectrum/stop_nm",
    "/spectrum/points",
)


class SweepParameterContract(ProtocolModel):
    """One finite allow-listed JSON-pointer parameter axis."""

    path: str
    values: list[float | int] = Field(min_length=1)

    @field_validator("path")
    @classmethod
    def _allowed_path(cls, value: str) -> str:
        if value.startswith("/stack/layers/") and value.endswith("/thickness_nm"):
            middle = value[len("/stack/layers/") : -len("/thickness_nm")]
            if middle.isdigit():
                return value
        if value.startswith("/illumination/angles_deg/"):
            if value.rsplit("/", 1)[-1].isdigit():
                return value
        if value in _ALLOWED_SWEEP_PATHS[2:]:
            return value
        raise ValueError("sweep parameter path is not in the public allow-list")


class SweepTaskPayload(ProtocolModel):
    """Finite Cartesian parameter study over a base simulation."""

    base_simulation: SimulationTaskPayload
    parameters: list[SweepParameterContract] = Field(min_length=1)
    metrics: list[MetricContract] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_axes_and_metrics(self) -> "SweepTaskPayload":
        paths = [axis.path for axis in self.parameters]
        if len(paths) != len(set(paths)):
            raise ValueError("sweep parameter paths must be unique")
        names = [metric.name for metric in self.metrics]
        if len(names) != len(set(names)):
            raise ValueError("sweep metric names must be unique")
        return self


class SweepTaskContract(ProtocolModel):
    """Versioned public sweep request."""

    schema_version: Literal["sweep-task-v1"] = "sweep-task-v1"
    mode: Literal["sweep"] = "sweep"
    sweep: SweepTaskPayload


class SensitivityTaskPayload(ProtocolModel):
    """Thickness sensitivity with an independent finite-difference audit."""

    simulation: SimulationTaskPayload
    metric: MetricContract
    parameters: Literal["optimizable_thicknesses"] = "optimizable_thicknesses"
    finite_difference_step_nm: PositiveFloat | None = None
    relative_error_tolerance: PositiveFloat = 1e-3
    absolute_error_tolerance: PositiveFloat = 1e-7


class SensitivityTaskContract(ProtocolModel):
    schema_version: Literal["sensitivity-task-v1"] = "sensitivity-task-v1"
    mode: Literal["sensitivity"] = "sensitivity"
    sensitivity: SensitivityTaskPayload


class LayerUncertaintyContract(ProtocolModel):
    layer_index: Annotated[int, Field(ge=0)]
    distribution: Literal["normal", "uniform"]
    sigma_nm: PositiveFloat | None = None
    half_width_nm: PositiveFloat | None = None

    @model_validator(mode="after")
    def _distribution_parameter(self) -> "LayerUncertaintyContract":
        if self.distribution == "normal" and self.sigma_nm is None:
            raise ValueError("normal uncertainty requires sigma_nm")
        if self.distribution == "uniform" and self.half_width_nm is None:
            raise ValueError("uniform uncertainty requires half_width_nm")
        if self.distribution == "normal" and self.half_width_nm is not None:
            raise ValueError("normal uncertainty cannot define half_width_nm")
        if self.distribution == "uniform" and self.sigma_nm is not None:
            raise ValueError("uniform uncertainty cannot define sigma_nm")
        return self


class YieldTargetContract(ProtocolModel):
    metric: MetricContract
    constraint: Literal["at_least", "at_most"]
    value: Annotated[float, Field(ge=0, le=1)]


class ToleranceTaskPayload(ProtocolModel):
    """Finite seeded thickness-uncertainty study."""

    simulation: SimulationTaskPayload
    uncertainties: list[LayerUncertaintyContract] = Field(min_length=1)
    metric: MetricContract
    target: YieldTargetContract
    sample_count: Annotated[int, Field(ge=1, le=1_000_000)] = 200
    seed: int = 42
    global_correlated_bias_nm: NonNegativeFloat | None = None
    boundary_policy: Literal["truncate"] = "truncate"
    min_thickness_physical_nm: PositiveFloat = 0.1

    @model_validator(mode="after")
    def _unique_layers(self) -> "ToleranceTaskPayload":
        indices = [item.layer_index for item in self.uncertainties]
        if len(indices) != len(set(indices)):
            raise ValueError("uncertainty layer_index values must be unique")
        if self.target.metric != self.metric:
            raise ValueError("target.metric must equal the study metric")
        return self


class ToleranceTaskContract(ProtocolModel):
    schema_version: Literal["tolerance-task-v1"] = "tolerance-task-v1"
    mode: Literal["tolerance"] = "tolerance"
    tolerance: ToleranceTaskPayload


class FailureRecordModel(ProtocolModel):
    """Pydantic representation of the runtime failure record."""

    code: FailureCode
    message: str
    recoverable: bool
    suggested_solver_family: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    severity: Literal["warning", "error", "fatal"] = "error"
    requires_user_choice: bool = False
    actions: list[FailureAction] = Field(default_factory=list)


class PhysicsCertificate(ProtocolModel):
    """Additive typed view of the runtime physics certificate mapping."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    accepted: bool
    limitations: list[str] = Field(default_factory=list)
    evidence_coverage: EvidenceCoverage | None = None
    uncertainty_budget: UncertaintyBudget | None = None


class PreflightReport(ProtocolModel):
    """Pydantic representation of :func:`tmm_engine.preflight.preflight_task`."""

    schema_version: str = "veritmm-preflight-v1"
    protocol_version: str = PROTOCOL_VERSION
    ok: bool
    operation: Literal["preflight"] = "preflight"
    mode: Literal["simulate", "optimize", "sweep", "sensitivity", "tolerance", "unknown"]
    status: Literal["ready", "rejected"]
    contract_valid: bool
    capability: dict[str, Any] | None
    backend_resolution: dict[str, Any]
    materials: list[dict[str, Any]]
    warnings: list[dict[str, Any]]
    failures: list[FailureRecordModel]
    estimated_work: dict[str, Any]
    study: dict[str, Any] | None = None


class ArtifactRef(ProtocolModel):
    """Reference emitted by :func:`tmm_engine.run_artifacts.index_artifacts`."""

    kind: str
    path: str
    schema_version: str
    sha256: str
    size_bytes: int


class ResponseMetadata(ProtocolModel):
    """Versioned profile metadata nested in a response summary."""

    schema_version: Literal["veritmm-response-v1"] = "veritmm-response-v1"
    profile: ResponseProfile
    available_profiles: tuple[ResponseProfile, ...] = ("compact", "standard", "full")
    artifact_backed: bool = False
    detail_available_via_profile: bool = False
    artifact_summary: dict[str, Any] = Field(default_factory=dict)
    context_budget: dict[str, Any] | None = None
    source_retention: dict[str, Any] = Field(default_factory=dict)
    omitted_fields: dict[str, int] = Field(default_factory=dict)
    truncated_fields: dict[str, int] = Field(default_factory=dict)


class RunResultEnvelope(ProtocolModel):
    """Pydantic representation of :func:`tmm_engine.run_artifacts.write_run_result`."""

    schema_version: str = "veritmm-run-result-v1"
    protocol_version: str = PROTOCOL_VERSION
    ok: bool
    run_id: str
    task_sha256: str | None
    task_hash_scope: Literal["normalized_operation_wrapper"]
    archive_schema_version: int | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    input_sha256: str | None
    operation: str
    status: str
    summary: dict[str, Any]
    warnings: list[Any]
    failures: list[FailureRecordModel] = Field(default_factory=list)
    certificate_id: str | None
    artifacts: list[ArtifactRef]
    next_machine_actions: list[FailureAction]
    cache_hit: bool = False
    source_run_id: str | None = None
    artifact_provenance: dict[str, Any] | None = None


__all__ = [
    "ArtifactRef",
    "Coherence",
    "ExcitationClass",
    "FailureAction",
    "FailureCode",
    "FailureRecordModel",
    "GeometryClass",
    "IlluminationContract",
    "LayerContract",
    "MaterialClass",
    "MediumContract",
    "Mode",
    "Observable",
    "OptimizationTaskContract",
    "OptimizationTaskPayload",
    "OptimizerContract",
    "PhysicsRequirementsContract",
    "Polarization",
    "PreflightReport",
    "PhysicsCertificate",
    "PROTOCOL_VERSION",
    "ProtocolModel",
    "RequestedOutput",
    "ResponseMetadata",
    "ResponseProfile",
    "RunResultEnvelope",
    "SimulationTaskContract",
    "SimulationTaskPayload",
    "Solver",
    "SpectralGridContract",
    "SpectralTargetContract",
    "StackContract",
    "MetricContract",
    "SweepParameterContract",
    "SweepTaskContract",
    "SweepTaskPayload",
    "RobustnessContract",
    "SensitivityTaskContract",
    "SensitivityTaskPayload",
    "LayerUncertaintyContract",
    "YieldTargetContract",
    "ToleranceTaskContract",
    "ToleranceTaskPayload",
]
