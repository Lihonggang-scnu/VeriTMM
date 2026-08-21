"""Static, machine-readable capability discovery for the public protocol."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Literal, TypeAlias, get_args, get_type_hints

from pydantic import Field

from ..schemas import SimulationTask
from .models import (
    PROTOCOL_VERSION,
    Coherence,
    Mode,
    ProtocolModel,
    RequestedOutput,
    Solver,
)

# This is the allow-list in SimulationTask.validate().  The internal dataclass
# intentionally remains unchanged; this named tuple makes the public protocol
# declaration auditable and test-locks the boundary to the current runtime.
SUPPORTED_REQUESTED_OUTPUTS: tuple[str, ...] = (
    "R",
    "T",
    "A",
    "amplitudes",
    "ellipsometry",
    "layer_absorption",
    "system_emissivity",
    "phase_dispersion",
)

# Solver names are read from the internal task annotation so a solver addition
# cannot silently go undocumented in the public manifest.
_RUNTIME_SOLVERS = tuple(get_args(get_type_hints(SimulationTask)["solver"]))
SUPPORTED_SOLVERS: tuple[str, ...] = tuple(str(item) for item in _RUNTIME_SOLVERS)

ArtifactType: TypeAlias = Literal[
    "normalized_task",
    "simulation_result",
    "optimization_result",
    "independent_validation",
    "physics_certificate",
    "preflight_report",
    "design_portfolio",
    "spectrum_table",
    "spectrum_plot",
    "plot_diagnostic",
    "legacy_run_manifest",
    "result_summary",
    "response_context",
    "run_result",
    "sweep_result",
    "sweep_table",
    "sensitivity_result",
    "tolerance_result",
    "robustness_report",
    "benchmark_result",
    "agent_trajectory",
    "agent_ab_result",
    "research_batch_manifest",
    "research_batch_index",
    "research_dataset_manifest",
    "research_dataset_index",
]


class UnitManifest(ProtocolModel):
    """Units used by the task contract."""

    wavelength: Literal["nm"] = "nm"
    thickness: Literal["nm"] = "nm"
    angle: Literal["deg"] = "deg"

    @property
    def wavelength_nm(self) -> str:
        """Compatibility accessor for callers using field-style unit names."""

        return self.wavelength

    @property
    def thickness_nm(self) -> str:
        """Compatibility accessor for callers using field-style unit names."""

        return self.thickness

    @property
    def angle_deg(self) -> str:
        """Compatibility accessor for callers using field-style unit names."""

        return self.angle


class MaterialModelManifest(ProtocolModel):
    """One supported passive optical-constant representation."""

    material_class: Literal["isotropic"] = "isotropic"
    passivity: Literal["passive"] = "passive"
    representation: Literal["scalar_nk"] = "scalar_nk"

    @property
    def class_name(self) -> str:
        """Alias for consumers that use ``class_name`` terminology."""

        return self.material_class


class CapabilityLimitation(ProtocolModel):
    """A declared boundary or a feature intentionally not formalized yet."""

    id: str
    status: Literal["unsupported", "not_formally_supported", "runtime_checked"]
    description: str


class MixedCoherenceManifest(ProtocolModel):
    """Support and output boundaries for coherent/incoherent layer mixtures."""

    supported: bool = True
    layer_coherence: tuple[Coherence, ...] = ("coherent", "incoherent")
    routing_solver: Literal["byrnes"] = "byrnes"
    supported_outputs: tuple[RequestedOutput, ...] = (
        "R",
        "T",
        "A",
        "layer_absorption",
        "system_emissivity",
    )
    unsupported_outputs: tuple[RequestedOutput, ...] = (
        "amplitudes",
        "ellipsometry",
        "phase_dispersion",
    )


class OptimizationManifest(ProtocolModel):
    """Declared inverse-design surface without overclaiming future workflows."""

    supported: bool = True
    methods: tuple[Literal["adam", "adam_lbfgs"], ...] = ("adam", "adam_lbfgs")
    parameterization: Literal["layer_thickness_nm"] = "layer_thickness_nm"
    objective_outputs: tuple[Literal["R", "T", "A"], ...] = ("R", "T", "A")
    features: tuple[str, ...] = (
        "multiband_targets",
        "thickness_bounds",
        "multistart",
        "quantization",
        "independent_reference_recompute",
        "robust_training_objectives",
        "independent_final_robustness_evaluation",
    )
    fixed_material_selection: bool = True
    not_formally_supported: tuple[str, ...] = ()


class ScientificAnalysisManifest(ProtocolModel):
    sensitivity_parameters: tuple[str, ...] = ("layer_thickness_nm",)
    sensitivity_audit: str = "independent_numpy_central_difference"
    uncertainty_parameters: tuple[str, ...] = ("layer_thickness_nm",)
    uncertainty_distributions: tuple[str, ...] = ("normal", "uniform")
    uncertainty_boundary_policies: tuple[str, ...] = ("truncate",)
    minimum_physical_thickness_nm: float = 0.1
    yield_interval: str = "wilson_score_interval"
    yield_denominator: str = "completed_sample_count"
    overall_success_denominator: str = "requested_sample_count"
    robust_objectives: tuple[str, ...] = (
        "expected_loss",
        "worst_case_loss",
        "mean_plus_k_sigma",
    )
    physics_validity_is_separate_from_robustness: bool = True


class AgentBenchManifest(ProtocolModel):
    """Offline evaluation surface; it never changes physics acceptance."""

    supported: bool = True
    offline_deterministic: bool = True
    llm_required: bool = False
    network_required: bool = False
    minimum_release_gate_cases: int = 80
    unsupported_false_accept_required: float = 0.0
    commands: tuple[str, ...] = ("benchmark", "agent-benchmark")
    metrics: tuple[str, ...] = (
        "valid_case_pass_rate",
        "invalid_case_rejection_rate",
        "expected_failure_code_accuracy",
        "artifact_completeness_rate",
        "certificate_success_rate",
        "reproducibility_rate",
        "unsupported_false_accept_rate",
    )
    transport_adapters: tuple[str, ...] = (
        "python_callable",
        "pre_recorded_trajectory",
    )
    mcp_status: Literal["optional_deferred"] = "optional_deferred"


class ResearchDesignSpaceManifest(ProtocolModel):
    """Deterministic candidate contract over the existing simulation task."""

    base_task_contract: Literal["SimulationTask"] = "SimulationTask"
    variable_kinds: tuple[str, ...] = (
        "continuous_thickness",
        "discrete_thickness",
        "material_choice",
    )
    fixed_layers_preserved: bool = True
    variable_layer_count: bool = False
    normalized_design_range: tuple[float, float] = (0.0, 1.0)
    identity: Literal["canonical_content_sha256"] = "canonical_content_sha256"


class ResearchObjectiveManifest(ProtocolModel):
    """Algorithm bookkeeping supported by the research interface."""

    observables: tuple[Literal["R", "T", "A"], ...] = ("R", "T", "A")
    directions: tuple[str, ...] = ("maximize", "minimize", "target")
    aggregations: tuple[str, ...] = ("mean", "min", "max")
    constraint_relations: tuple[str, ...] = ("at_least", "at_most")
    weighted_multi_objective: bool = True
    score_confers_physics_validity: bool = False


class ResearchEvaluationManifest(ProtocolModel):
    """Only managed execution and its verifier establish accepted physics."""

    evaluator: Literal["ResearchEvaluator"] = "ResearchEvaluator"
    managed_operation: Literal["simulate"] = "simulate"
    preflight_required: bool = True
    independent_certificate_required: bool = True
    batch_executor_replaceable: bool = True
    reference_batch_executor: Literal["sequential"] = "sequential"
    resumable_batch_ledger: bool = True
    response_profile: Literal["compact"] = "compact"


class ResearchDatasetManifest(ProtocolModel):
    """Artifact-backed verified dataset generation surface."""

    factory: Literal["DatasetFactory"] = "DatasetFactory"
    sampling_strategies: tuple[str, ...] = (
        "random",
        "grid",
        "latin_hypercube",
        "sobol",
    )
    sobol_core_max_dimension: int = 16
    deterministic_indexed_sampling: bool = True
    resumable: bool = True
    spectra_inline: bool = False


class ResearchAdapterManifest(ProtocolModel):
    """Algorithm-neutral optimizer, optional ML, and environment adapters."""

    optimizer_protocol: Literal["ask_tell"] = "ask_tell"
    reference_optimizer: Literal["random_search"] = "random_search"
    concrete_third_party_optimizers: bool = False
    torch_dataset: Literal["optional_lazy"] = "optional_lazy"
    gymnasium_required: bool = False
    environment: Literal["fixed_layer"] = "fixed_layer"
    actions: tuple[str, ...] = (
        "choose_material",
        "choose_thickness",
        "stop",
    )
    reserved_actions: tuple[str, ...] = ("add_layer", "remove_layer")


class ResearchInterfaceManifest(ProtocolModel):
    """Bounded discovery surface for the independent v0.6 research layer."""

    schema_version: Literal["veritmm-research-interface-v1"] = (
        "veritmm-research-interface-v1"
    )
    version: Literal["0.6.0"] = "0.6.0"
    supported: bool = True
    design_space: ResearchDesignSpaceManifest = Field(
        default_factory=ResearchDesignSpaceManifest
    )
    objectives: ResearchObjectiveManifest = Field(
        default_factory=ResearchObjectiveManifest
    )
    evaluation: ResearchEvaluationManifest = Field(
        default_factory=ResearchEvaluationManifest
    )
    dataset: ResearchDatasetManifest = Field(default_factory=ResearchDatasetManifest)
    adapters: ResearchAdapterManifest = Field(default_factory=ResearchAdapterManifest)
    validity_invariant: str = (
        "Algorithms propose; only managed VeriTMM execution plus an accepted "
        "independent physics certificate validates a result."
    )
    exclusions: tuple[str, ...] = (
        "variable_layer_count",
        "concrete_ml_or_rl_algorithms",
        "pinn_or_diffusion_models",
        "mcp_transport",
        "new_physics_or_solver_paths",
    )


class CapabilityManifest(ProtocolModel):
    """Complete public capability declaration for the current engine."""

    protocol_version: str = PROTOCOL_VERSION
    package_version: str
    capability_version: str = "tmm-isotropic-planar-v1"
    engine_id: str = "veritmm"
    modes: tuple[Mode, ...] = (
        "simulate",
        "optimize",
        "sweep",
        "sensitivity",
        "tolerance",
    )
    solvers: tuple[Solver, ...] = ("smatrix", "characteristic", "byrnes")
    geometry: tuple[Literal["layered_planar"], ...] = ("layered_planar",)
    material_models: tuple[MaterialModelManifest, ...] = (MaterialModelManifest(),)
    excitation: tuple[Literal["plane_wave"], ...] = ("plane_wave",)
    units: UnitManifest = Field(default_factory=UnitManifest)
    requested_outputs: tuple[RequestedOutput, ...] = SUPPORTED_REQUESTED_OUTPUTS  # type: ignore[assignment]
    limitations: tuple[CapabilityLimitation, ...] = (
        CapabilityLimitation(
            id="lateral_periodic_geometry",
            status="unsupported",
            description="Scalar TMM does not model gratings, metasurfaces, or diffraction orders.",
        ),
        CapabilityLimitation(
            id="anisotropic_or_tensor_materials",
            status="unsupported",
            description="Only passive isotropic scalar nk material data are supported.",
        ),
        CapabilityLimitation(
            id="non_plane_wave_excitation",
            status="unsupported",
            description="Finite beams, dipoles, and mode sources are outside this protocol.",
        ),
        CapabilityLimitation(
            id="time_domain_response",
            status="unsupported",
            description="The engine is frequency-domain TMM only.",
        ),
    )
    artifact_types: tuple[ArtifactType, ...] = (
        "normalized_task",
        "simulation_result",
        "optimization_result",
        "independent_validation",
        "physics_certificate",
        "preflight_report",
        "design_portfolio",
        "spectrum_table",
        "spectrum_plot",
        "plot_diagnostic",
        "legacy_run_manifest",
        "result_summary",
        "response_context",
        "run_result",
        "sweep_result",
        "sweep_table",
        "sensitivity_result",
        "tolerance_result",
        "robustness_report",
        "benchmark_result",
        "agent_trajectory",
        "agent_ab_result",
        "research_batch_manifest",
        "research_batch_index",
        "research_dataset_manifest",
        "research_dataset_index",
    )
    mixed_coherence: MixedCoherenceManifest = Field(default_factory=MixedCoherenceManifest)
    optimization: OptimizationManifest = Field(default_factory=OptimizationManifest)
    scientific_analysis: ScientificAnalysisManifest = Field(
        default_factory=ScientificAnalysisManifest
    )
    agent_bench: AgentBenchManifest = Field(default_factory=AgentBenchManifest)
    research_interface: ResearchInterfaceManifest = Field(
        default_factory=ResearchInterfaceManifest
    )

    @property
    def geometry_classes(self) -> tuple[str, ...]:
        """Compatibility accessor for the plural capability vocabulary."""

        return self.geometry

    @property
    def excitation_classes(self) -> tuple[str, ...]:
        """Compatibility accessor for the plural capability vocabulary."""

        return self.excitation

    @property
    def material_classes(self) -> tuple[str, ...]:
        """Return the supported material class names."""

        return tuple(item.material_class for item in self.material_models)


def describe_capabilities() -> CapabilityManifest:
    """Return a deterministic snapshot of the current public capability set."""

    # Construct a fresh model so a caller cannot mutate the process-wide
    # declaration through nested mutable state.
    # Prefer the executing source tree so an older editable installation cannot
    # make ``veritmm describe`` report stale metadata during an upgrade.
    try:
        from .. import __version__ as package_version
    except (ImportError, AttributeError):
        try:
            package_version = version("veritmm")
        except PackageNotFoundError:
            package_version = "0.6.0"
    return CapabilityManifest(
        package_version=package_version,
        modes=("simulate", "optimize", "sweep", "sensitivity", "tolerance"),
        solvers=SUPPORTED_SOLVERS,  # type: ignore[arg-type]
        requested_outputs=SUPPORTED_REQUESTED_OUTPUTS,  # type: ignore[arg-type]
    )


__all__ = [
    "AgentBenchManifest",
    "ArtifactType",
    "CapabilityLimitation",
    "CapabilityManifest",
    "MaterialModelManifest",
    "MixedCoherenceManifest",
    "OptimizationManifest",
    "ResearchAdapterManifest",
    "ResearchDatasetManifest",
    "ResearchDesignSpaceManifest",
    "ResearchEvaluationManifest",
    "ResearchInterfaceManifest",
    "ResearchObjectiveManifest",
    "ScientificAnalysisManifest",
    "SUPPORTED_REQUESTED_OUTPUTS",
    "SUPPORTED_SOLVERS",
    "UnitManifest",
    "describe_capabilities",
]
