"""Stable, model-agnostic contracts for multilayer optical simulations.

The contracts deliberately contain no LLM or optimizer logic.  A language model may
produce this structure later, but every downstream physics component consumes the
same validated representation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal, Optional, Sequence, Tuple

import numpy as np

Polarization = Literal["s", "p", "unpolarized"]
Coherence = Literal["coherent", "incoherent"]
Observable = Literal["R", "T", "A"]


@dataclass(frozen=True)
class PhysicsRequirements:
    """Domain-of-validity declaration supplied by an upstream planner.

    ``layered_planar`` includes multilayer films and one-dimensional photonic
    crystals whose refractive index varies only along the stack normal.  A
    grating that is periodic along the surface is ``lateral_periodic`` and is
    deliberately outside scalar TMM.
    """

    geometry_class: Literal[
        "layered_planar", "lateral_periodic", "arbitrary_2d", "arbitrary_3d"
    ] = "layered_planar"
    material_class: Literal[
        "isotropic", "anisotropic", "magneto_optic", "nonlinear"
    ] = "isotropic"
    excitation_class: Literal["plane_wave", "finite_beam", "dipole", "mode_source"] = (
        "plane_wave"
    )
    time_domain_required: bool = False

    def validate(self) -> None:
        if self.geometry_class not in (
            "layered_planar",
            "lateral_periodic",
            "arbitrary_2d",
            "arbitrary_3d",
        ):
            raise ValueError("unsupported geometry_class declaration")
        if self.material_class not in (
            "isotropic",
            "anisotropic",
            "magneto_optic",
            "nonlinear",
        ):
            raise ValueError("unsupported material_class declaration")
        if self.excitation_class not in ("plane_wave", "finite_beam", "dipole", "mode_source"):
            raise ValueError("unsupported excitation_class declaration")


def _finite_float(value: Any, name: str) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError("%s must be finite" % name)
    return out


@dataclass(frozen=True)
class MediumSpec:
    """A semi-infinite incident or exit medium.

    Either ``material`` or ``constant_n`` must be supplied.  ``constant_k`` is only
    meaningful together with ``constant_n``.
    """

    material: Optional[str] = None
    constant_n: Optional[float] = None
    constant_k: float = 0.0
    provider: Optional[str] = None
    dataset_id: Optional[str] = None

    def validate(self) -> None:
        if bool(self.material) == (self.constant_n is not None):
            raise ValueError("MediumSpec requires exactly one of material or constant_n")
        if self.constant_n is not None and _finite_float(self.constant_n, "constant_n") <= 0:
            raise ValueError("constant_n must be positive")
        if _finite_float(self.constant_k, "constant_k") < 0:
            raise ValueError("constant_k must be non-negative for passive media")

    @classmethod
    def air(cls) -> "MediumSpec":
        return cls(constant_n=1.0)


@dataclass(frozen=True)
class LayerSpec:
    material: Optional[str]
    thickness_nm: float
    coherence: Coherence = "coherent"
    provider: Optional[str] = None
    dataset_id: Optional[str] = None
    optimizable: bool = True
    min_thickness_nm: Optional[float] = None
    max_thickness_nm: Optional[float] = None
    label: Optional[str] = None
    constant_n: Optional[float] = None
    constant_k: float = 0.0

    def validate(self) -> None:
        if bool(self.material) == (self.constant_n is not None):
            raise ValueError("LayerSpec requires exactly one of material or constant_n")
        if self.constant_n is not None and _finite_float(self.constant_n, "constant_n") <= 0:
            raise ValueError("constant_n must be positive")
        if _finite_float(self.constant_k, "constant_k") < 0:
            raise ValueError("constant_k must be non-negative for passive media")
        thickness = _finite_float(self.thickness_nm, "thickness_nm")
        if thickness <= 0:
            raise ValueError("layer thickness_nm must be positive")
        if self.coherence not in ("coherent", "incoherent"):
            raise ValueError("coherence must be coherent or incoherent")
        lo = thickness if self.min_thickness_nm is None else _finite_float(self.min_thickness_nm, "min_thickness_nm")
        hi = thickness if self.max_thickness_nm is None else _finite_float(self.max_thickness_nm, "max_thickness_nm")
        if lo <= 0 or hi < lo:
            raise ValueError("invalid thickness bounds")
        if self.optimizable and not (lo <= thickness <= hi):
            raise ValueError("initial thickness is outside optimization bounds")

    def bounds_nm(self, default_window_nm: float = 150.0) -> Tuple[float, float]:
        lo = self.min_thickness_nm
        hi = self.max_thickness_nm
        if lo is None:
            lo = max(0.1, float(self.thickness_nm) - float(default_window_nm))
        if hi is None:
            hi = float(self.thickness_nm) + float(default_window_nm)
        return float(lo), float(hi)


@dataclass(frozen=True)
class StackSpec:
    layers: Tuple[LayerSpec, ...]
    incident: MediumSpec = field(default_factory=MediumSpec.air)
    exit: MediumSpec = field(default_factory=lambda: MediumSpec(material="sio2"))
    name: str = "multilayer_stack"

    def validate(self) -> None:
        self.incident.validate()
        self.exit.validate()
        if not self.layers:
            raise ValueError("stack must contain at least one finite layer")
        for layer in self.layers:
            layer.validate()

    @property
    def has_incoherent_layers(self) -> bool:
        return any(layer.coherence == "incoherent" for layer in self.layers)


@dataclass(frozen=True)
class SpectralGrid:
    start_nm: Optional[float] = None
    stop_nm: Optional[float] = None
    points: Optional[int] = None
    values_nm: Optional[Tuple[float, ...]] = None

    def wavelengths_nm(self) -> np.ndarray:
        if self.values_nm is not None:
            values = np.asarray(self.values_nm, dtype=np.float64)
        else:
            if self.start_nm is None or self.stop_nm is None or self.points is None:
                raise ValueError("spectral grid requires values_nm or start_nm/stop_nm/points")
            if int(self.points) < 2:
                raise ValueError("spectral grid points must be >= 2")
            values = np.linspace(float(self.start_nm), float(self.stop_nm), int(self.points), dtype=np.float64)
        if values.ndim != 1 or values.size < 1 or not np.all(np.isfinite(values)):
            raise ValueError("wavelengths must be a finite one-dimensional array")
        if np.any(values <= 0) or np.any(np.diff(values) <= 0):
            raise ValueError("wavelengths must be positive and strictly increasing")
        return values


@dataclass(frozen=True)
class IlluminationSpec:
    angles_deg: Tuple[float, ...] = (0.0,)
    polarizations: Tuple[Polarization, ...] = ("unpolarized",)

    def validate(self) -> None:
        if not self.angles_deg or not self.polarizations:
            raise ValueError("illumination requires at least one angle and polarization")
        for angle in self.angles_deg:
            val = _finite_float(angle, "angle_deg")
            if val < 0 or val >= 90:
                raise ValueError("angle_deg must satisfy 0 <= angle < 90")
        for pol in self.polarizations:
            if pol not in ("s", "p", "unpolarized"):
                raise ValueError("unsupported polarization: %s" % pol)


@dataclass(frozen=True)
class SimulationTask:
    stack: StackSpec
    spectrum: SpectralGrid
    illumination: IlluminationSpec = field(default_factory=IlluminationSpec)
    solver: Literal["smatrix", "characteristic", "byrnes"] = "smatrix"
    allow_material_extrapolation: bool = False
    requested_outputs: Tuple[str, ...] = ("R", "T", "A")
    physics: PhysicsRequirements = field(default_factory=PhysicsRequirements)

    def validate(self) -> None:
        self.stack.validate()
        self.spectrum.wavelengths_nm()
        self.illumination.validate()
        self.physics.validate()
        if self.solver not in ("smatrix", "characteristic", "byrnes"):
            raise ValueError("unsupported solver")
        allowed = {
            "R",
            "T",
            "A",
            "amplitudes",
            "ellipsometry",
            "layer_absorption",
            "system_emissivity",
            "phase_dispersion",
        }
        unknown = set(self.requested_outputs) - allowed
        if unknown:
            raise ValueError("unsupported requested outputs: %s" % sorted(unknown))


@dataclass(frozen=True)
class SpectralTarget:
    observable: Observable
    target: float
    wavelength_min_nm: float
    wavelength_max_nm: float
    weight: float = 1.0
    angle_deg: float = 0.0
    polarization: Polarization = "unpolarized"
    tolerance: Optional[float] = None
    name: Optional[str] = None
    constraint: Literal["match", "at_least", "at_most"] = "match"
    aggregation: Literal["mean", "worst_case"] = "mean"

    def validate(self) -> None:
        if self.observable not in ("R", "T", "A"):
            raise ValueError("target observable must be R, T, or A")
        if not (0.0 <= _finite_float(self.target, "target") <= 1.0):
            raise ValueError("target must be in [0, 1]")
        lo = _finite_float(self.wavelength_min_nm, "wavelength_min_nm")
        hi = _finite_float(self.wavelength_max_nm, "wavelength_max_nm")
        if lo <= 0 or hi < lo:
            raise ValueError("invalid target wavelength band")
        if _finite_float(self.weight, "weight") <= 0:
            raise ValueError("target weight must be positive")
        if self.constraint not in ("match", "at_least", "at_most"):
            raise ValueError("constraint must be match, at_least, or at_most")
        if self.aggregation not in ("mean", "worst_case"):
            raise ValueError("aggregation must be mean or worst_case")
        if self.tolerance is not None and _finite_float(self.tolerance, "tolerance") < 0:
            raise ValueError("tolerance must be non-negative")


@dataclass(frozen=True)
class OptimizerSpec:
    method: Literal["adam", "adam_lbfgs"] = "adam_lbfgs"
    max_steps: int = 120
    learning_rate: float = 0.05
    starts: int = 4
    seed: int = 42
    early_stop_patience: int = 20
    improvement_tolerance: float = 1e-7
    gradient_clip_norm: float = 10.0
    thickness_window_nm: float = 150.0
    quantization_nm: Optional[float] = None

    def validate(self) -> None:
        if self.method not in ("adam", "adam_lbfgs"):
            raise ValueError("unsupported optimizer method")
        if int(self.max_steps) < 1 or int(self.starts) < 1:
            raise ValueError("max_steps and starts must be positive")
        if _finite_float(self.learning_rate, "learning_rate") <= 0:
            raise ValueError("learning_rate must be positive")
        if self.quantization_nm is not None and float(self.quantization_nm) <= 0:
            raise ValueError("quantization_nm must be positive")


@dataclass(frozen=True)
class RobustnessSpec:
    enabled: bool = True
    distribution: Literal["normal"] = "normal"
    objective: Literal["expected_loss", "worst_case_loss", "mean_plus_k_sigma", "cvar"] = (
        "expected_loss"
    )
    samples_per_step: int = 8
    final_samples: int = 128
    seed: int = 42
    thickness_sigma_nm: float = 2.0
    k_sigma: float = 2.0
    cvar_alpha: Optional[float] = None
    boundary_policy: Literal["truncate"] = "truncate"
    min_thickness_physical_nm: float = 0.1

    def validate(self) -> None:
        if self.distribution != "normal":
            raise ValueError("unsupported robustness distribution")
        if self.objective not in (
            "expected_loss",
            "worst_case_loss",
            "mean_plus_k_sigma",
            "cvar",
        ):
            raise ValueError("unsupported robustness objective")
        if self.cvar_alpha is not None and not 0.0 < _finite_float(
            self.cvar_alpha, "cvar_alpha"
        ) < 1.0:
            raise ValueError("cvar_alpha must satisfy 0 < cvar_alpha < 1")
        if self.objective == "cvar" and self.cvar_alpha is None:
            raise ValueError("cvar objective requires cvar_alpha")
        if int(self.samples_per_step) < 2 or int(self.final_samples) < 8:
            raise ValueError("robustness sample counts are too small")
        if _finite_float(self.thickness_sigma_nm, "thickness_sigma_nm") <= 0:
            raise ValueError("thickness_sigma_nm must be positive")
        if _finite_float(self.k_sigma, "k_sigma") < 0:
            raise ValueError("k_sigma must be non-negative")
        if self.boundary_policy != "truncate":
            raise ValueError("unsupported robustness boundary_policy")
        if _finite_float(
            self.min_thickness_physical_nm, "min_thickness_physical_nm"
        ) <= 0:
            raise ValueError("min_thickness_physical_nm must be positive")


@dataclass(frozen=True)
class OptimizationTask:
    simulation: SimulationTask
    targets: Tuple[SpectralTarget, ...]
    optimizer: OptimizerSpec = field(default_factory=OptimizerSpec)
    robustness: Optional[RobustnessSpec] = None

    def validate(self) -> None:
        self.simulation.validate()
        self.optimizer.validate()
        if self.robustness is not None:
            self.robustness.validate()
        if not self.targets:
            raise ValueError("optimization task requires at least one target")
        for target in self.targets:
            target.validate()
            wavelengths = self.simulation.spectrum.wavelengths_nm()
            if not np.any(
                (wavelengths >= float(target.wavelength_min_nm))
                & (wavelengths <= float(target.wavelength_max_nm))
            ):
                raise ValueError("target band does not overlap the simulation grid")
            available = {
                (float(angle), str(pol))
                for angle in self.simulation.illumination.angles_deg
                for pol in self.simulation.illumination.polarizations
            }
            if (float(target.angle_deg), str(target.polarization)) not in available:
                raise ValueError(
                    "target angle/polarization must be declared in simulation illumination"
                )
        if not any(layer.optimizable for layer in self.simulation.stack.layers):
            raise ValueError("optimization task has no optimizable layers")


def stack_from_legacy(
    materials: Sequence[str],
    thicknesses_nm: Sequence[float],
    substrate: str = "sio2",
    *,
    n_incident: float = 1.0,
) -> StackSpec:
    if len(materials) != len(thicknesses_nm):
        raise ValueError("materials and thicknesses_nm must have equal length")
    return StackSpec(
        layers=tuple(LayerSpec(str(m), float(d)) for m, d in zip(materials, thicknesses_nm)),
        incident=MediumSpec(constant_n=float(n_incident)),
        exit=MediumSpec(material=str(substrate)),
    )


def dataclass_to_dict(value: Any) -> Dict[str, Any]:
    return asdict(value)


__all__ = [
    "Coherence",
    "IlluminationSpec",
    "LayerSpec",
    "MediumSpec",
    "Observable",
    "OptimizationTask",
    "OptimizerSpec",
    "RobustnessSpec",
    "Polarization",
    "PhysicsRequirements",
    "SimulationTask",
    "SpectralGrid",
    "SpectralTarget",
    "StackSpec",
    "dataclass_to_dict",
    "stack_from_legacy",
]
