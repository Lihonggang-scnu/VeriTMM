"""VeriTMM — configurable optical multilayer simulation and verification.

Usage (minimal):
  from tmm_engine import TMMEngine
  engine = TMMEngine()
  result = engine.simulate(
      materials=["TiO2", "MgF2", "Ag", "TiO2", "MgF2", "TiO2"],
      thicknesses_nm=[130, 130, 17, 188, 57, 88],
      substrate="SiO2",
  )
  # result.R, result.T, result.A  (arrays), result.wavelengths_nm

Usage (full control):
  result = engine.simulate(
      materials=["TiO2", "SiO2"],
      thicknesses_nm=[100, 150],
      substrate="Si",
      wavelengths_nm=(400, 700, 201),       # start, end, num_points
      angle_deg=30,                           # incident angle
      polarization="s",                       # "s", "p", or "unpolarized"
      solver="smatrix",                       # "smatrix", "abeles", "cm"
      substrate_mode="semi_infinite",         # "semi_infinite" or "finite"
      n_incident=1.0,                         # incident medium refractive index
  )

nk lookup:
  engine.lookup_nk("TiO2", wavelengths)       # → complex array n+ik
  engine.list_materials()                     # → ["SiO2", "TiO2", ...]
  engine.list_available_materials()           # compatibility alias
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import numpy as np

from ._version import __version__
from .tmm_solver import TMM, TMMConfig

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_MATERIALS_DIR = PACKAGE_ROOT / "materials"


def _load_local_csv(material: str, wl_um: np.ndarray) -> np.ndarray | None:
    """Load nk from local CSV file. Returns complex array or None."""
    path = DEFAULT_MATERIALS_DIR / f"{material.lower()}.csv"
    if not path.exists():
        return None
    try:
        data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
        wl_data = data[:, 0]
        n_data = data[:, 1]
        k_data = data[:, 2] if data.shape[1] >= 3 else np.zeros_like(n_data)
        if np.min(wl_um) < np.min(wl_data) or np.max(wl_um) > np.max(wl_data):
            return None
        n_interp = np.interp(wl_um, wl_data, n_data)
        k_interp = np.interp(wl_um, wl_data, k_data)
        return n_interp + 1j * np.maximum(k_interp, 0.0)
    except Exception:
        return None


class NkDatabase:
    """Multi-source nk lookup: local CSV → refractiveindex.info (if available)."""

    def __init__(self):
        self._cache: dict[str, dict[tuple, np.ndarray]] = {}
        self._rii_available = None  # lazy check

    def _try_rii(self, material: str, wl_um: np.ndarray) -> np.ndarray | None:
        """Try refractiveindex.info online database. Returns None if unavailable."""
        if self._rii_available is False:
            return None
        try:
            from refractivesqlite import Database
            if self._rii_available is None:
                rii_path = str(PACKAGE_ROOT / "rii_cache.db")
                db = Database(rii_path)
                # Test with a known material
                db.get_material(421)
                self._rii_available = True
                self._rii_db = db
            mat_name = material.capitalize()
            results = self._rii_db.search_pages(mat_name) or []
            for r in results:
                if r.hasrefractive:
                    mat = self._rii_db.get_material(r.pageid)
                    if mat.rangeMin <= wl_um[0] and mat.rangeMax >= wl_um[-1]:
                        ns = np.array([mat.get_refractiveindex(w) for w in wl_um], dtype=np.float64)
                        ks = np.array([mat.get_extinctioncoefficient(w) for w in wl_um], dtype=np.float64)
                        if not np.all(np.isfinite(ns)) or not np.all(np.isfinite(ks)):
                            continue
                        ks = np.maximum(ks, 0.0)
                        return ns + 1j * ks
            return None
        except Exception:
            self._rii_available = False
            return None

    def _cache_key(self, wl: np.ndarray) -> tuple:
        values = np.ascontiguousarray(np.asarray(wl, dtype=np.float64).reshape(-1))
        return (values.size, hashlib.sha256(values.tobytes()).hexdigest())

    def get_nk(self, material: str, wl_um: np.ndarray) -> np.ndarray:
        key = self._cache_key(wl_um)
        if material in self._cache and key in self._cache[material]:
            return self._cache[material][key]

        # 1. Local CSV (fast, reliable)
        nk = _load_local_csv(material, wl_um)
        # 2. refractiveindex.info online (if installed and working)
        if nk is None:
            nk = self._try_rii(material, wl_um)
        if nk is None:
            raise ValueError(
                f"No non-extrapolated optical-constant dataset covers material '{material}'. "
                f"Bundled CSV materials: {self.list_available()}. "
                "Use MaterialRegistry.search() to select an explicit refractiveindex.info dataset."
            )
        self._cache.setdefault(material, {})[key] = nk
        return nk

    def list_available(self) -> list[str]:
        return sorted(
            p.stem for p in DEFAULT_MATERIALS_DIR.glob("*.csv")
            if not p.stem.startswith("材料")
        )


@dataclass
class SimulationConfig:
    """Fully configurable simulation parameters.

    Agent can adjust ANY field for a specific task.
    """

    # ── spectral grid ──
    wl_start_nm: float = 380.0
    wl_end_nm: float = 780.0
    wl_points: int = 401

    # ── illumination ──
    angle_deg: float = 0.0
    polarization: Literal["s", "p", "unpolarized"] = "unpolarized"
    n_incident: float = 1.0

    # ── solver ──
    solver: Literal["smatrix", "abeles", "cm"] = "smatrix"

    # ── substrate ──
    substrate_mode: Literal["semi_infinite", "finite"] = "semi_infinite"

    # ── output options ──
    clip_R: bool = True
    clip_T: bool = False
    compute_absorption: bool = True

    def wavelengths_nm(self) -> np.ndarray:
        return np.linspace(self.wl_start_nm, self.wl_end_nm, self.wl_points, dtype=np.float64)

    def wavelengths_um(self) -> np.ndarray:
        return self.wavelengths_nm() * 1e-3


@dataclass
class SimulationResult:
    wavelengths_nm: np.ndarray
    R: np.ndarray
    T: np.ndarray
    A: np.ndarray | None = None
    materials: list[str] = field(default_factory=list)
    thicknesses_nm: list[float] = field(default_factory=list)
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "wavelengths_nm": self.wavelengths_nm.tolist(),
            "R": self.R.tolist(),
            "T": self.T.tolist(),
            "materials": self.materials,
            "thicknesses_nm": self.thicknesses_nm,
        }
        if self.A is not None:
            d["A"] = self.A.tolist()
        return d

    def summary(self) -> str:
        lines = [
            f"TMM Simulation: {'/'.join(self.materials)} on substrate",
            f"  Wavelength: {self.wavelengths_nm[0]:.1f}–{self.wavelengths_nm[-1]:.1f} nm"
            f" ({len(self.wavelengths_nm)} points)",
            f"  R avg: {np.mean(self.R):.4f}  T avg: {np.mean(self.T):.4f}",
        ]
        if self.A is not None:
            lines.append(f"  A avg: {np.mean(self.A):.4f}")
        return "\n".join(lines)


class TMMEngine:
    """Configurable TMM engine for Agent-driven optical multilayer simulation.

    The Agent reads TMM_SKILL.md for usage, creates a SimulationConfig,
    and calls simulate(). All parameters are exposed.
    """

    def __init__(self, materials_dir: str | Path | None = None):
        self.nk_db = NkDatabase()
        # Override materials dir if specified
        if materials_dir:
            global DEFAULT_MATERIALS_DIR
            DEFAULT_MATERIALS_DIR = Path(materials_dir)

    def simulate(
        self,
        materials: list[str],
        thicknesses_nm: list[float],
        substrate: str = "SiO2",
        *,
        config: SimulationConfig | None = None,
        wavelengths_nm: tuple[float, float, int] | None = None,
        angle_deg: float | None = None,
        polarization: str | None = None,
        solver: str | None = None,
        substrate_mode: str | None = None,
        n_incident: float | None = None,
    ) -> SimulationResult:
        """Run TMM simulation. Any kwarg overrides config.

        materials:  film layers from top (incident side) to bottom (substrate side)
        thicknesses_nm: corresponding thicknesses in nanometers
        substrate:  substrate material name (must be in nk database)
        """
        # Never mutate a caller-owned reusable configuration when keyword
        # overrides are supplied.
        cfg = replace(config) if config is not None else SimulationConfig()

        # Apply kwarg overrides
        if wavelengths_nm:
            cfg.wl_start_nm, cfg.wl_end_nm, cfg.wl_points = wavelengths_nm
        if angle_deg is not None:
            cfg.angle_deg = float(angle_deg)
        if polarization is not None:
            cfg.polarization = str(polarization)
        if solver is not None:
            cfg.solver = str(solver)
        if substrate_mode is not None:
            cfg.substrate_mode = str(substrate_mode)
        if n_incident is not None:
            cfg.n_incident = float(n_incident)

        if cfg.substrate_mode == "finite":
            raise ValueError(
                "The legacy simulate() API cannot describe a finite substrate because it has no "
                "substrate thickness or backing medium. Use StackSpec + TMMWorkbench and model the "
                "substrate as a finite LayerSpec instead."
            )

        n_mat = len(materials)
        if n_mat != len(thicknesses_nm):
            raise ValueError(
                f"materials ({n_mat}) and thicknesses_nm ({len(thicknesses_nm)}) must match"
            )

        wl_nm = cfg.wavelengths_nm()
        wl_um = wl_nm * 1e-3
        n_points = len(wl_nm)

        # Build n_list: films + substrate (thickness=0 for semi-infinite)
        n_list = []
        for mat in materials:
            nk = np.asarray(self.nk_db.get_nk(mat, wl_um), dtype=np.complex128)
            n_list.append(nk.reshape(n_points))
        nk_sub = np.asarray(self.nk_db.get_nk(substrate, wl_um), dtype=np.complex128)
        n_list.append(nk_sub.reshape(n_points))

        # Thicknesses in meters: films + 0 for substrate
        d_list = [float(t) * 1e-9 for t in thicknesses_nm] + [0.0]

        tmm = TMM(TMMConfig(
            wavelength_unit="um",
            n_incident=complex(cfg.n_incident, 0),
            treat_last_layer_as_substrate=(cfg.substrate_mode == "semi_infinite"),
            clip_R=cfg.clip_R,
            clip_T=cfg.clip_T,
        ))

        def _solve_one(pol: str):
            if cfg.solver == "smatrix":
                return tmm.rt_spectrum(
                    n_list, d_list, wl_um, theta=cfg.angle_deg, pol=pol,
                    theta_unit="deg",
                )
            if cfg.solver == "cm":
                return tmm.rt_spectrum_cm(
                    n_list, d_list, wl_um, theta=cfg.angle_deg, pol=pol,
                    theta_unit="deg",
                )
            if cfg.solver == "abeles":
                reflectance = tmm.reflectance_spectrum(
                    n_list, d_list, wl_um, theta=cfg.angle_deg, pol=pol,
                    theta_unit="deg", return_raw=True,
                )
                # Abeles recursion is an R-only algorithm.  Preserve the legacy
                # R/T result contract by obtaining T from the stable S-matrix.
                _, transmittance = tmm.rt_spectrum(
                    n_list, d_list, wl_um, theta=cfg.angle_deg, pol=pol,
                    theta_unit="deg",
                )
                return reflectance, transmittance
            raise ValueError(f"Unknown solver: {cfg.solver}")

        if cfg.polarization == "unpolarized":
            Rs, Ts = _solve_one("s")
            Rp, Tp = _solve_one("p")
            R = 0.5 * (Rs + Rp)
            T = 0.5 * (Ts + Tp)
        elif cfg.polarization == "s":
            R, T = _solve_one("s")
        elif cfg.polarization == "p":
            R, T = _solve_one("p")
        else:
            raise ValueError(f"Unknown polarization: {cfg.polarization}")

        R = np.asarray(R, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64)
        if cfg.clip_R:
            R = np.clip(R, 0.0, 1.0)
        if cfg.clip_T:
            T = np.clip(T, 0.0, 1.0)
        # Do not clip absorptance: a negative value is a useful signal that a
        # solver/material convention is wrong and must not be hidden.
        A = 1.0 - R - T if cfg.compute_absorption else None

        return SimulationResult(
            wavelengths_nm=wl_nm,
            R=R,
            T=T,
            A=A,
            materials=list(materials) + [substrate],
            thicknesses_nm=list(thicknesses_nm),
            config={
                "wl_range_nm": [cfg.wl_start_nm, cfg.wl_end_nm],
                "wl_points": cfg.wl_points,
                "angle_deg": cfg.angle_deg,
                "polarization": cfg.polarization,
                "solver": cfg.solver,
                "substrate_mode": cfg.substrate_mode,
                "n_incident": cfg.n_incident,
            },
        )

    def lookup_nk(self, material: str, wavelengths_nm: tuple[float, float, int] = (380, 780, 401)) -> tuple[np.ndarray, np.ndarray]:
        """Get n and k arrays for a material. Returns (n, k)."""
        wl_um = np.linspace(wavelengths_nm[0], wavelengths_nm[1], wavelengths_nm[2], dtype=np.float64) * 1e-3
        nk = self.nk_db.get_nk(material, wl_um)
        return np.real(nk), np.imag(nk)

    def list_materials(self) -> list[str]:
        return self.nk_db.list_available()

    def list_available_materials(self) -> list[str]:
        """Compatibility alias for older examples."""
        return self.list_materials()


# Stable v2 workbench API. Differentiable modules remain explicit imports so
# importing tmm_engine never requires PyTorch.
from .acceptance import AcceptanceSettings, CertifiedSimulation, certify_simulation  # noqa: E402
from .agent_bench import (  # noqa: E402
    BenchmarkAssertion,
    BenchmarkCase,
    load_benchmark_cases,
    run_offline_benchmark,
)
from .agent_harness import (  # noqa: E402
    AgentTrajectory,
    TrajectoryStep,
    build_exposure,
    run_agent_ab,
    score_trajectories,
)
from .analysis import (  # noqa: E402
    SpectralBand,
    bloch_trace_bilayer_normal_incidence,
    find_threshold_bands,
    hemispherical_average,
    phase_dispersion_from_amplitude,
    spectrum_similarity,
    thickness_tolerance_monte_carlo,
)
from .capabilities import (  # noqa: E402
    CapabilityAssessment,
    FailureAction,
    FailureCode,
    FailureRecord,
    PhysicsEngineError,
    assess_tmm_capability,
    enrich_failure_actions,
    failure_from_exception,
)
from .convergence import (  # noqa: E402
    SpectralConvergenceOutcome,
    SpectralConvergenceSettings,
    audit_spectral_convergence,
)
from .designs import (  # noqa: E402
    chirped_stack,
    defect_cavity_stack,
    periodic_stack,
    with_finite_substrate,
)
from .execution import ExecutionSettings, execute_task  # noqa: E402
from .material_registry import (  # noqa: E402
    MaterialAmbiguityError,
    MaterialNotFoundError,
    MaterialRangeError,
    MaterialRef,
    MaterialRegistry,
)
from .preflight import preflight_path, preflight_task  # noqa: E402
from .protocol import (  # noqa: E402
    COMPACT_MAX_BYTES,
    COMPACT_TARGET_BYTES,
    DEFAULT_RESPONSE_DETAIL,
    RESPONSE_DETAILS,
    RESPONSE_SCHEMA_VERSION,
    CapabilityManifest,
    ContextBudgetError,
    OptimizationTaskContract,
    PreflightReport,
    ResponseDetail,
    ResponseMetadata,
    ResponseProfile,
    RunResultEnvelope,
    SimulationTaskContract,
    compact_response,
    describe_capabilities,
    export_schema,
    guard_context_budget,
    is_projected_response,
    normalize_response_detail,
    project_response,
    response_profile,
    validate_artifact_references,
    validate_projected_response,
)
from .schemas import (  # noqa: E402
    IlluminationSpec,
    LayerSpec,
    MediumSpec,
    OptimizationTask,
    OptimizerSpec,
    PhysicsRequirements,
    SimulationTask,
    SpectralGrid,
    SpectralTarget,
    StackSpec,
)
from .workbench import ForwardSimulationResult, TMMWorkbench  # noqa: E402

__all__ = [
    "__version__",
    "ForwardSimulationResult",
    "AcceptanceSettings",
    "AgentTrajectory",
    "BenchmarkAssertion",
    "BenchmarkCase",
    "CertifiedSimulation",
    "CapabilityAssessment",
    "CapabilityManifest",
    "COMPACT_TARGET_BYTES",
    "COMPACT_MAX_BYTES",
    "ContextBudgetError",
    "DEFAULT_RESPONSE_DETAIL",
    "ExecutionSettings",
    "FailureAction",
    "FailureCode",
    "FailureRecord",
    "IlluminationSpec",
    "LayerSpec",
    "MaterialAmbiguityError",
    "MaterialNotFoundError",
    "MaterialRangeError",
    "MaterialRef",
    "MaterialRegistry",
    "MediumSpec",
    "OptimizationTask",
    "OptimizationTaskContract",
    "OptimizerSpec",
    "PhysicsEngineError",
    "PhysicsRequirements",
    "PreflightReport",
    "RESPONSE_DETAILS",
    "RESPONSE_SCHEMA_VERSION",
    "ResponseMetadata",
    "ResponseProfile",
    "ResponseDetail",
    "SimulationConfig",
    "SimulationResult",
    "SimulationTask",
    "SimulationTaskContract",
    "SpectralBand",
    "SpectralGrid",
    "SpectralTarget",
    "StackSpec",
    "TMMEngine",
    "TMMWorkbench",
    "TrajectoryStep",
    "RunResultEnvelope",
    "SpectralConvergenceOutcome",
    "SpectralConvergenceSettings",
    "assess_tmm_capability",
    "audit_spectral_convergence",
    "bloch_trace_bilayer_normal_incidence",
    "build_exposure",
    "chirped_stack",
    "defect_cavity_stack",
    "find_threshold_bands",
    "hemispherical_average",
    "phase_dispersion_from_amplitude",
    "periodic_stack",
    "load_benchmark_cases",
    "run_agent_ab",
    "run_offline_benchmark",
    "score_trajectories",
    "spectrum_similarity",
    "thickness_tolerance_monte_carlo",
    "with_finite_substrate",
    "certify_simulation",
    "compact_response",
    "describe_capabilities",
    "enrich_failure_actions",
    "execute_task",
    "export_schema",
    "failure_from_exception",
    "guard_context_budget",
    "is_projected_response",
    "normalize_response_detail",
    "preflight_path",
    "preflight_task",
    "project_response",
    "response_profile",
    "validate_artifact_references",
    "validate_projected_response",
]
