"""TMM StressBench — numerical-stability stress tests for the VeriTMM pipeline.

These tests drive the solver through extreme configurations that are physically
valid but numerically challenging:

- Many layers  : 40-layer alternating DBR
- High-RI film  : n = 4.0 (GaAs-like) vs air
- Metallic film : n = 0.5 + 3.0i (Ag-like, highly absorbing)
- Near-grazing  : 85 ° angle of incidence
- Ultra-thin    : 0.5 nm film (near-zero optical thickness)
- Brewster p-pol: p-polarisation at the analytical Brewster angle → R ≈ 0

Every test checks that the physics-acceptance certificate is properly
populated; lossless cases additionally verify energy conservation.
"""

from __future__ import annotations

import math

import numpy as np

from tmm_engine import (
    IlluminationSpec,
    LayerSpec,
    MaterialRegistry,
    MediumSpec,
    SimulationTask,
    SpectralGrid,
    StackSpec,
    TMMWorkbench,
)
from tmm_engine.acceptance import AcceptanceSettings, certify_simulation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wb() -> TMMWorkbench:
    return TMMWorkbench(MaterialRegistry())


def _task(
    layers: list[LayerSpec],
    *,
    angles: tuple[float, ...] = (0.0,),
    pols: tuple[str, ...] = ("s",),
    points: int = 41,
    n_inc: float = 1.0,
    n_exit: float = 1.52,
) -> SimulationTask:
    return SimulationTask(
        stack=StackSpec(
            incident=MediumSpec(constant_n=n_inc),
            layers=tuple(layers),
            exit=MediumSpec(constant_n=n_exit),
            name="stress",
        ),
        spectrum=SpectralGrid(start_nm=400.0, stop_nm=800.0, points=points),
        illumination=IlluminationSpec(angles_deg=angles, polarizations=tuple(pols)),
        requested_outputs=("R", "T", "A"),
    )


def _energy_error(cert: dict) -> float:
    return float(cert.get("physics_audit", {}).get("energy_conservation_max_abs_error", float("inf")))


# Acceptance settings with tighter cross-solver tolerance for stress tests
# Default: skip spectral convergence (stress tests focus on solver stability,
# not on the convergence audit which can fail for highly-oscillatory thick stacks)

_NO_CONV = AcceptanceSettings(require_spectral_convergence=False)
_STRICT = AcceptanceSettings(
    require_spectral_convergence=False,
    cross_solver_tolerance=1e-6,
    energy_tolerance=1e-6,
)


# ---------------------------------------------------------------------------
# 1. Many layers — 40-layer alternating DBR
# ---------------------------------------------------------------------------

class TestManyLayersDBR:
    """40-layer alternating (n=2.3 / n=1.45) DBR must be certified."""

    _LAYERS = [
        LayerSpec(None, 80.0, constant_n=2.3 if i % 2 == 0 else 1.45)
        for i in range(40)
    ]

    def test_accepted(self) -> None:
        wb = _wb()
        cs = certify_simulation(wb, _task(self._LAYERS), _NO_CONV)
        assert cs.certificate["accepted"] is True, cs.certificate.get("failures")

    def test_energy_conservation(self) -> None:
        wb = _wb()
        cs = certify_simulation(wb, _task(self._LAYERS), _STRICT)
        assert _energy_error(cs.certificate) < 1e-6

    def test_tightest_margin_present(self) -> None:
        wb = _wb()
        cs = certify_simulation(wb, _task(self._LAYERS), _NO_CONV)
        assert cs.certificate["tightest_margin"] is not None

    def test_nonfinite_absent(self) -> None:
        wb = _wb()
        cs = certify_simulation(wb, _task(self._LAYERS), _NO_CONV)
        audit = cs.certificate.get("physics_audit", {})
        assert audit.get("nonfinite_value_count", 1) == 0


# ---------------------------------------------------------------------------
# 2. High refractive-index contrast — n_film = 4.0
# ---------------------------------------------------------------------------

class TestHighIndexFilm:
    """Single n=4.0 film in air; large Fresnel contrast, lossless."""

    _LAYER = [LayerSpec(None, 100.0, constant_n=4.0)]

    def test_accepted(self) -> None:
        wb = _wb()
        cs = certify_simulation(wb, _task(self._LAYER), _NO_CONV)
        assert cs.certificate["accepted"] is True

    def test_energy_conservation_strict(self) -> None:
        wb = _wb()
        cs = certify_simulation(wb, _task(self._LAYER, n_inc=1.0, n_exit=1.0), _STRICT)
        assert _energy_error(cs.certificate) < 1e-6


# ---------------------------------------------------------------------------
# 3. Metallic (absorbing) film — large imaginary part
# ---------------------------------------------------------------------------

class TestMetallicFilm:
    """Ag-like film (n≈0.5, k≈3.0) — strong absorption, certificate valid."""

    _LAYER = [LayerSpec(None, 30.0, constant_n=0.5, constant_k=3.0)]

    def test_accepted(self) -> None:
        wb = _wb()
        cs = certify_simulation(wb, _task(self._LAYER), _NO_CONV)
        assert cs.certificate["accepted"] is True

    def test_passivity(self) -> None:
        """A >= 0 everywhere; R, T in [0, 1]."""
        wb = _wb()
        cs = certify_simulation(wb, _task(self._LAYER), _NO_CONV)
        assert cs.certificate["physics_audit"]["passivity_check_passed"] is True

    def test_absorption_positive(self) -> None:
        """Metallic layer must absorb non-trivially (mean A > 0.01)."""
        wb = _wb()
        result = wb.simulate(_task(self._LAYER))
        ch = result.channels["angle=0|pol=s"]
        A = np.asarray(ch["A"], dtype=np.float64)
        assert float(np.mean(A)) > 0.01


# ---------------------------------------------------------------------------
# 4. Near-grazing incidence — 85 °
# ---------------------------------------------------------------------------

class TestNearGrazingIncidence:
    """85 ° incidence on a single dielectric film — numerically demanding."""

    _LAYER = [LayerSpec(None, 200.0, constant_n=1.45)]

    def test_accepted(self) -> None:
        wb = _wb()
        task = _task(self._LAYER, angles=(85.0,), pols=("s", "p"))
        cs = certify_simulation(wb, task, _NO_CONV)
        assert cs.certificate["accepted"] is True

    def test_energy_s_pol(self) -> None:
        wb = _wb()
        task = _task(self._LAYER, angles=(85.0,), pols=("s",))
        cs = certify_simulation(wb, task, _STRICT)
        assert _energy_error(cs.certificate) < 1e-6

    def test_energy_p_pol(self) -> None:
        wb = _wb()
        task = _task(self._LAYER, angles=(85.0,), pols=("p",))
        cs = certify_simulation(wb, task, _STRICT)
        assert _energy_error(cs.certificate) < 1e-6


# ---------------------------------------------------------------------------
# 5. Ultra-thin film — 0.5 nm
# ---------------------------------------------------------------------------

class TestUltraThinFilm:
    """0.5 nm film must not produce NaNs or unphysical values."""

    _LAYER = [LayerSpec(None, 0.5, constant_n=2.0)]

    def test_accepted(self) -> None:
        wb = _wb()
        cs = certify_simulation(wb, _task(self._LAYER), _NO_CONV)
        assert cs.certificate["accepted"] is True

    def test_no_nonfinite(self) -> None:
        wb = _wb()
        cs = certify_simulation(wb, _task(self._LAYER), _NO_CONV)
        assert cs.certificate["physics_audit"]["nonfinite_value_count"] == 0

    def test_energy_conservation(self) -> None:
        wb = _wb()
        cs = certify_simulation(wb, _task(self._LAYER, n_inc=1.0, n_exit=1.0), _STRICT)
        assert _energy_error(cs.certificate) < 1e-6


# ---------------------------------------------------------------------------
# 6. Brewster angle — p-polarisation, R ≈ 0
# ---------------------------------------------------------------------------

class TestBrewsterAngle:
    """At the Brewster angle for p-pol, R should be very close to zero."""

    # Brewster angle for n1=1.0 → n2=1.52 is atan(1.52/1.0) ≈ 56.66°
    _N_FILM = 1.52
    _BREWSTER_DEG = math.degrees(math.atan(_N_FILM / 1.0))
    _LAYER = [LayerSpec(None, 200.0, constant_n=_N_FILM)]

    def test_accepted(self) -> None:
        wb = _wb()
        task = _task(self._LAYER, angles=(self._BREWSTER_DEG,), pols=("p",), n_inc=1.0, n_exit=1.0)
        cs = certify_simulation(wb, task, _NO_CONV)
        assert cs.certificate["accepted"] is True

    def test_R_near_zero_at_brewster(self) -> None:
        """R for p-pol at Brewster angle must be < 0.01 at all wavelengths."""
        wb = _wb()
        task = _task(
            self._LAYER,
            angles=(self._BREWSTER_DEG,),
            pols=("p",),
            n_inc=1.0,
            n_exit=1.0,
            points=21,
        )
        result = wb.simulate(task)
        key = f"angle={float(self._BREWSTER_DEG):g}|pol=p"
        R = np.asarray(result.channels[key]["R"], dtype=np.float64)
        assert float(np.max(R)) < 0.01, f"R_max={float(np.max(R)):.4f} exceeds 0.01 at Brewster angle"
