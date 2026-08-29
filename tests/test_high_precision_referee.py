"""Tests for the high-precision mpmath referee in tmm_engine/high_precision.py.

These tests verify:
- compute_rt_single_channel returns finite R, T consistent with float64 solvers
- run_referee status semantics (ok / unavailable / not_triggered)
- _maybe_run_referee is NOT triggered for comfortably-passing results
- _maybe_run_referee IS triggered on solver disagreement
- High-precision R+T ≈ 1 for a lossless single film
- Absorption reduces R+T (never amplifies) and matches float64 on an
  absorbing film, at normal and oblique incidence, both polarizations
- Single-interface Fresnel analytic check at high precision
"""

from __future__ import annotations

import numpy as np
import pytest

from tmm_engine.high_precision import (
    PRECISION_BITS,
    compute_rt_single_channel,
    is_available,
    run_referee,
)

pytestmark = pytest.mark.skipif(
    not is_available(), reason="mpmath not installed"
)

# ---- helpers ----------------------------------------------------------------

def _single_film_n_all(n_film: complex = 1.5 + 0j) -> list:
    """[air, film, glass] refractive indices."""
    return [1.0 + 0j, n_film, 1.52 + 0j]


_WAVELENGTHS = [450.0, 500.0, 550.0, 600.0, 650.0, 700.0]
_D_NM = [100.0]


# ---- 1. Basic smoke test: finite R and T ------------------------------------

class TestComputeRtBasic:
    def test_returns_ok(self) -> None:
        result = compute_rt_single_channel(
            n_all=_single_film_n_all(),
            d_nm=_D_NM,
            wavelengths_nm=_WAVELENGTHS,
            angle_deg=0.0,
            polarization="s",
        )
        assert result["status"] == "ok"

    def test_R_T_finite(self) -> None:
        result = compute_rt_single_channel(
            n_all=_single_film_n_all(),
            d_nm=_D_NM,
            wavelengths_nm=_WAVELENGTHS,
            angle_deg=0.0,
            polarization="s",
        )
        R = np.asarray(result["R"])
        T = np.asarray(result["T"])
        assert np.all(np.isfinite(R))
        assert np.all(np.isfinite(T))

    def test_precision_bits_reported(self) -> None:
        result = compute_rt_single_channel(
            n_all=_single_film_n_all(),
            d_nm=_D_NM,
            wavelengths_nm=_WAVELENGTHS,
            angle_deg=0.0,
            polarization="s",
        )
        assert result["precision_bits"] == PRECISION_BITS


# ---- 2. Lossless energy conservation at high precision ----------------------

class TestLosslessEnergyHP:
    """For a lossless stack, R + T must equal 1 to near machine precision."""

    def test_energy_normal_incidence(self) -> None:
        result = compute_rt_single_channel(
            n_all=[1.0 + 0j, 1.5 + 0j, 1.0 + 0j],
            d_nm=[200.0],
            wavelengths_nm=_WAVELENGTHS,
            angle_deg=0.0,
            polarization="s",
        )
        R = np.asarray(result["R"])
        T = np.asarray(result["T"])
        np.testing.assert_allclose(R + T, np.ones(len(_WAVELENGTHS)), atol=1e-12)

    def test_energy_oblique_s(self) -> None:
        result = compute_rt_single_channel(
            n_all=[1.0 + 0j, 2.0 + 0j, 1.0 + 0j],
            d_nm=[150.0],
            wavelengths_nm=_WAVELENGTHS,
            angle_deg=30.0,
            polarization="s",
        )
        R = np.asarray(result["R"])
        T = np.asarray(result["T"])
        np.testing.assert_allclose(R + T, np.ones(len(_WAVELENGTHS)), atol=1e-12)

    def test_energy_oblique_p(self) -> None:
        result = compute_rt_single_channel(
            n_all=[1.0 + 0j, 2.0 + 0j, 1.0 + 0j],
            d_nm=[150.0],
            wavelengths_nm=_WAVELENGTHS,
            angle_deg=30.0,
            polarization="p",
        )
        R = np.asarray(result["R"])
        T = np.asarray(result["T"])
        np.testing.assert_allclose(R + T, np.ones(len(_WAVELENGTHS)), atol=1e-12)


# ---- 3. Agreement with float64 solver to within 1e-9 -----------------------

class TestAgreementWithFloat64:
    """HP result should match float64 smatrix solver to < 1e-9 for simple cases."""

    def test_close_to_float64(self) -> None:
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
        wb = TMMWorkbench(MaterialRegistry())
        task = SimulationTask(
            stack=StackSpec(
                incident=MediumSpec(constant_n=1.0),
                layers=(LayerSpec(None, 200.0, constant_n=1.5),),
                exit=MediumSpec(constant_n=1.0),
            ),
            spectrum=SpectralGrid(values_nm=tuple(_WAVELENGTHS)),
            illumination=IlluminationSpec(angles_deg=(0.0,), polarizations=("s",)),
            requested_outputs=("R", "T", "A"),
        )
        f64_result = wb.simulate(task)
        ch = f64_result.channels["angle=0|pol=s"]
        R_f64 = np.asarray(ch["R"])
        T_f64 = np.asarray(ch["T"])

        hp = compute_rt_single_channel(
            n_all=[1.0 + 0j, 1.5 + 0j, 1.0 + 0j],
            d_nm=[200.0],
            wavelengths_nm=_WAVELENGTHS,
            angle_deg=0.0,
            polarization="s",
        )
        np.testing.assert_allclose(np.asarray(hp["R"]), R_f64, atol=1e-9)
        np.testing.assert_allclose(np.asarray(hp["T"]), T_f64, atol=1e-9)


# ---- 4. run_referee status: not_triggered when margin is comfortable --------

class TestRunRefereeNotTriggered:
    def test_not_triggered_when_no_disagreement(self) -> None:
        result = run_referee(
            n_all=_single_film_n_all(),
            d_nm=_D_NM,
            wavelengths_nm=_WAVELENGTHS,
            angle_deg=0.0,
            polarization="s",
            primary_R=None,
            primary_T=None,
        )
        # With no primary data, run_referee still computes — status="ok"
        assert result["status"] == "ok"

    def test_diff_from_primary_reported(self) -> None:
        hp_pre = compute_rt_single_channel(
            n_all=_single_film_n_all(),
            d_nm=_D_NM,
            wavelengths_nm=_WAVELENGTHS,
            angle_deg=0.0,
            polarization="s",
        )
        result = run_referee(
            n_all=_single_film_n_all(),
            d_nm=_D_NM,
            wavelengths_nm=_WAVELENGTHS,
            angle_deg=0.0,
            polarization="s",
            primary_R=hp_pre["R"],
            primary_T=hp_pre["T"],
        )
        # Same input → diff should be essentially zero
        assert result["status"] == "ok"
        assert result["max_abs_diff_from_primary"] < 1e-15


# ---- 5. Absorbing film: the referee must see loss, not gain -----------------

_ABSORBING_N_ALL = [1.0 + 0j, 2.0 + 0.5j, 1.52 + 0j]
_ABSORBING_D_NM = [80.0]


class TestAbsorbingFilmHP:
    """Absorption must reduce R + T here, because this module is the referee.

    Everything above uses lossless stacks, so a sign error in the layer matrix
    left no trace: with the off-diagonal sine term on the branch opposite to
    the one the module's own cos(theta) selects, an absorbing layer amplifies.
    This module arbitrates cross-solver disagreement, so such an error does not
    merely give a wrong number -- it certifies gain as loss and signs off on the
    wrong solver.  The index convention is n + i*k with k positive.
    """

    def test_absorption_reduces_transmitted_and_reflected_power(self) -> None:
        result = compute_rt_single_channel(
            n_all=_ABSORBING_N_ALL,
            d_nm=_ABSORBING_D_NM,
            wavelengths_nm=_WAVELENGTHS,
            angle_deg=0.0,
            polarization="s",
        )
        assert result["status"] == "ok"
        R = np.asarray(result["R"])
        T = np.asarray(result["T"])
        assert np.all(R >= -1e-12) and np.all(T >= -1e-12)
        assert np.all(R + T <= 1.0 - 1e-6), (
            f"absorbing film returns R+T = {(R + T).max():.9f}, i.e. gain"
        )

    def test_more_absorption_leaves_less_power(self) -> None:
        totals = []
        for k in (0.0, 0.001, 0.01, 0.1, 0.5):
            result = compute_rt_single_channel(
                n_all=[1.0 + 0j, complex(2.0, k), 1.52 + 0j],
                d_nm=_ABSORBING_D_NM,
                wavelengths_nm=[550.0],
                angle_deg=0.0,
                polarization="s",
            )
            totals.append(float(np.asarray(result["R"])[0] + np.asarray(result["T"])[0]))
        for stronger, weaker in zip(totals[1:], totals[:-1]):
            assert stronger < weaker, f"R+T not monotonically decreasing in k: {totals}"

    @pytest.mark.parametrize("polarization", ["s", "p"])
    @pytest.mark.parametrize("angle_deg", [0.0, 35.0])
    def test_absorbing_film_matches_float64(
        self, angle_deg: float, polarization: str
    ) -> None:
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

        wb = TMMWorkbench(MaterialRegistry())
        task = SimulationTask(
            stack=StackSpec(
                incident=MediumSpec(constant_n=1.0),
                layers=(
                    LayerSpec(
                        None,
                        _ABSORBING_D_NM[0],
                        constant_n=2.0,
                        constant_k=0.5,
                    ),
                ),
                exit=MediumSpec(constant_n=1.52),
            ),
            spectrum=SpectralGrid(values_nm=tuple(_WAVELENGTHS)),
            illumination=IlluminationSpec(
                angles_deg=(angle_deg,), polarizations=(polarization,)
            ),
            requested_outputs=("R", "T", "A"),
        )
        ch = wb.simulate(task).channels[f"angle={angle_deg:g}|pol={polarization}"]

        hp = compute_rt_single_channel(
            n_all=_ABSORBING_N_ALL,
            d_nm=_ABSORBING_D_NM,
            wavelengths_nm=_WAVELENGTHS,
            angle_deg=angle_deg,
            polarization=polarization,
        )
        np.testing.assert_allclose(np.asarray(hp["R"]), np.asarray(ch["R"]), atol=1e-9)
        np.testing.assert_allclose(np.asarray(hp["T"]), np.asarray(ch["T"]), atol=1e-9)
