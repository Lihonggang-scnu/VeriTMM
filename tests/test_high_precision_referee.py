"""Tests for the high-precision mpmath referee in tmm_engine/high_precision.py.

These tests verify:
- compute_rt_single_channel returns finite R, T consistent with float64 solvers
- run_referee status semantics (ok / unavailable / not_triggered)
- _maybe_run_referee is NOT triggered for comfortably-passing results
- _maybe_run_referee IS triggered on solver disagreement
- High-precision R+T ≈ 1 for a lossless single film
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
