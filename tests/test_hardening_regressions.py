"""Regression tests for the four hardening items fixed in PR #3.

Covers:
1. n_by_wavelength: dispersive indices participate per-wavelength (not first λ only)
2. closer_solver: correct when primary/secondary distances differ
3. offending_channel: referee targets the true worst-case channel in multi-angle tasks
4. energy trigger: referee targets energy worst-case channel, not cross-solver channel
5. mpmath workprec: global precision unchanged after compute_rt_single_channel
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WAVELENGTHS = [450.0, 500.0, 550.0, 600.0, 650.0, 700.0]
_N = len(_WAVELENGTHS)
_D_NM = [150.0]


def _make_task(
    *,
    angles=(0.0,),
    pols=("s",),
    n_inc=1.0,
    n_film=1.5,
    n_exit=1.0,
    d_nm=150.0,
    points=11,
):
    from tmm_engine import (
        IlluminationSpec,
        LayerSpec,
        MediumSpec,
        SimulationTask,
        SpectralGrid,
        StackSpec,
    )

    return SimulationTask(
        stack=StackSpec(
            incident=MediumSpec(constant_n=n_inc),
            layers=(LayerSpec(None, d_nm, constant_n=n_film),),
            exit=MediumSpec(constant_n=n_exit),
        ),
        spectrum=SpectralGrid(start_nm=400.0, stop_nm=800.0, points=points),
        illumination=IlluminationSpec(angles_deg=angles, polarizations=tuple(pols)),
        requested_outputs=("R", "T", "A"),
    )


def _wb():
    from tmm_engine import MaterialRegistry, TMMWorkbench

    return TMMWorkbench(MaterialRegistry())


# ===========================================================================
# 1. n_by_wavelength: dispersive path is active, not first-λ fallback
# ===========================================================================

hp_available = pytest.importorskip if False else pytest.mark.skipif


@pytest.fixture(scope="module")
def _hp_skip():
    from tmm_engine.high_precision import is_available

    return not is_available()


class TestDispersiveNByWavelength:
    """n_by_wavelength passes the correct per-λ index to every wavelength point."""

    @pytest.fixture(autouse=True)
    def _skip_no_mpmath(self, _hp_skip):
        if _hp_skip:
            pytest.skip("mpmath not installed")

    def test_dispersive_differs_from_first_wavelength_scalar(self) -> None:
        """Result with varying n must differ from result computed with n[0] only."""
        from tmm_engine.high_precision import compute_rt_single_channel

        # Build a dispersive n array: strongly varying across wavelengths
        n_inc = [1.0] * _N
        n_exit = [1.0] * _N
        # Film n varies from 1.3 to 2.2 across the wavelength range
        n_film = [1.3 + 0.9 * i / (_N - 1) for i in range(_N)]

        dispersive = compute_rt_single_channel(
            n_by_wavelength=[n_inc, n_film, n_exit],
            d_nm=_D_NM,
            wavelengths_nm=_WAVELENGTHS,
            angle_deg=0.0,
            polarization="s",
        )
        # Scalar path: use only n_film[0] for all wavelengths
        scalar = compute_rt_single_channel(
            n_all=[1.0, n_film[0], 1.0],
            d_nm=_D_NM,
            wavelengths_nm=_WAVELENGTHS,
            angle_deg=0.0,
            polarization="s",
        )

        assert dispersive["status"] == "ok"
        assert scalar["status"] == "ok"
        # The dispersive result must differ from the first-wavelength scalar result
        # at later wavelengths where n has changed significantly
        R_disp = np.asarray(dispersive["R"])
        R_scal = np.asarray(scalar["R"])
        assert float(np.max(np.abs(R_disp - R_scal))) > 1e-4, (
            "dispersive and scalar results are suspiciously identical — "
            "n_by_wavelength may not be indexing per wavelength"
        )

    def test_n_by_wavelength_first_point_matches_scalar(self) -> None:
        """At wavelength index 0, dispersive and scalar results must agree."""
        from tmm_engine.high_precision import compute_rt_single_channel

        n_film_val = 1.7
        dispersive = compute_rt_single_channel(
            n_by_wavelength=[[1.0] * _N, [n_film_val] * _N, [1.0] * _N],
            d_nm=_D_NM,
            wavelengths_nm=_WAVELENGTHS,
            angle_deg=0.0,
            polarization="s",
        )
        scalar = compute_rt_single_channel(
            n_all=[1.0, n_film_val, 1.0],
            d_nm=_D_NM,
            wavelengths_nm=_WAVELENGTHS,
            angle_deg=0.0,
            polarization="s",
        )
        np.testing.assert_allclose(
            np.asarray(dispersive["R"]), np.asarray(scalar["R"]), atol=1e-12
        )


# ===========================================================================
# 2. closer_solver: correct verdict when distances differ
# ===========================================================================

class TestCloserSolver:
    """run_referee returns the correct closer_solver label."""

    @pytest.fixture(autouse=True)
    def _skip_no_mpmath(self, _hp_skip):
        if _hp_skip:
            pytest.skip("mpmath not installed")

    def test_primary_closer(self) -> None:
        from tmm_engine.high_precision import compute_rt_single_channel, run_referee

        hp = compute_rt_single_channel(
            n_all=[1.0, 1.5, 1.0],
            d_nm=_D_NM,
            wavelengths_nm=_WAVELENGTHS,
            angle_deg=0.0,
            polarization="s",
        )
        hp_R = np.asarray(hp["R"])
        hp_T = np.asarray(hp["T"])

        # primary is very close (add tiny noise)
        primary_R = (hp_R + 1e-12).tolist()
        primary_T = (hp_T - 1e-12).tolist()
        # secondary is significantly further
        secondary_R = (hp_R + 5e-4).tolist()
        secondary_T = (hp_T - 5e-4).tolist()

        result = run_referee(
            n_all=[1.0, 1.5, 1.0],
            d_nm=_D_NM,
            wavelengths_nm=_WAVELENGTHS,
            angle_deg=0.0,
            polarization="s",
            primary_R=primary_R,
            primary_T=primary_T,
            secondary_R=secondary_R,
            secondary_T=secondary_T,
        )
        assert result["status"] == "ok"
        assert result["closer_solver"] == "primary"

    def test_secondary_closer(self) -> None:
        from tmm_engine.high_precision import compute_rt_single_channel, run_referee

        hp = compute_rt_single_channel(
            n_all=[1.0, 1.5, 1.0],
            d_nm=_D_NM,
            wavelengths_nm=_WAVELENGTHS,
            angle_deg=0.0,
            polarization="s",
        )
        hp_R = np.asarray(hp["R"])
        hp_T = np.asarray(hp["T"])

        # primary is far
        primary_R = (hp_R + 5e-4).tolist()
        primary_T = (hp_T - 5e-4).tolist()
        # secondary is very close
        secondary_R = (hp_R + 1e-12).tolist()
        secondary_T = (hp_T - 1e-12).tolist()

        result = run_referee(
            n_all=[1.0, 1.5, 1.0],
            d_nm=_D_NM,
            wavelengths_nm=_WAVELENGTHS,
            angle_deg=0.0,
            polarization="s",
            primary_R=primary_R,
            primary_T=primary_T,
            secondary_R=secondary_R,
            secondary_T=secondary_T,
        )
        assert result["status"] == "ok"
        assert result["closer_solver"] == "secondary"

    def test_none_when_secondary_missing(self) -> None:
        from tmm_engine.high_precision import run_referee

        result = run_referee(
            n_all=[1.0, 1.5, 1.0],
            d_nm=_D_NM,
            wavelengths_nm=_WAVELENGTHS,
            angle_deg=0.0,
            polarization="s",
            primary_R=[0.1] * _N,
            primary_T=[0.9] * _N,
            secondary_R=None,
            secondary_T=None,
        )
        assert result["closer_solver"] is None


# ===========================================================================
# 3. offending_channel: multi-angle tasks → referee hits the right channel
# ===========================================================================

class TestOffendingChannelMultiAngle:
    """For multi-angle tasks the referee must report the channel with max diff."""

    def test_certificate_offending_channel_matches_referee_channel(self) -> None:
        """In a two-angle cert the referee channel == cross_solver offending_channel."""
        from tmm_engine import (
            IlluminationSpec,
            LayerSpec,
            MediumSpec,
            SimulationTask,
            SpectralGrid,
            StackSpec,
        )
        from tmm_engine.acceptance import AcceptanceSettings, certify_simulation

        task = SimulationTask(
            stack=StackSpec(
                incident=MediumSpec(constant_n=1.0),
                layers=(LayerSpec(None, 200.0, constant_n=2.5),),
                exit=MediumSpec(constant_n=1.0),
            ),
            spectrum=SpectralGrid(start_nm=400.0, stop_nm=700.0, points=21),
            illumination=IlluminationSpec(
                angles_deg=(0.0, 45.0),
                polarizations=("s", "p"),
            ),
            requested_outputs=("R", "T", "A"),
        )
        settings = AcceptanceSettings(require_spectral_convergence=False)
        cs = certify_simulation(_wb(), task, settings)

        cross = cs.certificate.get("independent_solver_check", {})
        referee = cs.certificate.get("high_precision_referee", {})

        offending = cross.get("offending_channel")
        if offending is None or referee.get("status") in ("not_triggered", "unavailable"):
            pytest.skip("referee not triggered or cross_solver unavailable")

        assert referee["channel"] == offending, (
            f"referee channel {referee['channel']!r} != "
            f"cross-solver offending {offending!r}"
        )


# ===========================================================================
# 4. energy trigger: referee targets energy worst-case channel/location
# ===========================================================================

class TestEnergyTriggerChannel:
    """When tightest_margin is energy_conservation the referee uses the energy channel."""

    def test_audit_exposes_energy_worst_case_fields(self) -> None:
        """physics_audit must contain energy worst-case location fields."""
        task = _make_task(angles=(0.0, 30.0), pols=("s", "p"), points=21)
        result = _wb().simulate(task)
        audit = result.audit
        assert "energy_worst_case_channel" in audit
        assert "energy_worst_case_wavelength_nm" in audit
        assert "energy_worst_case_wavelength_idx" in audit

    def test_tightest_margin_energy_has_worst_case_channel(self) -> None:
        """_compute_tightest_margin attaches worst_case_channel for energy check."""
        from tmm_engine.acceptance import AcceptanceSettings, _compute_tightest_margin

        # Fabricate a physics_audit with a known energy worst-case
        fake_audit = {
            "energy_conservation_max_abs_error": 8e-8,   # close to tolerance=1e-7
            "energy_worst_case_channel": "angle=45|pol=p",
            "energy_worst_case_wavelength_nm": 532.0,
            "energy_worst_case_wavelength_idx": 5,
        }
        # Cross-solver comfortably passing — energy is tightest
        fake_cross = {
            "status": "passed",
            "maximum_absolute_difference": 1e-10,
            "offending_channel": "angle=0|pol=s",  # different channel
            "offending_observable": "R",
        }
        settings = AcceptanceSettings(energy_tolerance=1e-7, cross_solver_tolerance=1e-7)
        margin = _compute_tightest_margin(fake_audit, fake_cross, settings)

        assert margin is not None
        assert margin["check"] == "energy_conservation"
        assert margin["worst_case_channel"] == "angle=45|pol=p"
        assert margin["worst_case_wavelength_nm"] == pytest.approx(532.0)

    def test_cross_solver_margin_has_offending_channel(self) -> None:
        """_compute_tightest_margin attaches cross_solver offending_channel when that check wins."""
        from tmm_engine.acceptance import AcceptanceSettings, _compute_tightest_margin

        fake_audit = {
            "energy_conservation_max_abs_error": 1e-10,  # comfortably under tolerance
            "energy_worst_case_channel": "angle=0|pol=s",
            "energy_worst_case_wavelength_nm": 500.0,
            "energy_worst_case_wavelength_idx": 2,
        }
        fake_cross = {
            "status": "passed",
            "maximum_absolute_difference": 8e-8,  # close to cross_solver_tolerance
            "offending_channel": "angle=45|pol=p",
            "offending_observable": "T",
        }
        settings = AcceptanceSettings(energy_tolerance=1e-7, cross_solver_tolerance=1e-7)
        margin = _compute_tightest_margin(fake_audit, fake_cross, settings)

        assert margin is not None
        assert margin["check"] == "cross_solver_agreement"
        assert margin["worst_case_channel"] == "angle=45|pol=p"
        assert margin.get("worst_case_observable") == "T"


# ===========================================================================
# 5. mpmath workprec: global precision is not permanently mutated
# ===========================================================================

class TestMpmathPrecisionIsolation:
    """compute_rt_single_channel must not change the global mpmath precision."""

    @pytest.fixture(autouse=True)
    def _skip_no_mpmath(self, _hp_skip):
        if _hp_skip:
            pytest.skip("mpmath not installed")

    def test_global_precision_unchanged_after_call(self) -> None:
        import mpmath

        from tmm_engine.high_precision import PRECISION_BITS, compute_rt_single_channel

        prec_before = mpmath.mp.prec
        assert prec_before != PRECISION_BITS, (
            "test precondition: global prec must differ from PRECISION_BITS to be meaningful"
        )

        compute_rt_single_channel(
            n_all=[1.0, 1.5, 1.0],
            d_nm=_D_NM,
            wavelengths_nm=_WAVELENGTHS,
            angle_deg=0.0,
            polarization="s",
        )

        assert mpmath.mp.prec == prec_before, (
            "compute_rt_single_channel permanently mutated mpmath.mp.prec"
        )

    def test_custom_precision_bits_does_not_leak(self) -> None:
        import mpmath

        from tmm_engine.high_precision import compute_rt_single_channel

        prec_before = mpmath.mp.prec
        compute_rt_single_channel(
            n_all=[1.0, 1.5, 1.0],
            d_nm=_D_NM,
            wavelengths_nm=_WAVELENGTHS,
            angle_deg=0.0,
            polarization="s",
            precision_bits=256,
        )
        assert mpmath.mp.prec == prec_before
