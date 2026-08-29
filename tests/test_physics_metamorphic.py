"""Physics metamorphic invariant tests for the VeriTMM TMM solver.

These tests verify that the solver obeys physical identities that must hold
regardless of what the correct numerical answer is.  They are independent of
any golden reference and complement the acceptance-certificate checks.

Invariants covered:
- split-layer: 100 nm A == 50 nm A + 50 nm A
- zero-thickness: A / 0 nm B / C == A / C
- lossless energy: R + T == 1 for non-absorbing stacks
- passivity: A >= 0 everywhere
- normal-incidence s/p equivalence
- wavelength-grid refinement consistency
- Fresnel single-interface analytic check (n1=1.0 -> n2=1.5, normal incidence)
- reciprocity: swapping incident/exit gives same R
- passivity and cross-solver agreement on an absorbing stack, for every solver
  the engine declares (not just the default one)
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tmm_engine import (
    AcceptanceSettings,
    IlluminationSpec,
    LayerSpec,
    MaterialRegistry,
    MediumSpec,
    SimulationTask,
    SpectralGrid,
    StackSpec,
    TMMWorkbench,
)
from tmm_engine.protocol.capabilities import SUPPORTED_SOLVERS


def _workbench() -> TMMWorkbench:
    return TMMWorkbench(MaterialRegistry())


def _simulate(task: SimulationTask) -> dict:
    wb = _workbench()
    result = wb.simulate(task)
    return result.channels


def _rta(channels: dict, angle: float = 0.0, pol: str = "s") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = f"angle={float(angle):g}|pol={pol}"
    ch = channels[key]
    return (
        np.asarray(ch["R"], dtype=np.float64),
        np.asarray(ch["T"], dtype=np.float64),
        np.asarray(ch["A"], dtype=np.float64),
    )


def _simple_task(
    layers: list[LayerSpec],
    *,
    angles: tuple = (0.0,),
    pols: tuple = ("s",),
    points: int = 61,
) -> SimulationTask:
    return SimulationTask(
        stack=StackSpec(
            incident=MediumSpec(constant_n=1.0),
            layers=tuple(layers),
            exit=MediumSpec(constant_n=1.52),
            name="metamorphic",
        ),
        spectrum=SpectralGrid(start_nm=450.0, stop_nm=750.0, points=points),
        illumination=IlluminationSpec(
            angles_deg=angles,
            polarizations=pols,
        ),
        requested_outputs=("R", "T", "A"),
    )


LAYER_A = LayerSpec(None, 100.0, constant_n=1.45, constant_k=0.0)
LAYER_A_HALF = LayerSpec(None, 50.0, constant_n=1.45, constant_k=0.0)
LAYER_ABSORBING = LayerSpec(None, 80.0, constant_n=2.0, constant_k=0.5)
LAYER_ZERO = LayerSpec(None, 0.0, constant_n=1.45, constant_k=0.0)


class TestSplitLayerInvariance:
    """100 nm SiO2 == 50 nm SiO2 + 50 nm SiO2."""

    def test_split_layer_R(self) -> None:
        whole = _simulate(_simple_task([LAYER_A]))
        split = _simulate(_simple_task([LAYER_A_HALF, LAYER_A_HALF]))
        R1, _, _ = _rta(whole)
        R2, _, _ = _rta(split)
        np.testing.assert_allclose(R1, R2, atol=1e-10)

    def test_split_layer_T(self) -> None:
        whole = _simulate(_simple_task([LAYER_A]))
        split = _simulate(_simple_task([LAYER_A_HALF, LAYER_A_HALF]))
        _, T1, _ = _rta(whole)
        _, T2, _ = _rta(split)
        np.testing.assert_allclose(T1, T2, atol=1e-10)


class TestZeroThicknessInvariance:
    """A near-zero-thickness layer (1e-4 nm) must not meaningfully change R, T, A."""

    def test_zero_layer_R(self) -> None:
        base = _simulate(_simple_task([LAYER_A]))
        near_zero = LayerSpec(None, 1e-4, constant_n=1.45, constant_k=0.0)
        with_thin = _simulate(_simple_task([near_zero, LAYER_A]))
        R1, _, _ = _rta(base)
        R2, _, _ = _rta(with_thin)
        np.testing.assert_allclose(R1, R2, atol=1e-6)

    def test_zero_layer_T(self) -> None:
        base = _simulate(_simple_task([LAYER_A]))
        near_zero = LayerSpec(None, 1e-4, constant_n=1.45, constant_k=0.0)
        with_thin = _simulate(_simple_task([near_zero, LAYER_A]))
        _, T1, _ = _rta(base)
        _, T2, _ = _rta(with_thin)
        np.testing.assert_allclose(T1, T2, atol=1e-6)


class TestLosslessEnergyConservation:
    """R + T == 1 for non-absorbing stacks."""

    def test_lossless_normal(self) -> None:
        channels = _simulate(_simple_task([LAYER_A]))
        R, T, A = _rta(channels)
        np.testing.assert_allclose(R + T, 1.0, atol=1e-10)
        assert float(np.max(A)) < 1e-10

    def test_lossless_oblique_s(self) -> None:
        channels = _simulate(
            _simple_task([LAYER_A], angles=(30.0,), pols=("s",))
        )
        R, T, A = _rta(channels, angle=30.0, pol="s")
        np.testing.assert_allclose(R + T, 1.0, atol=1e-10)

    def test_lossless_oblique_p(self) -> None:
        channels = _simulate(
            _simple_task([LAYER_A], angles=(30.0,), pols=("p",))
        )
        R, T, A = _rta(channels, angle=30.0, pol="p")
        np.testing.assert_allclose(R + T, 1.0, atol=1e-10)


class TestPassivity:
    """A >= 0 and R, T in [0, 1] everywhere."""

    @pytest.mark.parametrize("layer", [LAYER_A, LAYER_ABSORBING])
    def test_A_non_negative(self, layer: LayerSpec) -> None:
        channels = _simulate(_simple_task([layer]))
        _, _, A = _rta(channels)
        assert float(np.min(A)) >= -1e-10

    def test_R_in_unit_interval(self) -> None:
        channels = _simulate(_simple_task([LAYER_ABSORBING]))
        R, _, _ = _rta(channels)
        assert float(np.min(R)) >= -1e-10
        assert float(np.max(R)) <= 1.0 + 1e-10

    def test_T_in_unit_interval(self) -> None:
        channels = _simulate(_simple_task([LAYER_ABSORBING]))
        _, T, _ = _rta(channels)
        assert float(np.min(T)) >= -1e-10
        assert float(np.max(T)) <= 1.0 + 1e-10


class TestNormalIncidenceSPEquivalence:
    """At 0 degrees, s and p polarization must give identical R, T, A."""

    def test_s_equals_p_normal_incidence(self) -> None:
        channels = _simulate(
            _simple_task([LAYER_A], angles=(0.0,), pols=("s", "p"))
        )
        Rs, Ts, As = _rta(channels, angle=0.0, pol="s")
        Rp, Tp, Ap = _rta(channels, angle=0.0, pol="p")
        np.testing.assert_allclose(Rs, Rp, atol=1e-10)
        np.testing.assert_allclose(Ts, Tp, atol=1e-10)
        np.testing.assert_allclose(As, Ap, atol=1e-10)

    def test_s_equals_p_for_absorbing_layer(self) -> None:
        channels = _simulate(
            _simple_task([LAYER_ABSORBING], angles=(0.0,), pols=("s", "p"))
        )
        Rs, _, _ = _rta(channels, angle=0.0, pol="s")
        Rp, _, _ = _rta(channels, angle=0.0, pol="p")
        np.testing.assert_allclose(Rs, Rp, atol=1e-10)


class TestWavelengthGridRefinementConsistency:
    """Coarse and fine grids should agree at shared wavelengths."""

    def test_coarse_fine_grid_agreement(self) -> None:
        coarse_task = SimulationTask(
            stack=StackSpec(
                incident=MediumSpec(constant_n=1.0),
                layers=(LAYER_A,),
                exit=MediumSpec(constant_n=1.52),
                name="metamorphic",
            ),
            spectrum=SpectralGrid(values_nm=(500.0, 600.0, 700.0)),
            illumination=IlluminationSpec(
                angles_deg=(0.0,), polarizations=("s",)
            ),
            requested_outputs=("R", "T", "A"),
        )
        fine_task = SimulationTask(
            stack=coarse_task.stack,
            spectrum=SpectralGrid(start_nm=450.0, stop_nm=750.0, points=121),
            illumination=coarse_task.illumination,
            requested_outputs=("R", "T", "A"),
        )
        wb = _workbench()
        coarse_result = wb.simulate(coarse_task)
        fine_result = wb.simulate(fine_task)

        coarse_R, _, _ = _rta(coarse_result.channels)
        fine_wl = fine_result.wavelengths_nm

        # Find fine-grid indices closest to coarse wavelengths
        for coarse_wl, cr_val in zip([500.0, 600.0, 700.0], coarse_R):
            idx = int(np.argmin(np.abs(fine_wl - coarse_wl)))
            fine_R_at_wl = fine_result.channels["angle=0|pol=s"]["R"][idx]
            assert abs(float(cr_val) - float(fine_R_at_wl)) < 1e-8, (
                f"Coarse and fine grids disagree at {coarse_wl} nm: "
                f"{float(cr_val):.6f} vs {float(fine_R_at_wl):.6f}"
            )


class TestFresnelSingleInterface:
    """Single-film normal-incidence Fresnel analytic check."""

    def test_near_zero_film_matches_fresnel_interface(self) -> None:
        n1, n2 = 1.0, 1.5
        r = (n1 - n2) / (n1 + n2)
        R_fresnel = r * r

        task = SimulationTask(
            stack=StackSpec(
                incident=MediumSpec(constant_n=n1),
                layers=(LayerSpec(None, 1e-3, constant_n=1.0),),  # near-zero air gap
                exit=MediumSpec(constant_n=n2),
                name="fresnel",
            ),
            spectrum=SpectralGrid(start_nm=549.0, stop_nm=551.0, points=3),
            illumination=IlluminationSpec(
                angles_deg=(0.0,), polarizations=("s",)
            ),
            requested_outputs=("R", "T", "A"),
        )
        wb = _workbench()
        result = wb.simulate(task)
        # Use the middle wavelength point (550 nm)
        R_sim = float(result.channels["angle=0|pol=s"]["R"][1])
        assert abs(R_sim - R_fresnel) < 1e-6, (
            f"Near-zero film: simulated R={R_sim:.8f} vs Fresnel R={R_fresnel:.8f}"
        )


class TestReciprocity:
    """Swapping incident and exit media gives the same reflectance at normal incidence."""

    def test_forward_backward_R(self) -> None:
        n_inc, n_exit = 1.0, 1.52
        forward_task = SimulationTask(
            stack=StackSpec(
                incident=MediumSpec(constant_n=n_inc),
                layers=(LAYER_A,),
                exit=MediumSpec(constant_n=n_exit),
                name="forward",
            ),
            spectrum=SpectralGrid(start_nm=450.0, stop_nm=750.0, points=61),
            illumination=IlluminationSpec(
                angles_deg=(0.0,), polarizations=("s",)
            ),
            requested_outputs=("R", "T", "A"),
        )
        backward_task = SimulationTask(
            stack=StackSpec(
                incident=MediumSpec(constant_n=n_exit),
                layers=(LayerSpec(None, 100.0, constant_n=1.45, constant_k=0.0),),
                exit=MediumSpec(constant_n=n_inc),
                name="backward",
            ),
            spectrum=forward_task.spectrum,
            illumination=forward_task.illumination,
            requested_outputs=("R", "T", "A"),
        )
        wb = _workbench()
        fwd = wb.simulate(forward_task)
        bwd = wb.simulate(backward_task)
        R_fwd, _, _ = _rta(fwd.channels)
        R_bwd, _, _ = _rta(bwd.channels)
        np.testing.assert_allclose(R_fwd, R_bwd, atol=1e-10)


class TestAbsorptionAttenuatesOnEverySolver:
    """Passivity and cross-solver agreement for a strongly absorbing stack.

    Every invariant above runs on the default solver, which is how a solver
    that modelled absorption as gain stayed invisible: the characteristic
    matrix carried its off-diagonal sine term on the branch opposite to the one
    its own cos(theta) selects, so an absorbing layer amplified instead of
    attenuating.  A took negative values and R + T left 1 by five orders of
    magnitude more than the acceptance gate allows, which rejected every
    candidate of a real research route and consumed its whole budget.

    Parameterising over the engine's own declared solver set is the part that
    generalises: a solver added later has to satisfy passivity here before it
    can reach a task, and no separate list needs updating.
    """

    @pytest.mark.parametrize("solver", SUPPORTED_SOLVERS)
    def test_absorbing_layer_is_passive(self, solver: str) -> None:
        channels = _simulate(
            replace(_simple_task([LAYER_ABSORBING]), solver=solver)
        )
        R, T, A = _rta(channels)
        assert np.all(A >= -1e-12), f"{solver} reports gain: min A = {A.min():.3e}"
        assert np.all(R + T <= 1.0 + 1e-12), (
            f"{solver} returns more power than entered: max R+T = "
            f"{(R + T).max():.9f}"
        )
        np.testing.assert_allclose(R + T + A, 1.0, atol=1e-12)

    @pytest.mark.parametrize("solver", [s for s in SUPPORTED_SOLVERS if s != "smatrix"])
    def test_absorbing_layer_agrees_with_the_default_solver(self, solver: str) -> None:
        # Tie the bound to the gate that actually adjudicates disagreement, so
        # loosening one cannot silently outpace the other.
        tolerance = AcceptanceSettings().cross_solver_tolerance
        base = _simple_task([LAYER_ABSORBING])
        R0, T0, A0 = _rta(_simulate(base))
        R1, T1, A1 = _rta(_simulate(replace(base, solver=solver)))
        for name, expected, actual in (("R", R0, R1), ("T", T0, T1), ("A", A0, A1)):
            deviation = float(np.max(np.abs(actual - expected)))
            assert deviation <= tolerance, (
                f"{solver} disagrees with smatrix on {name} by {deviation:.3e}, "
                f"above the {tolerance:.0e} cross-solver tolerance"
            )
