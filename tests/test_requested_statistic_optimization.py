from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("torch")
pytestmark = pytest.mark.requires_torch

from tmm_engine import IlluminationSpec, LayerSpec, MaterialRegistry, MediumSpec, OptimizationTask, OptimizerSpec, SimulationTask, SpectralGrid, SpectralTarget, StackSpec, TMMWorkbench
from tmm_engine.optimization import DifferentiableThicknessOptimizer
from tmm_engine.spectral_reduction import reduce_spectral_mean


def task():
    return OptimizationTask(
        simulation=SimulationTask(
            stack=StackSpec(layers=(
                LayerSpec(None, 80., constant_n=1.46, min_thickness_nm=40., max_thickness_nm=160.),
                LayerSpec(None, 80., constant_n=2.4, min_thickness_nm=40., max_thickness_nm=160.)),
                incident=MediumSpec.air(), exit=MediumSpec(constant_n=1.52)),
            spectrum=SpectralGrid(450., 700., 51), illumination=IlluminationSpec((0.,), ("s",))),
        targets=(SpectralTarget("R", .03, 450., 700., polarization="s", constraint="at_most", mean_reduction="sample_mean"),),
        optimizer=OptimizerSpec(method="adam_lbfgs", starts=3, max_steps=64, seed=17))


@pytest.mark.parametrize("reduction", ["sample_mean", "wavelength_integral_mean"])
@pytest.mark.parametrize("constraint", ["at_most", "at_least", "match"])
def test_explicit_statistic_is_the_optimized_scalar(reduction, constraint):
    native = task()
    native = replace(native, simulation=replace(native.simulation, spectrum=SpectralGrid(values_nm=(450., 500., 700.))),
        targets=(replace(native.targets[0], constraint=constraint, mean_reduction=reduction),))
    result = TMMWorkbench(MaterialRegistry()).simulate(native.simulation)
    observed = reduce_spectral_mean(result.wavelengths_nm, result.channel(0., "s")["R"], reduction)
    expected = (observed - .03) ** 2 if constraint == "match" else 1 - observed if constraint == "at_least" else observed
    loss, _ = DifferentiableThicknessOptimizer(MaterialRegistry()).evaluate(native, [80., 80.])
    assert loss == pytest.approx(expected, abs=1e-12)
    if constraint == "at_most":
        assert loss != pytest.approx(np.mean(result.channel(0., "s")["R"] ** 2))


def test_budget_reserves_a_real_local_finish_without_reference_parameters():
    native = task()
    result = DifferentiableThicknessOptimizer(MaterialRegistry()).optimize(native, maximum_forward_evaluations=26)
    assert result.evaluation_count <= 26
    assert result.audit["local_finish_forward_evaluations"] > 0
    assert result.audit["local_finish_reserved_evaluations"] > 0
    assert result.optimized_loss < .03
    assert all(40. <= value <= 160. for value in result.optimized_thicknesses_nm)
    simulation = replace(native.simulation, solver="byrnes", stack=replace(native.simulation.stack,
        layers=tuple(replace(layer, thickness_nm=value) for layer, value in zip(native.simulation.stack.layers, result.optimized_thicknesses_nm))))
    reference = TMMWorkbench(MaterialRegistry()).simulate(simulation)
    assert result.optimized_loss == pytest.approx(float(reference.channel(0., "s")["R"].mean()), abs=1e-12)
