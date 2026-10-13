from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.requires_torch

from tmm_engine import (
    IlluminationSpec,
    LayerSpec,
    MaterialRegistry,
    MediumSpec,
    OptimizationTask,
    OptimizerSpec,
    SimulationTask,
    SpectralGrid,
    SpectralTarget,
    StackSpec,
)
from tmm_engine.optimization import DifferentiableThicknessOptimizer
from tmm_engine.schemas import RobustnessSpec


def _task(*, starts=1, method="adam", steps=12, robust=False, quantization=None):
    simulation = SimulationTask(
        stack=StackSpec(
            layers=(LayerSpec(None, 180.0, constant_n=1.23,
                              min_thickness_nm=40.0, max_thickness_nm=250.0),),
            incident=MediumSpec.air(),
            exit=MediumSpec(constant_n=1.5),
        ),
        spectrum=SpectralGrid(500.0, 600.0, 7),
        illumination=IlluminationSpec((0.0,), ("s",)),
    )
    return OptimizationTask(
        simulation=simulation,
        targets=(SpectralTarget("R", 0.0, 500.0, 600.0, polarization="s"),),
        optimizer=OptimizerSpec(method=method, max_steps=steps, starts=starts,
                                quantization_nm=quantization, seed=4),
        robustness=RobustnessSpec(samples_per_step=3, final_samples=8) if robust else None,
    )


def _instrument(optimizer, monkeypatch):
    batches = []
    original = optimizer._solver_class.forward

    def forward(solver, thicknesses, *args, **kwargs):
        batches.append(int(thicknesses.shape[0]))
        return original(solver, thicknesses, *args, **kwargs)

    monkeypatch.setattr(optimizer._solver_class, "forward", forward)
    return batches


@pytest.mark.parametrize(
    "options,cap",
    [({}, 1), ({"starts": 4}, 8), ({"robust": True}, 1),
     ({"robust": True, "starts": 2}, 15),
     ({"method": "adam_lbfgs", "steps": 1}, 5),
     ({"steps": 1, "quantization": 7.0}, 2),
     ({"robust": True, "starts": 2, "steps": 1, "quantization": 7.0}, 10)],
)
def test_budget_bounds_actual_design_dispatch_and_keeps_best(options, cap, monkeypatch):
    task = _task(**options)
    optimizer = DifferentiableThicknessOptimizer(MaterialRegistry())
    batches = _instrument(optimizer, monkeypatch)
    result = optimizer.optimize(task, maximum_forward_evaluations=cap)

    assert sum(batches) == result.evaluation_count <= cap
    assert result.audit["evaluation_count_unit"] == "design_rows_all_target_channels"
    assert result.audit["maximum_forward_evaluations"] == cap
    assert result.audit["budget_exhausted"]
    assert result.stop_reason == "maximum_forward_evaluations"
    assert result.status == "best_effort"
    assert np.isfinite(result.optimized_loss)
    assert result.candidate_designs
    json.dumps(result.to_dict(), allow_nan=False)
    if not options.get("robust"):
        assert result.optimized_loss <= result.initial_loss
    measured, _ = optimizer.evaluate(task, result.optimized_thicknesses_nm)
    assert measured == pytest.approx(result.optimized_loss)


def test_multichannel_budget_counts_complete_design_rows(monkeypatch):
    task = _task(starts=2)
    task = replace(task, simulation=replace(
        task.simulation, illumination=IlluminationSpec((0.0, 25.0), ("s", "p"))
    ), targets=task.targets + (
        replace(task.targets[0], angle_deg=25.0, polarization="p"),
    ))
    optimizer = DifferentiableThicknessOptimizer(MaterialRegistry())
    batches = _instrument(optimizer, monkeypatch)
    result = optimizer.optimize(task, maximum_forward_evaluations=7)

    assert result.evaluation_count == 7
    assert sum(batches) == 2 * result.evaluation_count
    assert batches[::2] == batches[1::2]


def test_budget_caps_lbfgs_closures_without_swallowing_exhaustion(monkeypatch):
    optimizer = DifferentiableThicknessOptimizer(MaterialRegistry())
    batches = _instrument(optimizer, monkeypatch)
    closures = []

    def step(_self, closure):
        for _ in range(50):
            closure()
            closures.append(True)

    monkeypatch.setattr(torch.optim.LBFGS, "step", step)
    result = optimizer.optimize(_task(method="adam_lbfgs", steps=1),
                                maximum_forward_evaluations=5)
    assert len(closures) == 3
    assert sum(batches) == result.evaluation_count == 5
    assert result.stop_reason == "maximum_forward_evaluations"
    assert result.audit["budget_exhausted"]


def test_budget_is_per_run_and_invalid_limits_dispatch_nothing(monkeypatch):
    optimizer = DifferentiableThicknessOptimizer(MaterialRegistry())
    batches = _instrument(optimizer, monkeypatch)
    for limit in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            optimizer.optimize(_task(), maximum_forward_evaluations=limit)
    assert not batches
    first = optimizer.optimize(_task(), maximum_forward_evaluations=1)
    batches.clear()
    second = optimizer.optimize(_task(steps=2))
    assert first.evaluation_count == 1
    assert second.evaluation_count == sum(batches) > 1
    assert second.audit["maximum_forward_evaluations"] is None
    assert not second.audit["budget_exhausted"]
