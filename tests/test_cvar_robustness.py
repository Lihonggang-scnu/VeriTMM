"""Tests for the CVaR tail-risk objective and independent final evaluation."""

from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import ValidationError

from tmm_engine import (
    IlluminationSpec,
    LayerSpec,
    MediumSpec,
    OptimizationTask,
    OptimizerSpec,
    SimulationTask,
    SpectralGrid,
    SpectralTarget,
    StackSpec,
)
from tmm_engine.protocol.models import RobustnessContract
from tmm_engine.robust_optimization import (
    conditional_value_at_risk,
    evaluate_robust_portfolio,
)
from tmm_engine.schemas import RobustnessSpec
from tmm_engine.uncertainty import final_robustness_seed, sample_normal_offsets


def test_cvar_analytic_case() -> None:
    losses = np.arange(1.0, 11.0)
    assert conditional_value_at_risk(losses, 0.2) == pytest.approx(9.5)


def test_cvar_is_non_increasing_as_alpha_increases() -> None:
    losses = np.arange(1.0, 11.0)
    values = [conditional_value_at_risk(losses, alpha) for alpha in (0.1, 0.2, 0.5, 0.9)]
    assert values == sorted(values, reverse=True)


def test_cvar_limits_are_explicit() -> None:
    losses = np.arange(1.0, 11.0)
    assert conditional_value_at_risk(losses, 0.999999) == pytest.approx(np.mean(losses))
    assert conditional_value_at_risk(losses, 0.000001) == pytest.approx(np.max(losses))


def test_cvar_replay_is_deterministic() -> None:
    first = sample_normal_offsets(seed=17, sample_count=32, layer_count=1, sigma_nm=2.0)
    second = sample_normal_offsets(seed=17, sample_count=32, layer_count=1, sigma_nm=2.0)
    assert conditional_value_at_risk(first[:, 0], 0.05) == conditional_value_at_risk(
        second[:, 0], 0.05
    )


class _Forward:
    wavelengths_nm = np.asarray([500.0, 550.0, 600.0])

    def __init__(self, thickness: float) -> None:
        self.audit = {
            "nonfinite_value_count": 0,
            "passivity_check_passed": True,
            "energy_conservation_max_abs_error": 0.0,
        }
        self._transmission = 0.9 + 0.001 * (float(thickness) - 100.0)

    def channel(self, _angle: float = 0.0, _polarization: str = "unpolarized"):
        return {"T": np.full(3, self._transmission, dtype=np.float64)}


class _Workbench:
    def simulate(self, task: SimulationTask) -> _Forward:
        return _Forward(float(task.stack.layers[0].thickness_nm))


def _cvar_task() -> OptimizationTask:
    simulation = SimulationTask(
        stack=StackSpec(
            layers=(LayerSpec(None, 100.0, constant_n=1.4, optimizable=True),),
            incident=MediumSpec.air(),
            exit=MediumSpec(constant_n=1.5),
        ),
        spectrum=SpectralGrid(start_nm=500.0, stop_nm=600.0, points=3),
        illumination=IlluminationSpec((0.0,), ("unpolarized",)),
    )
    return OptimizationTask(
        simulation=simulation,
        targets=(
            SpectralTarget(
                "T",
                0.9,
                500.0,
                600.0,
                constraint="at_least",
                tolerance=0.1,
            ),
        ),
        optimizer=OptimizerSpec(max_steps=1, starts=1),
        robustness=RobustnessSpec(
            objective="cvar",
            cvar_alpha=0.2,
            samples_per_step=2,
            final_samples=10,
            seed=7,
            thickness_sigma_nm=2.0,
        ),
    )


def _portfolio() -> dict[str, object]:
    return {
        "selected_roles": {"most_robust": "candidate"},
        "candidates": [
            {
                "candidate_id": "candidate",
                "independent_validation_status": "passed",
                "physics_status": "physically_valid",
                "certificate_id": "certificate",
                "metadata": {"objective_loss": 0.1, "thicknesses_nm": [100.0]},
            }
        ],
    }


def test_reported_cvar_matches_independent_final_ensemble() -> None:
    task = _cvar_task()
    updated, report = evaluate_robust_portfolio(task, _Workbench(), _portfolio())
    formal = updated["candidates"][0]["formal_robustness"]
    offsets = sample_normal_offsets(
        seed=final_robustness_seed(7),
        sample_count=10,
        layer_count=1,
        sigma_nm=2.0,
    )[:, 0]
    independent_losses = (1.0 - (0.9 + 0.001 * offsets)) ** 2

    assert formal["cvar"] == pytest.approx(
        conditional_value_at_risk(independent_losses, 0.2)
    )
    assert formal["final_seed"] == final_robustness_seed(7)
    assert formal["training_seed"] == 7
    assert formal["cvar_tail_sample_count"] == 2
    budget = formal["uncertainty_budget"]
    assert budget["sampling_components"][0]["source"] == "cvar_finite_sample"
    assert budget["sampling_components"][0]["degrees_of_freedom"] == 1
    assert report["uncertainty_budget"] == budget


def test_same_final_seed_replays_identical_cvar() -> None:
    task = _cvar_task()
    first, _ = evaluate_robust_portfolio(task, _Workbench(), _portfolio())
    second, _ = evaluate_robust_portfolio(task, _Workbench(), _portfolio())
    first_value = first["candidates"][0]["formal_robustness"]["cvar"]
    second_value = second["candidates"][0]["formal_robustness"]["cvar"]
    assert first_value == second_value


@pytest.mark.parametrize("alpha", [0.0, 1.0, 1.5])
def test_cvar_rejects_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError):
        conditional_value_at_risk(np.asarray([1.0, 2.0]), alpha)
    with pytest.raises(ValueError):
        RobustnessSpec(objective="cvar", cvar_alpha=alpha).validate()


def test_cvar_requires_alpha_in_both_task_contracts() -> None:
    with pytest.raises(ValueError, match="requires cvar_alpha"):
        RobustnessSpec(objective="cvar").validate()
    with pytest.raises(ValidationError, match="requires cvar_alpha"):
        RobustnessContract.model_validate({"objective": "cvar"})


def test_cvar_rejects_empty_or_nonfinite_losses() -> None:
    with pytest.raises(ValueError):
        conditional_value_at_risk(np.asarray([]), 0.1)
    with pytest.raises(ValueError):
        conditional_value_at_risk(np.asarray([1.0, math.inf]), 0.1)


def test_differentiable_training_accepts_cvar_objective() -> None:
    pytest.importorskip("torch")
    from tmm_engine import MaterialRegistry
    from tmm_engine.optimization import DifferentiableThicknessOptimizer

    result = DifferentiableThicknessOptimizer(MaterialRegistry()).optimize(_cvar_task())
    assert result.audit["robust_training"]["objective"] == "cvar"
