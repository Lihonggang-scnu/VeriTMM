from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tmm_engine import (  # noqa: E402
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
from tmm_engine.optimization import DifferentiableThicknessOptimizer  # noqa: E402
from tmm_engine.schemas import RobustnessSpec  # noqa: E402
from tmm_engine.uncertainty import (  # noqa: E402
    apply_thickness_boundary_policy,
    apply_thickness_boundary_policy_torch,
    final_robustness_seed,
    sample_normal_offsets,
)

pytestmark = pytest.mark.requires_torch


def test_training_and_final_use_same_truncate_boundary_semantics() -> None:
    nominal = np.asarray([[0.2, 5.0]], dtype=np.float64)
    offsets = np.asarray([[-1.0, -10.0], [0.5, 1.0]], dtype=np.float64)
    raw = nominal + offsets
    numpy_bounded = apply_thickness_boundary_policy(
        raw,
        boundary_policy="truncate",
        min_thickness_physical_nm=0.1,
    )
    torch_bounded = apply_thickness_boundary_policy_torch(
        torch.tensor(raw, dtype=torch.float64),
        boundary_policy="truncate",
        min_thickness_physical_nm=0.1,
    ).numpy()
    np.testing.assert_allclose(torch_bounded, numpy_bounded, rtol=0.0, atol=0.0)
    assert np.all(numpy_bounded >= 0.1)


def test_training_and_final_ensembles_are_reproducible_but_seed_disjoint() -> None:
    training = sample_normal_offsets(
        seed=7,
        sample_count=8,
        layer_count=2,
        sigma_nm=2.0,
    )
    replay = sample_normal_offsets(
        seed=7,
        sample_count=8,
        layer_count=2,
        sigma_nm=2.0,
    )
    final = sample_normal_offsets(
        seed=final_robustness_seed(7),
        sample_count=8,
        layer_count=2,
        sigma_nm=2.0,
    )
    np.testing.assert_array_equal(training, replay)
    assert not np.array_equal(training, final)
    assert final_robustness_seed(7) != 7


def test_real_differentiable_optimizer_serializes_shared_robust_policy() -> None:
    simulation = SimulationTask(
        stack=StackSpec(
            layers=(
                LayerSpec(
                    None,
                    100.0,
                    constant_n=1.38,
                    min_thickness_nm=20.0,
                    max_thickness_nm=220.0,
                ),
            ),
            incident=MediumSpec.air(),
            exit=MediumSpec(constant_n=1.5),
        ),
        spectrum=SpectralGrid(start_nm=520.0, stop_nm=580.0, points=9),
        illumination=IlluminationSpec((0.0,), ("unpolarized",)),
    )
    task = OptimizationTask(
        simulation=simulation,
        targets=(SpectralTarget("R", 0.0, 540.0, 560.0),),
        optimizer=OptimizerSpec(max_steps=2, starts=1, seed=9),
        robustness=RobustnessSpec(
            samples_per_step=2,
            final_samples=8,
            seed=17,
            thickness_sigma_nm=2.0,
            boundary_policy="truncate",
            min_thickness_physical_nm=0.1,
        ),
    )

    result = DifferentiableThicknessOptimizer(MaterialRegistry()).optimize(task)
    audit = result.audit["robust_training"]
    assert result.status == "completed"
    assert audit["boundary_policy"] == "truncate"
    assert audit["min_thickness_physical_nm"] == 0.1
    assert audit["training_seed"] == 17
    assert audit["final_seed"] == final_robustness_seed(17)
    assert audit["final_validation_is_independent"] is True
