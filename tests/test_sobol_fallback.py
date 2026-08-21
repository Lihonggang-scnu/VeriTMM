"""Tests for the Sobol >16D graceful fallback to LHS."""

from __future__ import annotations

import numpy as np

from tmm_engine.research import (
    ContinuousThicknessVariable,
    DesignSpace,
    DesignSpaceContract,
)
from tmm_engine.research.sampling import (
    SOBOL_MAX_DIMENSION,
    SamplingPlan,
    _normalized_samples_impl,
    sample_candidates,
)
from tmm_engine.schemas import (
    IlluminationSpec,
    LayerSpec,
    MediumSpec,
    SimulationTask,
    SpectralGrid,
    StackSpec,
)


def _make_large_design_space(n_dims: int = SOBOL_MAX_DIMENSION + 1) -> DesignSpace:
    """Build a design space with n_dims ContinuousThicknessVariable entries."""
    layers = tuple(
        LayerSpec(
            None,
            100.0,
            constant_n=1.5,
            min_thickness_nm=50.0,
            max_thickness_nm=200.0,
            label=f"layer_{i}",
        )
        for i in range(n_dims)
    )
    task = SimulationTask(
        stack=StackSpec(
            layers=layers,
            incident=MediumSpec(constant_n=1.0),
            exit=MediumSpec(constant_n=1.5),
        ),
        spectrum=SpectralGrid(start_nm=400.0, stop_nm=800.0, points=11),
        illumination=IlluminationSpec(angles_deg=(0.0,), polarizations=("s",)),
        requested_outputs=("R", "T", "A"),
    )
    variables = tuple(
        ContinuousThicknessVariable(
            name=f"layer_{i}_nm",
            layer_index=i,
            lower_nm=50.0,
            upper_nm=200.0,
        )
        for i in range(n_dims)
    )
    return DesignSpace(DesignSpaceContract(base_task=task, variables=variables))


def _sobol_plan(n: int = 8) -> SamplingPlan:
    return SamplingPlan(strategy="sobol", sample_count=n)


def _sobol_plan_with_fallback(n: int = 8) -> SamplingPlan:
    return SamplingPlan(strategy="sobol", sample_count=n, options={"fallback_policy": "lhs"})


def test_sobol_within_limit_uses_sobol() -> None:
    space = _make_large_design_space(SOBOL_MAX_DIMENSION)
    plan = _sobol_plan()
    _, effective = _normalized_samples_impl(space, plan)
    assert effective == "sobol"


def test_sobol_exceeds_limit_falls_back_to_lhs() -> None:
    space = _make_large_design_space(SOBOL_MAX_DIMENSION + 1)
    plan = _sobol_plan_with_fallback()
    _, effective = _normalized_samples_impl(space, plan)
    assert effective == "latin_hypercube"


def test_fallback_matrix_shape_is_correct() -> None:
    n_dims = SOBOL_MAX_DIMENSION + 1
    n_samples = 12
    space = _make_large_design_space(n_dims)
    plan = _sobol_plan_with_fallback(n_samples)
    matrix, _ = _normalized_samples_impl(space, plan)
    assert matrix.shape == (n_samples, n_dims)


def test_fallback_matrix_values_are_in_unit_interval() -> None:
    space = _make_large_design_space(SOBOL_MAX_DIMENSION + 1)
    plan = _sobol_plan_with_fallback(16)
    matrix, _ = _normalized_samples_impl(space, plan)
    assert (matrix >= 0.0).all()
    assert (matrix <= 1.0).all()


def test_fallback_candidates_have_effective_strategy_metadata() -> None:
    space = _make_large_design_space(SOBOL_MAX_DIMENSION + 1)
    plan = _sobol_plan_with_fallback(4)
    candidates = sample_candidates(space, plan)
    for candidate in candidates:
        meta = candidate.metadata
        assert meta.get("effective_strategy") == "latin_hypercube"
        assert meta.get("declared_strategy") == "sobol"


def test_fallback_is_deterministic_with_same_seed() -> None:
    space = _make_large_design_space(SOBOL_MAX_DIMENSION + 1)
    plan_a = SamplingPlan(strategy="sobol", sample_count=8, seed=42, options={"fallback_policy": "lhs"})
    plan_b = SamplingPlan(strategy="sobol", sample_count=8, seed=42, options={"fallback_policy": "lhs"})
    mat_a, _ = _normalized_samples_impl(space, plan_a)
    mat_b, _ = _normalized_samples_impl(space, plan_b)
    assert np.array_equal(mat_a, mat_b)
