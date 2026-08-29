from __future__ import annotations

import numpy as np

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
from tmm_engine.robust_optimization import (
    evaluate_robust_portfolio,
    invalidate_unverified_robust_roles,
    select_robust_roles,
)
from tmm_engine.schemas import RobustnessSpec


def _candidate(
    candidate_id: str,
    *,
    nominal_loss: float,
    robust_loss: float,
    source: str = "optimized_best",
    physics_status: str = "physically_valid",
    validation_status: str = "passed",
    thickness_nm: float = 100.0,
    robustness_complete: bool = True,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "independent_validation_status": validation_status,
        "physics_status": physics_status,
        "certificate_id": f"certificate-{candidate_id}",
        "metadata": {
            "objective_loss": nominal_loss,
            "source": source,
            "thicknesses_nm": [thickness_nm],
        },
        "formal_robustness": {
            "robust_objective": robust_loss if robustness_complete else None,
            "robustness_complete": robustness_complete,
            "failed_sample_count": 0 if robustness_complete else 1,
            "eligible_for_robust_selection": robustness_complete,
        },
    }


def test_nominal_and_robust_roles_can_select_different_admissible_candidates() -> None:
    candidates = [
        _candidate("fragile_nominal", nominal_loss=0.10, robust_loss=0.80),
        _candidate("stable_robust", nominal_loss=0.20, robust_loss=0.20),
        _candidate(
            "quantized",
            nominal_loss=0.30,
            robust_loss=0.30,
            source="quantized_best",
        ),
        _candidate(
            "invalid_not_admissible",
            nominal_loss=-100.0,
            robust_loss=-100.0,
            physics_status="rejected_physics",
        ),
    ]

    roles = select_robust_roles(candidates)
    assert roles == {
        "best_nominal": "fragile_nominal",
        "best_robust": "stable_robust",
        "best_quantized": "quantized",
    }
    assert roles["best_nominal"] != roles["best_robust"]


def test_unverified_robustness_can_never_leave_a_heuristic_winner() -> None:
    portfolio = {
        "selected_roles": {
            "best_performance": "nominal",
            "most_robust": "heuristic_only",
            "easiest_to_manufacture": "simple",
        },
        "candidates": [],
    }
    failure = {"code": "numerical_failure", "message": "controlled"}

    updated = invalidate_unverified_robust_roles(
        portfolio,
        status="formal_evaluation_failed",
        failure=failure,
    )

    assert updated["selected_roles"]["best_nominal"] == "nominal"
    assert updated["selected_roles"]["most_robust"] is None
    assert updated["selected_roles"]["best_robust"] is None
    assert updated["selected_roles"]["best_quantized"] is None
    assert updated["selected_roles"]["easiest_to_manufacture"] == "simple"
    assert updated["robust_selection_status"] == "formal_evaluation_failed"
    assert updated["robust_selection_failure"] == failure


def test_robust_portfolio_keeps_invalid_candidates_out_of_formal_evaluation() -> None:
    class _Forward:
        wavelengths_nm = np.asarray([500.0, 550.0, 600.0])

        def channel(self, _angle: float = 0.0, _polarization: str = "unpolarized"):
            return {"T": np.asarray([0.9, 0.9, 0.9])}

    class _Workbench:
        def simulate(self, _task: SimulationTask) -> _Forward:
            return _Forward()

    simulation = SimulationTask(
        stack=StackSpec(
            layers=(LayerSpec(None, 100.0, constant_n=1.4, optimizable=True),),
            incident=MediumSpec.air(),
            exit=MediumSpec(constant_n=1.5),
        ),
        spectrum=SpectralGrid(start_nm=500.0, stop_nm=600.0, points=3),
        illumination=IlluminationSpec((0.0,), ("unpolarized",)),
    )
    task = OptimizationTask(
        simulation=simulation,
        targets=(
            SpectralTarget(
                "T",
                0.9,
                500.0,
                600.0,
                constraint="at_least",
                tolerance=0.01,
                name="transmission",
            ),
        ),
        optimizer=OptimizerSpec(max_steps=1, starts=1),
        robustness=RobustnessSpec(final_samples=8, samples_per_step=2, seed=4),
    )
    portfolio = {
        "selected_roles": {"most_robust": "fragile_survivor"},
        "candidates": [
            _candidate("valid", nominal_loss=0.1, robust_loss=0.1),
            _candidate(
                "not_validated",
                nominal_loss=-1.0,
                robust_loss=-1.0,
                validation_status="failed",
            ),
        ],
    }

    updated, report = evaluate_robust_portfolio(task, _Workbench(), portfolio)
    invalid = next(item for item in updated["candidates"] if item["candidate_id"] == "not_validated")
    valid = next(item for item in updated["candidates"] if item["candidate_id"] == "valid")
    assert invalid["formal_robustness"] is None
    assert valid["formal_robustness"]["backend"] == "independent_numpy_smatrix"
    assert report["physics_validity_is_separate"] is True
    assert report["training_monte_carlo_is_not_final_proof"] is True
    assert report["settings"]["distribution"] == "normal"
    assert report["settings"]["boundary_policy"] == "truncate"
    assert report["settings"]["training_seed"] == 4
    assert report["settings"]["final_seed"] != 4
    assert report["selected_roles"]["best_nominal"] == "valid"
    assert report["selected_roles"]["best_robust"] == "valid"
    assert updated["selected_roles"]["most_robust"] == "valid"


def test_incomplete_candidate_cannot_win_robust_role_but_can_remain_best_nominal() -> None:
    candidates = [
        _candidate(
            "fragile_survivor",
            nominal_loss=0.01,
            robust_loss=0.001,
            robustness_complete=False,
        ),
        _candidate(
            "complete_candidate",
            nominal_loss=0.012,
            robust_loss=0.02,
        ),
    ]

    roles = select_robust_roles(candidates)
    assert roles["best_nominal"] == "fragile_survivor"
    assert roles["best_robust"] == "complete_candidate"


def test_final_robustness_failure_nulls_objective_and_prevents_survivor_bias() -> None:
    class _Forward:
        wavelengths_nm = np.asarray([500.0, 550.0, 600.0])

        def channel(self, _angle: float = 0.0, _polarization: str = "unpolarized"):
            return {"T": np.asarray([0.9, 0.9, 0.9])}

    class _FaultInjectingWorkbench:
        calls_for_fragile = 0

        def simulate(self, simulation: SimulationTask) -> _Forward:
            thickness = float(simulation.stack.layers[0].thickness_nm)
            if thickness < 150.0:
                self.calls_for_fragile += 1
                if self.calls_for_fragile % 5 == 0:
                    raise FloatingPointError("controlled robust sample failure")
            return _Forward()

    simulation = SimulationTask(
        stack=StackSpec(
            layers=(LayerSpec(None, 100.0, constant_n=1.4, optimizable=True),),
            incident=MediumSpec.air(),
            exit=MediumSpec(constant_n=1.5),
        ),
        spectrum=SpectralGrid(start_nm=500.0, stop_nm=600.0, points=3),
        illumination=IlluminationSpec((0.0,), ("unpolarized",)),
    )
    task = OptimizationTask(
        simulation=simulation,
        targets=(
            SpectralTarget(
                "T",
                0.9,
                500.0,
                600.0,
                constraint="at_least",
                tolerance=0.01,
            ),
        ),
        optimizer=OptimizerSpec(max_steps=1, starts=1),
        robustness=RobustnessSpec(
            final_samples=10,
            samples_per_step=2,
            seed=4,
            thickness_sigma_nm=0.1,
        ),
    )
    portfolio = {
        "selected_roles": {},
        "candidates": [
            _candidate(
                "fragile_survivor",
                nominal_loss=0.01,
                robust_loss=0.001,
                thickness_nm=100.0,
            ),
            _candidate(
                "complete_candidate",
                nominal_loss=0.012,
                robust_loss=0.02,
                thickness_nm=200.0,
            ),
        ],
    }

    updated, report = evaluate_robust_portfolio(
        task, _FaultInjectingWorkbench(), portfolio
    )
    fragile = next(
        item
        for item in updated["candidates"]
        if item["candidate_id"] == "fragile_survivor"
    )
    complete = next(
        item
        for item in updated["candidates"]
        if item["candidate_id"] == "complete_candidate"
    )
    assert fragile["formal_robustness"]["completion_fraction"] == 0.8
    assert fragile["formal_robustness"]["robustness_complete"] is False
    assert fragile["formal_robustness"]["eligible_for_robust_selection"] is False
    assert fragile["formal_robustness"]["robust_objective"] is None
    assert fragile["formal_robustness"]["failure_taxonomy"]["numerical_failure"] == 2
    assert complete["formal_robustness"]["robustness_complete"] is True
    assert complete["formal_robustness"]["robust_objective"] is not None
    assert report["selected_roles"]["best_nominal"] == "fragile_survivor"
    assert report["selected_roles"]["best_robust"] == "complete_candidate"
    assert updated["selected_roles"]["most_robust"] == "complete_candidate"


def test_invalid_forward_audit_makes_candidate_ineligible_for_robust_role() -> None:
    class _Forward:
        def __init__(self, valid: bool) -> None:
            self.audit = {
                "nonfinite_value_count": 0,
                "passivity_check_passed": valid,
                "minimum_observable": 0.0 if valid else -0.2,
                "maximum_observable": 1.0,
                "energy_conservation_max_abs_error": 0.0,
            }

        def channel(self, _angle: float = 0.0, _polarization: str = "unpolarized"):
            return {"T": np.asarray([0.9, 0.9, 0.9])}

    class _Workbench:
        def simulate(self, simulation: SimulationTask) -> _Forward:
            return _Forward(float(simulation.stack.layers[0].thickness_nm) >= 150.0)

    simulation = SimulationTask(
        stack=StackSpec(
            layers=(LayerSpec(None, 100.0, constant_n=1.4, optimizable=True),),
            incident=MediumSpec.air(),
            exit=MediumSpec(constant_n=1.5),
        ),
        spectrum=SpectralGrid(start_nm=500.0, stop_nm=600.0, points=3),
        illumination=IlluminationSpec((0.0,), ("unpolarized",)),
    )
    task = OptimizationTask(
        simulation=simulation,
        targets=(
            SpectralTarget(
                "T",
                0.9,
                500.0,
                600.0,
                constraint="at_least",
                tolerance=0.01,
            ),
        ),
        optimizer=OptimizerSpec(max_steps=1, starts=1),
        robustness=RobustnessSpec(
            final_samples=8,
            samples_per_step=2,
            seed=4,
            thickness_sigma_nm=0.1,
        ),
    )
    portfolio = {
        "selected_roles": {"most_robust": "bad_audit"},
        "candidates": [
            _candidate(
                "bad_audit",
                nominal_loss=0.01,
                robust_loss=0.001,
                thickness_nm=100.0,
            ),
            _candidate(
                "valid_audit",
                nominal_loss=0.02,
                robust_loss=0.02,
                thickness_nm=200.0,
            ),
        ],
    }

    updated, _ = evaluate_robust_portfolio(task, _Workbench(), portfolio)
    invalid = next(
        item for item in updated["candidates"] if item["candidate_id"] == "bad_audit"
    )
    assert invalid["formal_robustness"]["failed_sample_count"] == 8
    assert invalid["formal_robustness"]["failure_taxonomy"]["numerical_failure"] == 8
    assert invalid["formal_robustness"]["eligible_for_robust_selection"] is False
    assert updated["selected_roles"]["best_robust"] == "valid_audit"
    assert updated["selected_roles"]["most_robust"] == "valid_audit"
