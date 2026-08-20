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
from tmm_engine.robust_optimization import evaluate_robust_portfolio, select_robust_roles
from tmm_engine.schemas import RobustnessSpec


def _candidate(
    candidate_id: str,
    *,
    nominal_loss: float,
    robust_loss: float,
    source: str = "optimized_best",
    physics_status: str = "physically_valid",
    validation_status: str = "passed",
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "independent_validation_status": validation_status,
        "physics_status": physics_status,
        "certificate_id": f"certificate-{candidate_id}",
        "metadata": {"objective_loss": nominal_loss, "source": source, "thicknesses_nm": [100.0]},
        "formal_robustness": {"robust_objective": robust_loss},
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
        "selected_roles": {},
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
    assert report["selected_roles"]["best_nominal"] == "valid"
    assert report["selected_roles"]["best_robust"] == "valid"
