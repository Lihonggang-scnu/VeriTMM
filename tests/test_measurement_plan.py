from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tmm_engine.cli import main as cli_main
from tmm_engine.fitting.fit_task import (
    FitParameter,
    FitResult,
    FitTask,
    IdentifiabilityReport,
    MeasuredDataPoint,
    MeasurementType,
)
from tmm_engine.fitting.measurement_plan import (
    MeasurementAction,
    MeasurementPlanTask,
    build_measurement_plan,
)


def _fit_result(jacobian: list[list[float]]) -> FitResult:
    parameter_count = len(jacobian[0])
    layers = [
        {"constant_n": 2.0 - 0.1 * index, "thickness_nm": 100.0 - 10.0 * index}
        for index in range(parameter_count)
    ]
    measurements = [
        MeasuredDataPoint(
            wavelength_nm=500.0 + index,
            measurement_type=MeasurementType.REFLECTANCE,
            value=0.2,
            uncertainty=1.0,
        )
        for index in range(len(jacobian))
    ]
    task = FitTask(
        structure={"layers": layers, "substrate": {"constant_n": 1.5}},
        measurements=measurements,
        fit_parameters=[
            FitParameter(
                name=f"thickness_layer_{index}",
                layer_index=index,
                bounds=(50.0, 150.0),
                initial_guess=100.0 - 10.0 * index,
            )
            for index in range(parameter_count)
        ],
    )
    rank = min(parameter_count, len(jacobian))
    return FitResult(
        task=task,
        converged=True,
        iterations=1,
        best_fit_parameters={
            f"thickness_layer_{index}": 100.0 - 10.0 * index
            for index in range(parameter_count)
        },
        residuals=[0.0] * len(jacobian),
        jacobian=jacobian,
        identifiability=IdentifiabilityReport(
            rmse=0.0,
            degrees_of_freedom=0,
            jacobian_condition_number=1.0 if rank == parameter_count else float("inf"),
            singular_values=[1.0] * rank,
            effective_rank=rank,
            parameter_correlation_matrix=[
                [1.0 if row == column else 0.0 for column in range(parameter_count)]
                for row in range(parameter_count)
            ],
            identifiability_status=(
                "well_determined" if rank == parameter_count else "non_identifiable"
            ),
        ),
        fit_certificate={
            "physics_certificate": None,
            "physics_validity": "not_certified",
            "certificate_authority": "fit_quality_only",
        },
    )


def _action(wavelength: float, sigma: float = 1.0) -> MeasurementAction:
    return MeasurementAction(
        wavelength_nm=wavelength,
        angle_deg=0.0,
        polarization="s",
        measurement_type=MeasurementType.REFLECTANCE,
        sigma=sigma,
    )


def _provider(values: dict[float, list[float]]):
    return lambda _fit, action: values[action.wavelength_nm]


def test_d_optimal_one_parameter_matches_hand_computation() -> None:
    fit = _fit_result([[1.0]])
    candidates = [_action(510.0), _action(520.0)]
    result = build_measurement_plan(
        MeasurementPlanTask(fit_result=fit, candidates=candidates, criterion="d_optimal"),
        jacobian_provider=_provider({510.0: [3.0], 520.0: [1.0]}),
    )

    assert result.selected_actions == [candidates[0]]
    assert result.selected_information_gain[0] == pytest.approx(math.log(10.0))
    assert result.rejected_alternatives[0].action == candidates[1]
    assert result.method == "local_fisher_deterministic"
    assert result.scope_note.startswith("Local linearization")


def test_a_optimal_is_available_and_uses_uncertainty_reduction() -> None:
    fit = _fit_result([[1.0]])
    result = build_measurement_plan(
        MeasurementPlanTask(
            fit_result=fit,
            candidates=[_action(510.0), _action(520.0)],
            criterion="a_optimal",
        ),
        jacobian_provider=_provider({510.0: [2.0], 520.0: [1.0]}),
    )

    assert result.criterion == "a_optimal"
    assert result.selected_actions[0].wavelength_nm == 510.0
    assert result.selected_information_gain[0] > 0.0


def test_d_optimal_prefers_orthogonal_direction_to_collinear_candidate() -> None:
    fit = _fit_result([[1.0, 0.0]])
    collinear = _action(510.0)
    orthogonal = _action(520.0)
    result = build_measurement_plan(
        MeasurementPlanTask(
            fit_result=fit,
            candidates=[collinear, orthogonal],
            criterion="d_optimal",
        ),
        jacobian_provider=_provider({510.0: [2.0, 0.0], 520.0: [0.0, 1.0]}),
    )

    assert result.selected_actions == [orthogonal]
    assert result.status == "completed_with_pseudoinverse"
    assert result.used_pseudoinverse is True
    assert result.failure is not None
    assert result.failure.code == "rank_deficient_fisher"


def test_candidate_pool_permutation_does_not_change_selected_set() -> None:
    fit = _fit_result([[1.0, 0.0], [0.0, 1.0]])
    candidates = [_action(510.0), _action(520.0), _action(530.0)]
    values = {510.0: [5.0, 0.0], 520.0: [0.0, 2.0], 530.0: [2.0, 1.0]}
    first = build_measurement_plan(
        MeasurementPlanTask(fit_result=fit, candidates=candidates, n_select=2),
        jacobian_provider=_provider(values),
    )
    second = build_measurement_plan(
        MeasurementPlanTask(fit_result=fit, candidates=list(reversed(candidates)), n_select=2),
        jacobian_provider=_provider(values),
    )

    assert first.selected_action_ids == second.selected_action_ids
    assert [item.action_id for item in first.rejected_alternatives] == [
        item.action_id for item in second.rejected_alternatives
    ]


def test_noise_scaling_reduces_information_gain() -> None:
    fit = _fit_result([[1.0]])
    low_noise = _action(510.0, sigma=1.0)
    high_noise = _action(520.0, sigma=2.0)
    result = build_measurement_plan(
        MeasurementPlanTask(fit_result=fit, candidates=[low_noise, high_noise]),
        jacobian_provider=_provider({510.0: [2.0], 520.0: [2.0]}),
    )
    scores = {item.action.wavelength_nm: item.information_gain for item in result.candidate_scores}

    assert scores[510.0] > scores[520.0]


def test_greedy_second_pick_recomputes_information() -> None:
    fit = _fit_result([[1.0, 0.0], [0.0, 1.0]])
    candidates = [_action(510.0), _action(520.0), _action(530.0)]
    values = {510.0: [10.0, 0.0], 520.0: [5.0, 0.0], 530.0: [0.0, 2.0]}
    result = build_measurement_plan(
        MeasurementPlanTask(fit_result=fit, candidates=candidates, n_select=2),
        jacobian_provider=_provider(values),
    )

    assert [action.wavelength_nm for action in result.selected_actions] == [510.0, 530.0]
    collinear_score = next(item for item in result.candidate_scores if item.action.wavelength_nm == 520.0)
    orthogonal_score = next(item for item in result.candidate_scores if item.action.wavelength_nm == 530.0)
    assert len(collinear_score.score_history) == 2
    assert collinear_score.score_history[1]["information_gain"] < orthogonal_score.score_history[1]["information_gain"]


def test_real_fit_result_and_plan_measurement_cli(tmp_path: Path) -> None:
    fit_output = tmp_path / "fit_result.json"
    candidate_path = tmp_path / "candidates.json"
    plan_output = tmp_path / "measurement_plan.json"
    source = Path("benchmarks/cases/fitting/fit_synthetic_ar_coating.json")
    candidate_path.write_text(
        json.dumps(
            [
                _action(525.0, sigma=0.01).model_dump(mode="json"),
                _action(575.0, sigma=0.01).model_dump(mode="json"),
            ]
        ),
        encoding="utf-8",
    )

    assert cli_main(["fit", str(source), "--output", str(fit_output)]) == 0
    assert (
        cli_main(
            [
                "plan-measurement",
                str(fit_output),
                "--candidates",
                str(candidate_path),
                "--criterion",
                "d_optimal",
                "--n",
                "1",
                "--output",
                str(plan_output),
            ]
        )
        == 0
    )
    payload = json.loads(plan_output.read_text(encoding="utf-8"))

    assert payload["method"] == "local_fisher_deterministic"
    assert len(payload["candidate_scores"]) == 2
    assert len(payload["rejected_alternatives"]) == 1
    assert "evidence_coverage" not in payload
    assert payload["scope_note"].find("not global Bayesian") >= 0
