"""Deterministic next-measurement planning from local Fisher information."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Literal, Sequence

import numpy as np
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from ..material_registry import MaterialRegistry
from ..workbench import TMMWorkbench
from .fit_task import FitResult, MeasuredDataPoint, MeasurementType
from .optimizer import (
    build_simulation_task,
    execute_forward_simulation,
    extract_simulation_value,
)


class MeasurementPlanError(ValueError):
    """Typed failure raised when a plan cannot be constructed safely."""

    def __init__(self, code: str, message: str) -> None:
        self.code = str(code)
        super().__init__(message)


class MeasurementAction(BaseModel):
    """One candidate scalar observation for the next experiment."""

    model_config = ConfigDict(extra="forbid")

    wavelength_nm: float = Field(gt=0)
    angle_deg: float = Field(default=0.0, ge=0.0, lt=90.0)
    polarization: Literal["s", "p", "unpolarized"] = "unpolarized"
    measurement_type: MeasurementType
    sigma: float = Field(gt=0.0)


class MeasurementPlanTask(BaseModel):
    """Inputs to a deterministic local-Fisher measurement plan."""

    model_config = ConfigDict(extra="forbid")

    fit_result: FitResult
    candidates: list[MeasurementAction] = Field(
        min_length=1,
        validation_alias=AliasChoices("candidates", "candidate_pool"),
    )
    criterion: Literal["d_optimal", "a_optimal"] = "d_optimal"
    n_select: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _validate_selection_count(self) -> "MeasurementPlanTask":
        if self.n_select > len(self.candidates):
            raise ValueError("n_select cannot exceed the candidate pool size")
        return self

    @property
    def candidate_pool(self) -> list[MeasurementAction]:
        """Compatibility name for callers that use the contract terminology."""

        return self.candidates


class MeasurementPlanFailure(BaseModel):
    """Typed diagnostic attached when a safe fallback was required."""

    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    fallback: str | None = None
    recoverable: bool = True


class MeasurementCandidateScore(BaseModel):
    """Auditable score history for one candidate action."""

    model_config = ConfigDict(extra="forbid")

    action: MeasurementAction
    action_id: str
    jacobian: list[float]
    information_gain: float
    score_history: list[dict[str, Any]] = Field(default_factory=list)
    selected: bool = False
    selection_round: int | None = None
    rejection_reason: str | None = None


class MeasurementPlanResult(BaseModel):
    """Complete next-measurement plan and retained rejected alternatives."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["measurement-plan-v1"] = "measurement-plan-v1"
    status: Literal["completed", "completed_with_pseudoinverse"]
    method: Literal["local_fisher_deterministic"] = "local_fisher_deterministic"
    scope_note: str = (
        "Local linearization around the fitted point; this is deterministic Fisher "
        "information, not global Bayesian experimental design."
    )
    criterion: Literal["d_optimal", "a_optimal"]
    n_select: int
    parameter_names: list[str]
    selected_actions: list[MeasurementAction]
    selected_action_ids: list[str]
    selected_information_gain: list[float]
    candidate_scores: list[MeasurementCandidateScore]
    rejected_alternatives: list[MeasurementCandidateScore]
    fisher_information_before: list[list[float]]
    fisher_information_after: list[list[float]]
    baseline_rank: int
    final_rank: int
    rank_deficient: bool
    used_pseudoinverse: bool
    failure: MeasurementPlanFailure | None = None


JacobianProvider = Callable[[FitResult, MeasurementAction], Sequence[float]]


def measurement_action_id(action: MeasurementAction) -> str:
    """Return a stable content identity for one action."""

    payload = {
        "angle_deg": float(action.angle_deg),
        "measurement_type": action.measurement_type.value,
        "polarization": action.polarization,
        "sigma": float(action.sigma),
        "wavelength_nm": float(action.wavelength_nm),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _action_sort_key(action: MeasurementAction) -> tuple[Any, ...]:
    return (
        float(action.wavelength_nm),
        float(action.angle_deg),
        str(action.polarization),
        action.measurement_type.value,
        float(action.sigma),
    )


def _action_measurement(action: MeasurementAction) -> MeasuredDataPoint:
    return MeasuredDataPoint(
        wavelength_nm=action.wavelength_nm,
        angle_deg=action.angle_deg,
        polarization=action.polarization,
        measurement_type=action.measurement_type,
        value=0.0,
        uncertainty=action.sigma,
    )


def _parameter_values(fit_result: FitResult) -> tuple[list[str], dict[str, float]]:
    names = [parameter.name for parameter in fit_result.task.fit_parameters]
    values: dict[str, float] = {}
    for parameter in fit_result.task.fit_parameters:
        if parameter.name in fit_result.best_fit_parameters:
            value = float(fit_result.best_fit_parameters[parameter.name])
        elif parameter.initial_guess is not None:
            value = float(parameter.initial_guess)
        else:
            value = float(sum(parameter.bounds) / 2.0)
        if not parameter.bounds[0] <= value <= parameter.bounds[1]:
            raise MeasurementPlanError(
                "fit_parameter_out_of_bounds",
                f"fitted parameter {parameter.name!r} lies outside its declared bounds",
            )
        values[parameter.name] = value
    return names, values


def _evaluate_action(
    fit_result: FitResult,
    action: MeasurementAction,
    parameters: dict[str, float],
    *,
    workbench: TMMWorkbench,
) -> float:
    task = build_simulation_task(
        fit_result.task.structure,
        [_action_measurement(action)],
        parameters,
    )
    result = execute_forward_simulation(task, workbench=workbench)
    value = extract_simulation_value(result, _action_measurement(action))
    if not np.isfinite(value):
        raise MeasurementPlanError(
            "nonfinite_candidate_prediction",
            f"candidate action {measurement_action_id(action)} produced a non-finite prediction",
        )
    return float(value)


def _finite_difference_action(
    fit_result: FitResult,
    action: MeasurementAction,
    *,
    workbench: TMMWorkbench,
) -> np.ndarray:
    """Build one unweighted candidate Jacobian row at the fitted point."""

    names, base = _parameter_values(fit_result)
    row: list[float] = []
    for parameter in fit_result.task.fit_parameters:
        name = parameter.name
        value = float(base[name])
        low, high = (float(parameter.bounds[0]), float(parameter.bounds[1]))
        step = max(1e-4, abs(value) * 1e-5)
        can_minus = value - step >= low
        can_plus = value + step <= high
        if not can_minus and not can_plus:
            row.append(0.0)
            continue
        if can_minus and can_plus:
            minus_params = dict(base)
            plus_params = dict(base)
            minus_params[name] = value - step
            plus_params[name] = value + step
            minus = _evaluate_action(fit_result, action, minus_params, workbench=workbench)
            plus = _evaluate_action(fit_result, action, plus_params, workbench=workbench)
            row.append(float((plus - minus) / (2.0 * step)))
        elif can_plus:
            plus_params = dict(base)
            plus_params[name] = value + step
            base_value = _evaluate_action(fit_result, action, base, workbench=workbench)
            plus = _evaluate_action(fit_result, action, plus_params, workbench=workbench)
            row.append(float((plus - base_value) / step))
        else:
            minus_params = dict(base)
            minus_params[name] = value - step
            base_value = _evaluate_action(fit_result, action, base, workbench=workbench)
            minus = _evaluate_action(fit_result, action, minus_params, workbench=workbench)
            row.append(float((base_value - minus) / step))
    if len(row) != len(names) or not np.all(np.isfinite(row)):
        raise MeasurementPlanError(
            "nonfinite_candidate_jacobian",
            f"candidate action {measurement_action_id(action)} produced an invalid Jacobian row",
        )
    return np.asarray(row, dtype=np.float64)


def _measurement_sigma(measurement: MeasuredDataPoint) -> float:
    if measurement.uncertainty is not None:
        return float(measurement.uncertainty)
    return float(1.0 / measurement.weight)


def _weighted_existing_jacobian(
    fit_result: FitResult,
    *,
    names: Sequence[str],
    workbench: TMMWorkbench,
    jacobian_provider: JacobianProvider | None,
) -> np.ndarray:
    expected_shape = (len(fit_result.task.measurements), len(names))
    if fit_result.jacobian is not None:
        matrix = np.asarray(fit_result.jacobian, dtype=np.float64)
        if matrix.shape != expected_shape:
            raise MeasurementPlanError(
                "jacobian_shape_mismatch",
                f"stored FitResult.jacobian has shape {matrix.shape}; expected {expected_shape}",
            )
        if not np.all(np.isfinite(matrix)):
            raise MeasurementPlanError("nonfinite_fit_jacobian", "stored FitResult.jacobian is non-finite")
        return matrix

    rows: list[np.ndarray] = []
    for measurement in fit_result.task.measurements:
        action = MeasurementAction(
            wavelength_nm=measurement.wavelength_nm,
            angle_deg=measurement.angle_deg,
            polarization=measurement.polarization,
            measurement_type=measurement.measurement_type,
            sigma=_measurement_sigma(measurement),
        )
        derivative = (
            np.asarray(jacobian_provider(fit_result, action), dtype=np.float64)
            if jacobian_provider is not None
            else _finite_difference_action(fit_result, action, workbench=workbench)
        )
        if derivative.shape != (len(names),) or not np.all(np.isfinite(derivative)):
            raise MeasurementPlanError(
                "invalid_fit_jacobian",
                f"existing measurement {measurement_action_id(action)} has an invalid Jacobian row",
            )
        rows.append(derivative / _measurement_sigma(measurement))
    return np.asarray(rows, dtype=np.float64)


def _matrix_rank(matrix: np.ndarray) -> tuple[int, np.ndarray, float]:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0, 1.0)
    tolerance = max(scale * 1e-10, 1e-12)
    rank = int(np.sum(eigenvalues > tolerance))
    return rank, eigenvalues, tolerance


def _log_pseudodeterminant(matrix: np.ndarray, tolerance: float) -> float | None:
    eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    positive = eigenvalues[eigenvalues > tolerance]
    if positive.size == 0:
        return None
    return float(np.sum(np.log(positive)))


def _information_gain(
    before: np.ndarray,
    after: np.ndarray,
    criterion: Literal["d_optimal", "a_optimal"],
) -> tuple[float, int, int, bool]:
    before_rank, before_eigenvalues, before_tolerance = _matrix_rank(before)
    after_rank, after_eigenvalues, after_tolerance = _matrix_rank(after)
    rank_deficient = before_rank < before.shape[0] or after_rank < after.shape[0]
    if criterion == "d_optimal":
        before_logdet = _log_pseudodeterminant(before, before_tolerance)
        after_logdet = _log_pseudodeterminant(after, after_tolerance)
        if after_logdet is None:
            gain = 0.0
        elif before_logdet is None:
            # With no positive baseline eigenvalue, compare the candidate's
            # pseudo-determinant directly.  This preserves the expected
            # one-parameter ordering without pretending log(det(0)) exists.
            gain = after_logdet
        else:
            gain = after_logdet - before_logdet
        if after_rank > before_rank:
            # Rank growth is the dominant scientific event under the explicit
            # pseudo-determinant fallback; the flag in the result makes this
            # convention auditable rather than silently regularizing it away.
            gain += 1e6 * float(after_rank - before_rank)
    else:
        scale = max(
            float(np.max(np.abs(before_eigenvalues))) if before_eigenvalues.size else 0.0,
            float(np.max(np.abs(after_eigenvalues))) if after_eigenvalues.size else 0.0,
            1.0,
        )
        if rank_deficient:
            regularizer = max(scale * 1e-12, 1e-12)
            before_trace = float(np.trace(np.linalg.pinv(before + regularizer * np.eye(before.shape[0]))))
            after_trace = float(np.trace(np.linalg.pinv(after + regularizer * np.eye(after.shape[0]))))
        else:
            before_trace = float(np.trace(np.linalg.pinv(before)))
            after_trace = float(np.trace(np.linalg.pinv(after)))
        gain = before_trace - after_trace
    return max(float(gain), 0.0), before_rank, after_rank, rank_deficient


def build_measurement_plan(
    task: MeasurementPlanTask,
    *,
    workbench: TMMWorkbench | None = None,
    jacobian_provider: JacobianProvider | None = None,
) -> MeasurementPlanResult:
    """Greedily select actions using deterministic local Fisher information.

    ``jacobian_provider`` is an in-process testing and adapter seam.  When it
    is omitted, candidate rows are built by bounded central finite differences
    through the same forward model used by fitting.
    """

    fit_result = task.fit_result
    if not fit_result.converged:
        raise MeasurementPlanError(
            "fit_not_converged",
            "measurement planning requires a converged FitResult",
        )
    names, _ = _parameter_values(fit_result)
    if not names:
        raise MeasurementPlanError("no_fit_parameters", "measurement planning requires fit parameters")
    workbench = workbench or TMMWorkbench(MaterialRegistry())

    actions = sorted(task.candidates, key=_action_sort_key)
    action_ids = [measurement_action_id(action) for action in actions]
    if len(set(action_ids)) != len(action_ids):
        raise MeasurementPlanError("duplicate_candidate_action", "candidate actions must be unique")

    baseline_rows = _weighted_existing_jacobian(
        fit_result,
        names=names,
        workbench=workbench,
        jacobian_provider=jacobian_provider,
    )
    fisher_before = baseline_rows.T @ baseline_rows
    current_fisher = np.asarray(fisher_before, dtype=np.float64)
    baseline_rank, _, _ = _matrix_rank(current_fisher)
    used_pseudoinverse = baseline_rank < len(names)
    candidate_rows: dict[str, np.ndarray] = {}
    for action, action_id in zip(actions, action_ids, strict=True):
        derivative = (
            np.asarray(jacobian_provider(fit_result, action), dtype=np.float64)
            if jacobian_provider is not None
            else _finite_difference_action(fit_result, action, workbench=workbench)
        )
        if derivative.shape != (len(names),) or not np.all(np.isfinite(derivative)):
            raise MeasurementPlanError(
                "invalid_candidate_jacobian",
                f"candidate action {action_id} has an invalid Jacobian row",
            )
        candidate_rows[action_id] = derivative / float(action.sigma)

    records = {
        action_id: MeasurementCandidateScore(
            action=action,
            action_id=action_id,
            jacobian=candidate_rows[action_id].tolist(),
            information_gain=0.0,
        )
        for action, action_id in zip(actions, action_ids, strict=True)
    }
    remaining = list(action_ids)
    selected_ids: list[str] = []
    selected_gains: list[float] = []
    rank_deficient = used_pseudoinverse
    for selection_round in range(1, task.n_select + 1):
        round_scores: list[tuple[str, float]] = []
        for action_id in remaining:
            row = candidate_rows[action_id]
            after = current_fisher + np.outer(row, row)
            gain, _, after_rank, candidate_rank_deficient = _information_gain(
                current_fisher, after, task.criterion
            )
            records[action_id].information_gain = gain
            records[action_id].score_history.append(
                {"round": selection_round, "information_gain": gain}
            )
            round_scores.append((action_id, gain))
            rank_deficient = rank_deficient or candidate_rank_deficient or after_rank < len(names)
        if not round_scores:
            break
        chosen_id, chosen_gain = min(
            round_scores,
            key=lambda item: (-float(item[1]), item[0]),
        )
        selected_ids.append(chosen_id)
        selected_gains.append(float(chosen_gain))
        records[chosen_id].selected = True
        records[chosen_id].selection_round = selection_round
        current_fisher = current_fisher + np.outer(candidate_rows[chosen_id], candidate_rows[chosen_id])
        remaining.remove(chosen_id)
        used_pseudoinverse = used_pseudoinverse or rank_deficient

    final_rank, _, _ = _matrix_rank(current_fisher)
    used_pseudoinverse = used_pseudoinverse or final_rank < len(names)
    failure = None
    status: Literal["completed", "completed_with_pseudoinverse"] = "completed"
    if used_pseudoinverse:
        status = "completed_with_pseudoinverse"
        failure = MeasurementPlanFailure(
            code="rank_deficient_fisher",
            message=(
                "The accumulated Fisher matrix was rank-deficient at one or more "
                "greedy steps; pseudo-determinants or a regularized pseudo-inverse "
                "were used explicitly."
            ),
            fallback="pseudo_inverse",
            recoverable=True,
        )

    rejected: list[MeasurementCandidateScore] = []
    for action_id in remaining:
        records[action_id].rejection_reason = "not_selected_by_greedy_budget"
        rejected.append(records[action_id])
    ordered_records = [records[action_id] for action_id in action_ids]
    selected_actions = [records[action_id].action for action_id in selected_ids]
    return MeasurementPlanResult(
        status=status,
        criterion=task.criterion,
        n_select=task.n_select,
        parameter_names=names,
        selected_actions=selected_actions,
        selected_action_ids=selected_ids,
        selected_information_gain=selected_gains,
        candidate_scores=ordered_records,
        rejected_alternatives=rejected,
        fisher_information_before=fisher_before.tolist(),
        fisher_information_after=current_fisher.tolist(),
        baseline_rank=baseline_rank,
        final_rank=final_rank,
        rank_deficient=rank_deficient,
        used_pseudoinverse=used_pseudoinverse,
        failure=failure,
    )


__all__ = [
    "MeasurementAction",
    "MeasurementCandidateScore",
    "MeasurementPlanError",
    "MeasurementPlanFailure",
    "MeasurementPlanResult",
    "MeasurementPlanTask",
    "build_measurement_plan",
    "measurement_action_id",
]
