"""Independent post-optimization robustness evaluation and role selection."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from .capabilities import failure_from_exception
from .protocol.models import PROTOCOL_VERSION
from .protocol.uncertainty_budget import (
    UncertaintyBudget,
    UncertaintyComponent,
    UncertaintyType,
)
from .schemas import OptimizationTask
from .uncertainty import (
    apply_thickness_boundary_policy,
    classify_sample_failure,
    empty_failure_taxonomy,
    final_robustness_seed,
    sample_normal_offsets,
    validate_uncertainty_forward,
)


def conditional_value_at_risk(losses: np.ndarray, alpha: float) -> float:
    """Return the mean of the worst ``alpha`` fraction of finite losses.

    The release-stable convention is explicit: sort samples in ascending order,
    take ``ceil(alpha * N)`` largest samples (at least one), and return their
    arithmetic mean.  No interpolated percentile is used.
    """

    values = np.asarray(losses, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("CVaR requires at least one loss sample")
    if not np.isfinite(values).all():
        raise ValueError("CVaR losses must be finite")
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("CVaR alpha must satisfy 0 < alpha < 1")
    tail_count = max(1, int(np.ceil(alpha * values.size)))
    return float(np.mean(np.sort(values)[-tail_count:]))


def _objective_and_attainment(task: OptimizationTask, forward: Any) -> tuple[float, bool | None]:
    wavelengths = task.simulation.spectrum.wavelengths_nm()
    weighted = 0.0
    total_weight = 0.0
    all_defined = True
    all_passed = True
    for target in task.targets:
        channel = forward.channel(float(target.angle_deg), str(target.polarization))
        mask = (wavelengths >= float(target.wavelength_min_nm)) & (
            wavelengths <= float(target.wavelength_max_nm)
        )
        values = np.asarray(channel[target.observable], dtype=np.float64)[mask]
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise FloatingPointError(
                "perturbed robust objective received empty or non-finite observables"
            )
        if target.constraint == "match":
            errors = (values - float(target.target)) ** 2
        elif target.constraint == "at_least":
            errors = (1.0 - values) ** 2
        else:
            errors = values**2
        loss = float(
            np.max(errors) if target.aggregation == "worst_case" else np.mean(errors)
        )
        weighted += float(target.weight) * loss
        total_weight += float(target.weight)
        observed_min = float(np.min(values))
        observed_max = float(np.max(values))
        observed_mean = float(np.mean(values))
        observed = (
            observed_min
            if target.constraint == "at_least" and target.aggregation == "worst_case"
            else (
                observed_max
                if target.constraint == "at_most" and target.aggregation == "worst_case"
                else observed_mean
            )
        )
        tolerance = float(target.tolerance or 0.0)
        if target.constraint == "at_least":
            passed = observed >= float(target.target) - tolerance
        elif target.constraint == "at_most":
            passed = observed <= float(target.target) + tolerance
        elif target.tolerance is not None:
            deviation = (
                float(np.max(np.abs(values - float(target.target))))
                if target.aggregation == "worst_case"
                else float(np.mean(np.abs(values - float(target.target))))
            )
            passed = deviation <= tolerance
        else:
            all_defined = False
            passed = True
        all_passed = all_passed and bool(passed)
    objective = weighted / max(total_weight, 1e-12)
    if not np.isfinite(objective):
        raise FloatingPointError("perturbed robust objective is non-finite")
    return objective, (all_passed if all_defined else None)


def select_robust_roles(candidates: list[dict[str, Any]]) -> dict[str, str | None]:
    """Select interpretable roles without collapsing them into one opaque score."""

    nominal_admissible = [
        item
        for item in candidates
        if item.get("independent_validation_status") == "passed"
        and item.get("physics_status") in {"physically_valid", "physically_valid_with_limits"}
    ]
    robust_admissible = [
        item
        for item in nominal_admissible
        if isinstance(item.get("formal_robustness"), dict)
        and item["formal_robustness"].get("robust_objective") is not None
        and item["formal_robustness"].get("robustness_complete") is True
        and item["formal_robustness"].get("failed_sample_count") == 0
        and item["formal_robustness"].get("eligible_for_robust_selection") is True
    ]
    if not nominal_admissible:
        return {"best_nominal": None, "best_robust": None, "best_quantized": None}
    best_nominal = min(
        nominal_admissible,
        key=lambda item: float(item["metadata"].get("objective_loss", float("inf"))),
    )["candidate_id"]
    best_robust = (
        None
        if not robust_admissible
        else min(
            robust_admissible,
            key=lambda item: float(item["formal_robustness"]["robust_objective"]),
        )["candidate_id"]
    )
    quantized = [
        item
        for item in robust_admissible
        if item.get("metadata", {}).get("source") == "quantized_best"
    ]
    best_quantized = (
        None
        if not quantized
        else min(
            quantized,
            key=lambda item: float(item["formal_robustness"]["robust_objective"]),
        )["candidate_id"]
    )
    return {
        "best_nominal": str(best_nominal),
        "best_robust": None if best_robust is None else str(best_robust),
        "best_quantized": None if best_quantized is None else str(best_quantized),
    }


def invalidate_unverified_robust_roles(
    portfolio: dict[str, Any],
    *,
    status: str = "pending_formal_evaluation",
    failure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Prevent heuristic robustness scores from masquerading as formal results."""

    original_roles = dict(portfolio.get("selected_roles", {}))
    roles = {
        key: value
        for key, value in original_roles.items()
        if key not in {"most_robust", "best_robust", "best_quantized"}
    }
    roles["best_nominal"] = original_roles.get(
        "best_nominal", original_roles.get("best_performance")
    )
    roles["best_robust"] = None
    roles["best_quantized"] = None
    roles["most_robust"] = None
    updated = {
        **portfolio,
        "selected_roles": roles,
        "robust_selection_status": status,
        "robust_selection_policy": (
            "Robust roles remain null until an independent final ensemble completes "
            "without failed samples; heuristic screening scores are never promoted."
        ),
    }
    if failure is not None:
        updated["robust_selection_failure"] = failure
    else:
        updated.pop("robust_selection_failure", None)
    return updated


def evaluate_robust_portfolio(
    task: OptimizationTask,
    workbench: Any,
    portfolio: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate every admitted candidate with fresh NumPy TMM samples."""

    if task.robustness is None or not task.robustness.enabled:
        raise ValueError("formal robust portfolio requires enabled robustness settings")
    cfg = task.robustness
    final_seed = final_robustness_seed(int(cfg.seed))
    offsets = sample_normal_offsets(
        seed=final_seed,
        sample_count=int(cfg.final_samples),
        layer_count=len(task.simulation.stack.layers),
        sigma_nm=float(cfg.thickness_sigma_nm),
    )
    records: list[dict[str, Any]] = []
    for candidate in portfolio.get("candidates", []):
        row = dict(candidate)
        if row.get("independent_validation_status") != "passed" or row.get(
            "physics_status"
        ) not in {"physically_valid", "physically_valid_with_limits"}:
            row["formal_robustness"] = None
            records.append(row)
            continue
        nominal = np.asarray(row["metadata"]["thicknesses_nm"], dtype=np.float64)
        losses: list[float] = []
        passes: list[bool] = []
        failed_samples: list[dict[str, Any]] = []
        failure_taxonomy = empty_failure_taxonomy()
        for sample_index, delta in enumerate(offsets):
            try:
                raw_thicknesses = nominal + delta
                thicknesses = apply_thickness_boundary_policy(
                    raw_thicknesses,
                    boundary_policy=cfg.boundary_policy,
                    min_thickness_physical_nm=float(
                        cfg.min_thickness_physical_nm
                    ),
                )
                layers = tuple(
                    replace(
                        layer,
                        thickness_nm=float(value),
                        optimizable=False,
                        min_thickness_nm=None,
                        max_thickness_nm=None,
                    )
                    for layer, value in zip(
                        task.simulation.stack.layers, thicknesses, strict=True
                    )
                )
                simulation = replace(
                    task.simulation,
                    stack=replace(task.simulation.stack, layers=layers),
                    solver="smatrix",
                )
                simulation.validate()
                forward = workbench.simulate(simulation)
                validate_uncertainty_forward(forward)
                loss, passed = _objective_and_attainment(task, forward)
                losses.append(float(loss))
                if passed is not None:
                    passes.append(bool(passed))
            except Exception as exc:
                category = classify_sample_failure(exc)
                failure_taxonomy[category] += 1
                failed_samples.append(
                    {
                        "sample_index": sample_index,
                        "failure_category": category,
                        "failure": failure_from_exception(exc).to_dict(),
                    }
                )
        requested_sample_count = int(cfg.final_samples)
        completed_sample_count = len(losses)
        failed_sample_count = len(failed_samples)
        robustness_complete = (
            completed_sample_count == requested_sample_count
            and failed_sample_count == 0
        )
        if not losses:
            mean_loss = None
            std_loss = None
            worst_loss = None
        else:
            values = np.asarray(losses, dtype=np.float64)
            mean_loss = float(np.mean(values))
            std_loss = float(np.std(values))
            worst_loss = float(np.max(values))
        if not robustness_complete or mean_loss is None:
            robust_objective = None
            cvar_loss = None
            cvar_tail_sample_count = None
        elif cfg.objective == "expected_loss":
            robust_objective = mean_loss
            cvar_loss = None
            cvar_tail_sample_count = None
        elif cfg.objective == "worst_case_loss":
            robust_objective = worst_loss
            cvar_loss = None
            cvar_tail_sample_count = None
        elif cfg.objective == "cvar":
            if cfg.cvar_alpha is None:
                raise ValueError("cvar objective requires cvar_alpha")
            # This is recomputed from the independent final ensemble; training
            # candidates are proposals and cannot certify their own tail risk.
            cvar_loss = conditional_value_at_risk(values, float(cfg.cvar_alpha))
            cvar_tail_sample_count = max(
                1,
                int(np.ceil(float(cfg.cvar_alpha) * len(values))),
            )
            robust_objective = cvar_loss
        else:
            robust_objective = mean_loss + float(cfg.k_sigma) * float(std_loss)
            cvar_loss = None
            cvar_tail_sample_count = None
        conditional_yield = (
            None if not passes else float(sum(passes) / len(passes))
        )
        overall_success_fraction = (
            None
            if not passes
            else float(sum(passes) / requested_sample_count)
        )
        uncertainty_budget = None
        if cvar_loss is not None and cvar_tail_sample_count is not None:
            uncertainty_budget = UncertaintyBudget(
                sampling_components=[
                    UncertaintyComponent(
                        source="cvar_finite_sample",
                        uncertainty_type=UncertaintyType.TYPE_A,
                        degrees_of_freedom=max(0, cvar_tail_sample_count - 1),
                        notes="Tail mean estimated from the independent final Monte Carlo ensemble.",
                    )
                ]
            ).model_dump(mode="json")
        row["formal_robustness"] = {
            "objective": cfg.objective,
            "robust_objective": robust_objective,
            "expected_loss": mean_loss,
            "loss_std": std_loss,
            "worst_case_loss": worst_loss,
            "cvar": cvar_loss,
            "cvar_alpha": cfg.cvar_alpha,
            "cvar_tail_sample_count": cvar_tail_sample_count,
            "uncertainty_budget": uncertainty_budget,
            "yield": conditional_yield,
            "conditional_yield": conditional_yield,
            "overall_success_fraction": overall_success_fraction,
            "yield_is_defined": bool(passes),
            "sample_count": requested_sample_count,
            "requested_sample_count": requested_sample_count,
            "completed_sample_count": completed_sample_count,
            "failed_sample_count": failed_sample_count,
            "completion_fraction": float(
                completed_sample_count / requested_sample_count
            ),
            "robustness_complete": robustness_complete,
            "eligible_for_robust_selection": robustness_complete,
            "failure_taxonomy": failure_taxonomy,
            "failed_samples": failed_samples,
            "seed": final_seed,
            "training_seed": int(cfg.seed),
            "final_seed": final_seed,
            "distribution": cfg.distribution,
            "thickness_sigma_nm": float(cfg.thickness_sigma_nm),
            "boundary_policy": cfg.boundary_policy,
            "min_thickness_physical_nm": float(
                cfg.min_thickness_physical_nm
            ),
            "backend": "independent_numpy_smatrix",
        }
        records.append(row)
    roles = select_robust_roles(records)
    updated_portfolio = {
        **portfolio,
        "candidates": records,
        "selected_roles": {
            **{
                key: value
                for key, value in portfolio.get("selected_roles", {}).items()
                if key != "most_robust"
            },
            **roles,
            # Backward-compatible alias, now governed by the formal zero-failure gate.
            "most_robust": roles["best_robust"],
        },
        "robust_selection_policy": (
            "Nominal selection requires independent validation and accepted physics. "
            "Robust selection additionally requires a complete zero-failure final ensemble; "
            "survivor-only statistics are ineligible."
        ),
        "robust_selection_status": "evaluated",
    }
    report = {
        "schema_version": "veritmm-robustness-report-v2",
        "protocol_version": PROTOCOL_VERSION,
        "status": "evaluated",
        "physics_validity_is_separate": True,
        "training_monte_carlo_is_not_final_proof": True,
        "settings": {
            **cfg.__dict__,
            "distribution": cfg.distribution,
            "training_seed": int(cfg.seed),
            "final_seed": final_seed,
        },
        "selected_roles": roles,
        "uncertainty_budget": next(
            (
                item["formal_robustness"].get("uncertainty_budget")
                for item in records
                if isinstance(item.get("formal_robustness"), dict)
                and item["formal_robustness"].get("uncertainty_budget") is not None
            ),
            None,
        ),
        "candidates": [
            {
                "candidate_id": item["candidate_id"],
                "certificate_id": item.get("certificate_id"),
                "physics_status": item.get("physics_status"),
                "independent_validation_status": item.get("independent_validation_status"),
                "formal_robustness": item.get("formal_robustness"),
            }
            for item in records
        ],
    }
    return updated_portfolio, report


__all__ = [
    "conditional_value_at_risk",
    "evaluate_robust_portfolio",
    "invalidate_unverified_robust_roles",
    "select_robust_roles",
]
