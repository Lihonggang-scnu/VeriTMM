"""Independent post-optimization robustness evaluation and role selection."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from .capabilities import failure_from_exception
from .protocol.models import PROTOCOL_VERSION
from .schemas import OptimizationTask


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
    return weighted / max(total_weight, 1e-12), (all_passed if all_defined else None)


def select_robust_roles(candidates: list[dict[str, Any]]) -> dict[str, str | None]:
    """Select interpretable roles without collapsing them into one opaque score."""

    admissible = [
        item
        for item in candidates
        if item.get("independent_validation_status") == "passed"
        and item.get("physics_status") in {"physically_valid", "physically_valid_with_limits"}
        and isinstance(item.get("formal_robustness"), dict)
        and item["formal_robustness"].get("robust_objective") is not None
    ]
    if not admissible:
        return {"best_nominal": None, "best_robust": None, "best_quantized": None}
    best_nominal = min(
        admissible,
        key=lambda item: float(item["metadata"].get("objective_loss", float("inf"))),
    )["candidate_id"]
    best_robust = min(
        admissible,
        key=lambda item: float(item["formal_robustness"]["robust_objective"]),
    )["candidate_id"]
    quantized = [
        item for item in admissible if item.get("metadata", {}).get("source") == "quantized_best"
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
        "best_robust": str(best_robust),
        "best_quantized": None if best_quantized is None else str(best_quantized),
    }


def evaluate_robust_portfolio(
    task: OptimizationTask,
    workbench: Any,
    portfolio: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate every admitted candidate with fresh NumPy TMM samples."""

    if task.robustness is None or not task.robustness.enabled:
        raise ValueError("formal robust portfolio requires enabled robustness settings")
    cfg = task.robustness
    rng = np.random.default_rng(int(cfg.seed) + 1_000_003)
    offsets = rng.normal(
        0.0,
        float(cfg.thickness_sigma_nm),
        size=(int(cfg.final_samples), len(task.simulation.stack.layers)),
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
        for sample_index, delta in enumerate(offsets):
            try:
                thicknesses = nominal + delta
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
                loss, passed = _objective_and_attainment(task, forward)
                losses.append(float(loss))
                if passed is not None:
                    passes.append(bool(passed))
            except Exception as exc:
                failed_samples.append(
                    {
                        "sample_index": sample_index,
                        "failure": failure_from_exception(exc).to_dict(),
                    }
                )
        if not losses:
            robust_objective = None
            mean_loss = None
            std_loss = None
            worst_loss = None
        else:
            values = np.asarray(losses, dtype=np.float64)
            mean_loss = float(np.mean(values))
            std_loss = float(np.std(values))
            worst_loss = float(np.max(values))
            if cfg.objective == "expected_loss":
                robust_objective = mean_loss
            elif cfg.objective == "worst_case_loss":
                robust_objective = worst_loss
            else:
                robust_objective = mean_loss + float(cfg.k_sigma) * std_loss
        row["formal_robustness"] = {
            "objective": cfg.objective,
            "robust_objective": robust_objective,
            "expected_loss": mean_loss,
            "loss_std": std_loss,
            "worst_case_loss": worst_loss,
            "yield": None if not passes else float(sum(passes) / int(cfg.final_samples)),
            "yield_is_defined": bool(passes),
            "sample_count": int(cfg.final_samples),
            "completed_sample_count": len(losses),
            "failed_sample_count": len(failed_samples),
            "failed_samples": failed_samples,
            "seed": int(cfg.seed) + 1_000_003,
            "thickness_sigma_nm": float(cfg.thickness_sigma_nm),
            "backend": "independent_numpy_smatrix",
        }
        records.append(row)
    roles = select_robust_roles(records)
    updated_portfolio = {
        **portfolio,
        "candidates": records,
        "selected_roles": {**portfolio.get("selected_roles", {}), **roles},
        "robust_selection_policy": (
            "Only independently validated and physics-certified candidates are eligible. "
            "Nominal and robustness roles remain separate."
        ),
    }
    report = {
        "schema_version": "veritmm-robustness-report-v1",
        "protocol_version": PROTOCOL_VERSION,
        "status": "evaluated",
        "physics_validity_is_separate": True,
        "training_monte_carlo_is_not_final_proof": True,
        "settings": cfg.__dict__,
        "selected_roles": roles,
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


__all__ = ["evaluate_robust_portfolio", "select_robust_roles"]
