"""Reusable execution service behind the agent-facing CLI."""

from __future__ import annotations

import csv
import sys
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .acceptance import AcceptanceSettings, certify_simulation
from .archive.schema_registry import ARCHIVE_SCHEMA_VERSION
from .capabilities import (
    FailureCode,
    FailureRecord,
    enrich_failure_actions,
    failure_from_exception,
)
from .convergence import SpectralConvergenceSettings
from .material_registry import MaterialRegistry
from .preflight import preflight_task
from .protocol.responses import DEFAULT_RESPONSE_DETAIL
from .run_artifacts import (
    PROTOCOL_VERSION,
    build_result_summary,
    file_sha256,
    prepare_output_directory,
    stable_payload_sha256,
    write_json,
    write_run_result,
)
from .schemas import OptimizationTask, SimulationTask, dataclass_to_dict
from .task_io import write_normalized_task
from .workbench import TMMWorkbench


@dataclass(frozen=True)
class ExecutionSettings:
    """Numerical acceptance and output controls for a single run."""

    device: str = "cpu"
    skip_certificate: bool = False
    convergence_max_refinements: int = 6
    convergence_pointwise_tolerance: float = 5e-3
    convergence_integral_tolerance: float = 1e-3
    write_plot: bool = True
    portfolio_max_candidates: int = 6


def _write_spectra(path: Path, result: Any) -> None:
    channel_names = sorted(result.channels)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = ["wavelength_nm"]
        for key in channel_names:
            for observable in ("R", "T", "A", "E_system"):
                if observable in result.channels[key]:
                    header.append(f"{key}|{observable}")
        writer.writerow(header)
        for index, wavelength in enumerate(result.wavelengths_nm):
            row = [float(wavelength)]
            for key in channel_names:
                for observable in ("R", "T", "A", "E_system"):
                    if observable in result.channels[key]:
                        row.append(float(result.channels[key][observable][index]))
            writer.writerow(row)


def _plot(path: Path, result: Any) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        path.with_name("SPECTRA_PLOT_SKIPPED.txt").write_text(
            "matplotlib is not installed; numerical CSV and JSON results are complete.\n",
            encoding="utf-8",
        )
        return False
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    for key, values in sorted(result.channels.items()):
        axis.plot(result.wavelengths_nm, values["R"], label=f"R {key}")
    axis.set(xlabel="Wavelength (nm)", ylabel="Reflectance", ylim=(-0.02, 1.02))
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return True


def _acceptance_settings(settings: ExecutionSettings) -> AcceptanceSettings:
    return AcceptanceSettings(
        require_spectral_convergence=not settings.skip_certificate,
        require_independent_solver=not settings.skip_certificate,
        convergence=SpectralConvergenceSettings(
            max_refinements=settings.convergence_max_refinements,
            max_pointwise_deviation=settings.convergence_pointwise_tolerance,
            max_integral_deviation=settings.convergence_integral_tolerance,
        ),
    )


def _normalized_payload(
    mode: str, task: SimulationTask | OptimizationTask
) -> dict[str, Any]:
    return {
        "mode": mode,
        "simulation" if mode == "simulate" else "optimization": dataclass_to_dict(task),
    }


def _failure_from_validation(validation: dict[str, Any]) -> FailureRecord:
    return enrich_failure_actions(
        FailureRecord(
            FailureCode.OPTIMIZER_FAILURE,
            "Independent recomputation did not validate the differentiable proposal.",
            True,
            context={
                "validation_status": validation.get("status"),
                "absolute_loss_difference": validation.get("absolute_loss_difference"),
                "loss_tolerance": validation.get("loss_tolerance"),
            },
        )
    )


def _robustness_score(
    optimizer: Any,
    task: OptimizationTask,
    thicknesses: list[float],
) -> tuple[float, dict[str, Any]]:
    values = np.asarray(thicknesses, dtype=np.float64)
    delta = float(task.optimizer.quantization_nm or 1.0)
    base_loss, _ = optimizer.evaluate(task, values)
    perturbations: list[dict[str, Any]] = []
    losses: list[float] = []
    for index, layer in enumerate(task.simulation.stack.layers):
        if not layer.optimizable:
            continue
        for sign in (-1.0, 1.0):
            candidate = values.copy()
            low, high = layer.bounds_nm(task.optimizer.thickness_window_nm)
            candidate[index] = np.clip(candidate[index] + sign * delta, low, high)
            loss, _ = optimizer.evaluate(task, candidate)
            losses.append(float(loss))
            perturbations.append(
                {"layer_index": index, "delta_nm": sign * delta, "loss": float(loss)}
            )
    worst = max([float(base_loss), *losses])
    scale = max(abs(float(base_loss)), 1e-4)
    score = float(np.exp(-max(0.0, worst - float(base_loss)) / scale))
    return score, {
        "method": "deterministic_one_layer_at_a_time_thickness_perturbation",
        "delta_nm": delta,
        "base_loss": float(base_loss),
        "perturbations": perturbations,
        "worst_loss": worst,
    }


def _select_portfolio_candidates(
    candidates: list[dict[str, Any]], max_candidates: int
) -> list[dict[str, Any]]:
    limit = max(1, int(max_candidates))
    selected: list[dict[str, Any]] = []
    priority = []
    if candidates:
        priority.extend(
            [
                candidates[0],
                min(
                    candidates,
                    key=lambda row: float(row.get("objective_loss", float("inf"))),
                ),
            ]
        )
    priority.extend(
        next((row for row in candidates if row.get("source") == source), None)
        for source in ("quantized_best", "initial", "optimized_best")
    )
    for item in priority:
        if item is not None and item not in selected:
            selected.append(item)
        if len(selected) >= limit:
            return selected
    while len(selected) < limit:
        remaining = [item for item in candidates if item not in selected]
        if not remaining:
            break
        if not selected:
            selected.append(remaining[0])
            continue
        chosen = max(
            remaining,
            key=lambda item: min(
                np.linalg.norm(
                    np.asarray(item["thicknesses_nm"], dtype=np.float64)
                    - np.asarray(existing["thicknesses_nm"], dtype=np.float64)
                )
                for existing in selected
            ),
        )
        selected.append(chosen)
    return selected


def _write_optimization_portfolio(
    output: Path,
    task: OptimizationTask,
    optimization: Any,
    optimizer: Any,
    workbench: TMMWorkbench,
    certificate_settings: AcceptanceSettings,
    max_candidates: int,
) -> dict[str, Any]:
    candidates = _select_portfolio_candidates(
        list(optimization.candidate_designs), max_candidates
    )
    if not candidates:
        payload = {
            "schema_version": "veritmm-design-portfolio-v1",
            "status": "no_candidates",
            "candidates": [],
            "selected_roles": {},
        }
        write_json(output / "DESIGN_PORTFOLIO.json", payload)
        return payload
    best_thickness = np.asarray(candidates[0]["thicknesses_nm"], dtype=np.float64)
    spans = np.asarray(
        [
            max(
                layer.bounds_nm(task.optimizer.thickness_window_nm)[1]
                - layer.bounds_nm(task.optimizer.thickness_window_nm)[0],
                1.0,
            )
            for layer in task.simulation.stack.layers
        ],
        dtype=np.float64,
    )
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        candidate_dir = output / "candidates" / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        thicknesses = [float(value) for value in candidate["thicknesses_nm"]]
        candidate_result = replace(
            optimization,
            optimized_thicknesses_nm=thicknesses,
            quantized_thicknesses_nm=None,
            quantized_loss=None,
        )
        simulation, _, validation = optimizer.validate_result(
            task, candidate_result, workbench
        )
        certified = certify_simulation(workbench, simulation, certificate_settings)
        validation_path = candidate_dir / "INDEPENDENT_VALIDATION.json"
        certificate_path = candidate_dir / "PHYSICS_ACCEPTANCE_CERTIFICATE.json"
        write_json(validation_path, validation)
        write_json(certificate_path, certified.certificate)
        artifact_paths = [
            validation_path.relative_to(output).as_posix(),
            certificate_path.relative_to(output).as_posix(),
        ]
        if certified.result is not None:
            simulation_path = candidate_dir / "SIMULATION_RESULT.json"
            write_json(simulation_path, certified.result.to_dict())
            artifact_paths.append(simulation_path.relative_to(output).as_posix())
        robustness, robustness_audit = _robustness_score(
            optimizer, task, thicknesses
        )
        layer_count = len(task.simulation.stack.layers)
        simplicity = min(
            1.0,
            1.0 / (1.0 + 0.08 * layer_count)
            + (0.1 if candidate.get("source") == "quantized_best" else 0.0),
        )
        distance = np.linalg.norm(
            (np.asarray(thicknesses, dtype=np.float64) - best_thickness) / spans
        )
        records.append(
            {
                "candidate_id": candidate_id,
                "independent_validation_status": validation.get("status"),
                "physics_status": certified.certificate.get("status"),
                "target_attainment": validation.get("target_attainment", {}),
                "robustness_score": robustness,
                "simplicity_score": simplicity,
                "distinctiveness_score": float(
                    min(1.0, distance / max(np.sqrt(len(spans)), 1.0))
                ),
                "certificate_id": certified.certificate.get("certificate_id"),
                "artifact_paths": artifact_paths,
                "metadata": {
                    "source": candidate.get("source"),
                    "thicknesses_nm": thicknesses,
                    "objective_loss": candidate.get("objective_loss"),
                    "robustness_audit": robustness_audit,
                },
            }
        )
    admissible = [
        row
        for row in records
        if row["independent_validation_status"] == "passed"
        and row["physics_status"] in {"physically_valid", "physically_valid_with_limits"}
    ]
    if not admissible:
        payload = {
            "schema_version": "veritmm-design-portfolio-v1",
            "status": "no_physically_admissible_candidate",
            "candidates": records,
            "selected_roles": {},
        }
    else:
        payload = {
            "schema_version": "veritmm-design-portfolio-v1",
            "status": "completed",
            "selection_policy": (
                "Physics admission is mandatory; objective performance, robustness, "
                "manufacturability, and structural distinctiveness remain separate roles."
            ),
            "candidates": records,
            "selected_roles": {
                "best_performance": min(
                    admissible,
                    key=lambda row: (
                        float(row["metadata"]["objective_loss"])
                        if row["metadata"].get("objective_loss") is not None
                        else float("inf")
                    ),
                )["candidate_id"],
                "best_heuristic_robustness": max(
                    admissible, key=lambda row: float(row["robustness_score"])
                )["candidate_id"],
                "most_robust": None,
                "easiest_to_manufacture": max(
                    admissible, key=lambda row: float(row["simplicity_score"])
                )["candidate_id"],
                "structurally_distinctive": max(
                    admissible, key=lambda row: float(row["distinctiveness_score"])
                )["candidate_id"],
            },
        }
    write_json(output / "DESIGN_PORTFOLIO.json", payload)
    return payload


def execute_task(
    mode: str,
    task: SimulationTask | OptimizationTask,
    output_dir: str | Path,
    *,
    input_path: str | Path | None = None,
    settings: ExecutionSettings | None = None,
    registry: MaterialRegistry | None = None,
    detail: str = DEFAULT_RESPONSE_DETAIL,
) -> dict[str, Any]:
    """Execute one validated task and return a profiled ``RUN_RESULT``.

    Detailed simulation, optimization, certificate, and spectrum documents
    remain in the existing artifact files regardless of ``detail``.
    """

    settings = settings or ExecutionSettings()
    output = prepare_output_directory(output_dir)
    started = time.perf_counter()
    run_id = f"run_{uuid.uuid4().hex}"
    normalized = _normalized_payload(mode, task)
    task_sha256 = stable_payload_sha256(normalized)
    input_sha256 = None
    if input_path is not None and Path(input_path).is_file():
        input_sha256 = file_sha256(input_path)
    manifest: dict[str, Any] = {
        "mode": mode,
        "status": "running",
        "run_id": run_id,
        "task_sha256": task_sha256,
        "task_hash_scope": "normalized_operation_wrapper",
        "archive_schema_version": ARCHIVE_SCHEMA_VERSION,
        "input_sha256": input_sha256,
        "input": None if input_path is None else str(Path(input_path).resolve()),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
    }
    try:
        write_normalized_task(output / "NORMALIZED_TASK.json", mode, task)
        registry = registry or MaterialRegistry()
        preflight = preflight_task(mode, task, registry)
        write_json(output / "PREFLIGHT_REPORT.json", preflight)
    except Exception as exc:
        failure = failure_from_exception(exc).to_dict()
        preflight = {
            "schema_version": "veritmm-preflight-v1",
            "protocol_version": PROTOCOL_VERSION,
            "ok": False,
            "operation": "preflight",
            "mode": mode,
            "status": "rejected",
            "contract_valid": True,
            "capability": None,
            "backend_resolution": {
                "requested_solver": None,
                "resolved_solver": None,
                "reason": "preflight_initialization_failed",
            },
            "materials": [],
            "warnings": [],
            "failures": [failure],
            "estimated_work": {},
        }
        write_json(output / "PREFLIGHT_REPORT.json", preflight)
        summary = build_result_summary(
            mode=mode,
            forward=None,
            certificate=None,
            run_id=run_id,
            task_sha256=task_sha256,
            run_status="preflight_initialization_failed",
        )
        write_json(output / "RESULT_SUMMARY.json", summary)
        manifest.update(
            {
                "status": "preflight_initialization_failed",
                "failures": [failure],
                "wall_seconds": time.perf_counter() - started,
            }
        )
        write_json(output / "RUN_MANIFEST.json", manifest)
        return write_run_result(
            output,
            operation=mode,
            task_sha256=task_sha256,
            status="preflight_initialization_failed",
            ok=False,
            summary=summary,
            failures=[failure],
            run_id=run_id,
            input_sha256=input_sha256,
            detail=detail,
        )
    if not preflight["ok"]:
        failures = list(preflight["failures"])
        summary = build_result_summary(
            mode=mode,
            forward=None,
            certificate=None,
            warnings=preflight["warnings"],
            run_id=run_id,
            task_sha256=task_sha256,
            run_status="preflight_rejected",
        )
        write_json(output / "RESULT_SUMMARY.json", summary)
        manifest.update(
            {
                "status": "preflight_rejected",
                "failures": failures,
                "wall_seconds": time.perf_counter() - started,
            }
        )
        write_json(output / "RUN_MANIFEST.json", manifest)
        return write_run_result(
            output,
            operation=mode,
            task_sha256=task_sha256,
            status="preflight_rejected",
            ok=False,
            summary=summary,
            warnings=preflight["warnings"],
            failures=failures,
            run_id=run_id,
            input_sha256=input_sha256,
            detail=detail,
        )

    workbench = TMMWorkbench(registry)
    certificate: dict[str, Any] | None = None
    forward: Any | None = None
    optimization: Any | None = None
    failures: list[dict[str, Any]] = []
    status = "failed"
    ok = False
    plot_written = False
    portfolio: dict[str, Any] | None = None
    robustness_report: dict[str, Any] | None = None
    try:
        certificate_settings = _acceptance_settings(settings)
        if mode == "simulate":
            if not isinstance(task, SimulationTask):
                raise TypeError("simulate mode requires SimulationTask")
            certified = certify_simulation(workbench, task, certificate_settings)
            certificate = certified.certificate
            forward = certified.result
        elif mode == "optimize":
            if not isinstance(task, OptimizationTask):
                raise TypeError("optimize mode requires OptimizationTask")
            from .optimization import DifferentiableThicknessOptimizer

            optimizer = DifferentiableThicknessOptimizer(registry, device=settings.device)
            optimization = optimizer.optimize(task)
            write_json(output / "OPTIMIZATION_RESULT.json", optimization.to_dict())
            simulation, _, validation = optimizer.validate_result(
                task, optimization, workbench
            )
            write_json(output / "INDEPENDENT_VALIDATION.json", validation)
            if validation.get("status") != "passed":
                failure = _failure_from_validation(validation)
                failures.append(failure.to_dict())
            else:
                certified = certify_simulation(
                    workbench, simulation, certificate_settings
                )
                certificate = certified.certificate
                forward = certified.result
                if certificate.get("accepted"):
                    try:
                        portfolio = _write_optimization_portfolio(
                            output,
                            task,
                            optimization,
                            optimizer,
                            workbench,
                            certificate_settings,
                            settings.portfolio_max_candidates,
                        )
                    except Exception as exc:
                        preflight["warnings"].append(
                            {
                                "code": "optimization_portfolio_unavailable",
                                "severity": "warning",
                                "message": (
                                    "The primary optimized design remains independently certified, "
                                    "but the optional candidate portfolio could not be completed."
                                ),
                                "context": {
                                    "diagnostic": failure_from_exception(exc).to_dict()
                                },
                            }
                        )
                    if (
                        portfolio is not None
                        and task.robustness is not None
                        and task.robustness.enabled
                    ):
                        from .robust_optimization import (
                            evaluate_robust_portfolio,
                            invalidate_unverified_robust_roles,
                        )

                        portfolio = invalidate_unverified_robust_roles(portfolio)
                        write_json(output / "DESIGN_PORTFOLIO.json", portfolio)
                        try:
                            evaluated_portfolio, evaluated_report = evaluate_robust_portfolio(
                                task,
                                workbench,
                                portfolio,
                            )
                            write_json(
                                output / "DESIGN_PORTFOLIO.json", evaluated_portfolio
                            )
                            write_json(
                                output / "ROBUSTNESS_REPORT.json", evaluated_report
                            )
                            portfolio = evaluated_portfolio
                            robustness_report = evaluated_report
                        except Exception as exc:
                            diagnostic = failure_from_exception(exc).to_dict()
                            robustness_report = None
                            portfolio = invalidate_unverified_robust_roles(
                                portfolio,
                                status="formal_evaluation_failed",
                                failure=diagnostic,
                            )
                            write_json(output / "DESIGN_PORTFOLIO.json", portfolio)
                            preflight["warnings"].append(
                                {
                                    "code": "formal_robustness_unavailable",
                                    "severity": "warning",
                                    "message": (
                                        "The nominal candidate portfolio remains available, "
                                        "but no candidate is labelled most robust because the "
                                        "independent final ensemble did not complete."
                                    ),
                                    "context": {"diagnostic": diagnostic},
                                }
                            )
        else:
            raise ValueError("mode must be simulate or optimize")

        if certificate is not None:
            write_json(output / "PHYSICS_ACCEPTANCE_CERTIFICATE.json", certificate)
            failures.extend(list(certificate.get("failures") or []))
        if forward is not None:
            write_json(output / "SIMULATION_RESULT.json", forward.to_dict())
            _write_spectra(output / "SPECTRA.csv", forward)
            plot_written = bool(settings.write_plot and _plot(output / "SPECTRA.png", forward))

        ok = bool(certificate and certificate.get("accepted")) and not failures
        status = str(
            certificate.get("status")
            if certificate is not None
            else ("optimizer_validation_failed" if failures else "failed")
        )
    except Exception as exc:
        failure = failure_from_exception(exc)
        failures.append(failure.to_dict())
        status = "execution_failed"

    summary = build_result_summary(
        mode=mode,
        forward=forward,
        certificate=certificate,
        optimization=optimization,
        warnings=preflight["warnings"],
        run_id=run_id,
        task_sha256=task_sha256,
        run_status="completed" if ok else status,
    )
    if robustness_report is not None:
        summary["robustness"] = {
            "status": robustness_report.get("status"),
            "selected_roles": robustness_report.get("selected_roles", {}),
            "physics_validity_is_separate": True,
            "training_monte_carlo_is_not_final_proof": True,
        }
    write_json(output / "RESULT_SUMMARY.json", summary)
    manifest.update(
        {
            "status": "completed" if ok else status,
            "solver": None if forward is None else forward.solver,
            "physics_audit": None if forward is None else forward.audit,
            "acceptance_certificate_status": None
            if certificate is None
            else certificate.get("status"),
            "acceptance_certificate_id": None
            if certificate is None
            else certificate.get("certificate_id"),
            "plot_written": plot_written,
            "portfolio_candidate_count": (
                0 if portfolio is None else len(portfolio.get("candidates", []))
            ),
            "portfolio_roles": (
                {} if portfolio is None else portfolio.get("selected_roles", {})
            ),
            "robustness_report_status": (
                None if robustness_report is None else robustness_report.get("status")
            ),
            "failures": failures,
            "wall_seconds": time.perf_counter() - started,
        }
    )
    write_json(output / "RUN_MANIFEST.json", manifest)
    return write_run_result(
        output,
        operation=mode,
        task_sha256=task_sha256,
        status="completed" if ok else status,
        ok=ok,
        summary=summary,
        warnings=preflight["warnings"],
        failures=failures,
        certificate_id=None if certificate is None else certificate.get("certificate_id"),
        run_id=run_id,
        input_sha256=input_sha256,
        detail=detail,
    )


__all__ = ["ExecutionSettings", "execute_task"]
