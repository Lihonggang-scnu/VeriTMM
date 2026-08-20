"""Verifier-separated sensitivity and manufacturing-uncertainty studies."""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .acceptance import AcceptanceSettings, certify_simulation
from .capabilities import failure_from_exception
from .material_registry import MaterialRegistry
from .protocol.models import (
    PROTOCOL_VERSION,
    SensitivityTaskPayload,
    ToleranceTaskPayload,
)
from .run_artifacts import (
    build_result_summary,
    prepare_output_directory,
    stable_payload_sha256,
    write_json,
    write_run_result,
)
from .study_metrics import evaluate_metric, metric_constraint_passes
from .task_io import simulation_task_from_dict
from .workbench import TMMWorkbench

SENSITIVITY_RESULT_SCHEMA_VERSION = "veritmm-sensitivity-result-v1"
TOLERANCE_RESULT_SCHEMA_VERSION = "veritmm-tolerance-result-v1"
ROBUSTNESS_REPORT_SCHEMA_VERSION = "veritmm-robustness-report-v1"


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""

    if trials < 1 or successes < 0 or successes > trials:
        raise ValueError("Wilson interval requires 0 <= successes <= trials and trials > 0")
    n = float(trials)
    p = float(successes) / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _metric_tensor(optimizer: Any, simulation: Any, metric: Any, thicknesses_nm: Any) -> Any:
    torch = optimizer.torch
    spec = metric.model_dump(mode="python")
    aggregation = spec["aggregation"]
    if aggregation == "threshold_band_width":
        raise ValueError("threshold_band_width is non-differentiable and unsupported for sensitivity")
    wavelengths_nm = simulation.spectrum.wavelengths_nm()
    nk_stack, _ = optimizer._build_nk_stack(simulation)
    wavelength_t = torch.tensor(
        wavelengths_nm * 1e-3,
        dtype=optimizer.real_dtype,
        device=optimizer.device,
    )
    solver = optimizer._solver_class(
        polarization=spec["polarization"],
        dtype_real=optimizer.real_dtype,
        dtype_complex=optimizer.complex_dtype,
    ).to(optimizer.device)
    result = solver(
        thicknesses_nm * 1e-3,
        nk_stack.unsqueeze(0),
        wavelength_t,
        theta_rad=float(spec["angle_deg"]) * math.pi / 180.0,
    )
    values = {"R": result.R, "T": result.T, "A": result.A}[spec["observable"]][0]
    mask_np = np.ones(wavelengths_nm.shape, dtype=bool)
    if spec.get("wavelength_min_nm") is not None:
        mask_np &= wavelengths_nm >= float(spec["wavelength_min_nm"])
    if spec.get("wavelength_max_nm") is not None:
        mask_np &= wavelengths_nm <= float(spec["wavelength_max_nm"])
    if not np.any(mask_np):
        raise ValueError("sensitivity metric does not overlap the wavelength grid")
    mask = torch.tensor(mask_np, dtype=torch.bool, device=optimizer.device)
    selected = values[mask]
    if aggregation == "mean":
        return torch.mean(selected)
    if aggregation == "min":
        return torch.min(selected)
    if aggregation == "max":
        return torch.max(selected)
    if aggregation == "worst_case":
        return (
            torch.min(selected)
            if spec.get("threshold_direction", "at_least") == "at_least"
            else torch.max(selected)
        )
    if aggregation == "value_at_wavelength":
        target = float(spec["wavelength_nm"])
        if target < wavelengths_nm[0] or target > wavelengths_nm[-1]:
            raise ValueError("metric wavelength_nm lies outside the simulation grid")
        upper = int(np.searchsorted(wavelengths_nm, target, side="left"))
        if upper == 0:
            return values[0]
        if upper >= wavelengths_nm.size:
            return values[-1]
        lower = upper - 1
        fraction = (target - wavelengths_nm[lower]) / (
            wavelengths_nm[upper] - wavelengths_nm[lower]
        )
        return values[lower] * (1.0 - fraction) + values[upper] * fraction
    raise ValueError(f"unsupported sensitivity aggregation: {aggregation}")


def execute_sensitivity(
    request: SensitivityTaskPayload,
    output_dir: str | Path,
    *,
    device: str = "cpu",
    registry: MaterialRegistry | None = None,
) -> dict[str, Any]:
    """Calculate autograd thickness sensitivities and audit each by NumPy FD."""

    started = time.perf_counter()
    output = prepare_output_directory(output_dir)
    run_id = f"run_{uuid.uuid4().hex}"
    normalized = {
        "schema_version": "sensitivity-task-v1",
        "mode": "sensitivity",
        "sensitivity": request.model_dump(mode="json"),
    }
    task_sha256 = stable_payload_sha256(normalized)
    write_json(output / "NORMALIZED_TASK.json", normalized)
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    certificate: dict[str, Any] | None = None
    result_payload: dict[str, Any] | None = None
    try:
        simulation = simulation_task_from_dict(request.simulation.model_dump(mode="python"))
        if simulation.stack.has_incoherent_layers:
            raise ValueError("sensitivity currently requires a fully coherent stack")
        registry = registry or MaterialRegistry()
        workbench = TMMWorkbench(registry)
        certified = certify_simulation(workbench, simulation, AcceptanceSettings())
        certificate = certified.certificate
        write_json(output / "PHYSICS_ACCEPTANCE_CERTIFICATE.json", certificate)
        if not certificate.get("accepted") or certified.result is None:
            failures.extend(certificate.get("failures", []))
            raise RuntimeError("nominal simulation did not pass physics acceptance")

        from .optimization import DifferentiableThicknessOptimizer

        optimizer = DifferentiableThicknessOptimizer(registry, device=device)
        torch = optimizer.torch
        thickness = torch.tensor(
            [[float(layer.thickness_nm) for layer in simulation.stack.layers]],
            dtype=optimizer.real_dtype,
            device=optimizer.device,
            requires_grad=True,
        )
        scalar = _metric_tensor(optimizer, simulation, request.metric, thickness)
        scalar.backward()
        autodiff = thickness.grad.detach().cpu().numpy()[0]
        metric_value = float(scalar.detach().cpu().item())
        rows: list[dict[str, Any]] = []
        for index, layer in enumerate(simulation.stack.layers):
            if not layer.optimizable:
                continue
            h = float(
                request.finite_difference_step_nm
                or max(1e-3, abs(float(layer.thickness_nm)) * 1e-4)
            )
            h = min(h, max(1e-6, float(layer.thickness_nm) * 0.49))
            plus_layers = list(simulation.stack.layers)
            minus_layers = list(simulation.stack.layers)
            plus_layers[index] = replace(
                layer,
                thickness_nm=float(layer.thickness_nm) + h,
                optimizable=False,
                min_thickness_nm=None,
                max_thickness_nm=None,
            )
            minus_layers[index] = replace(
                layer,
                thickness_nm=float(layer.thickness_nm) - h,
                optimizable=False,
                min_thickness_nm=None,
                max_thickness_nm=None,
            )
            plus = workbench.simulate(
                replace(simulation, stack=replace(simulation.stack, layers=tuple(plus_layers)))
            )
            minus = workbench.simulate(
                replace(simulation, stack=replace(simulation.stack, layers=tuple(minus_layers)))
            )
            fd = (evaluate_metric(plus, request.metric) - evaluate_metric(minus, request.metric)) / (
                2.0 * h
            )
            ad = float(autodiff[index])
            absolute_error = abs(ad - fd)
            near_zero = max(abs(ad), abs(fd)) < 1e-8
            relative_error = None if near_zero else absolute_error / max(abs(ad), abs(fd), 1e-15)
            audit_passed = (
                absolute_error <= float(request.absolute_error_tolerance)
                if near_zero
                else relative_error <= float(request.relative_error_tolerance)
            )
            normalized_derivative = (
                ad * float(layer.thickness_nm) / max(abs(metric_value), 1e-15)
            )
            rows.append(
                {
                    "layer_index": index,
                    "label": layer.label,
                    "thickness_nm": float(layer.thickness_nm),
                    "finite_difference_step_nm": h,
                    "autodiff_derivative_per_nm": ad,
                    "finite_difference_derivative_per_nm": float(fd),
                    "absolute_error": float(absolute_error),
                    "relative_error": None if relative_error is None else float(relative_error),
                    "near_zero_gradient": bool(near_zero),
                    "audit_passed": bool(audit_passed),
                    "normalized_derivative": float(normalized_derivative),
                    "absolute_importance": abs(float(normalized_derivative)),
                }
            )
        ranking = [
            item["layer_index"]
            for item in sorted(
                rows,
                key=lambda item: (-float(item["absolute_importance"]), int(item["layer_index"])),
            )
        ]
        result_payload = {
            "schema_version": SENSITIVITY_RESULT_SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "run_id": run_id,
            "task_sha256": task_sha256,
            "status": "passed" if all(item["audit_passed"] for item in rows) else "audit_failed",
            "metric": request.metric.model_dump(mode="json"),
            "metric_value": metric_value,
            "parameters": rows,
            "ranking": ranking,
            "fixed_layer_indices": [
                index
                for index, layer in enumerate(simulation.stack.layers)
                if not layer.optimizable
            ],
            "finite_difference_audit": {
                "method": "independent_numpy_central_difference",
                "relative_error_tolerance": float(request.relative_error_tolerance),
                "absolute_error_tolerance": float(request.absolute_error_tolerance),
                "passed": all(item["audit_passed"] for item in rows),
            },
        }
        write_json(output / "SENSITIVITY_RESULT.json", result_payload)
    except Exception as exc:
        failure = failure_from_exception(exc).to_dict()
        if failure not in failures:
            failures.append(failure)

    status = "completed" if result_payload and result_payload["status"] == "passed" else "failed"
    summary = build_result_summary(
        mode="sensitivity",
        forward=None,
        certificate=certificate,
        warnings=warnings,
        run_id=run_id,
        task_sha256=task_sha256,
        run_status=status,
    )
    if result_payload:
        summary["sensitivity"] = {
            "metric_value": result_payload["metric_value"],
            "ranking": result_payload["ranking"],
            "finite_difference_audit": result_payload["finite_difference_audit"],
        }
    write_json(output / "RESULT_SUMMARY.json", summary)
    write_json(
        output / "RUN_MANIFEST.json",
        {
            "schema_version": "veritmm-run-manifest-v1",
            "mode": "sensitivity",
            "status": status,
            "run_id": run_id,
            "task_sha256": task_sha256,
            "wall_seconds": time.perf_counter() - started,
        },
    )
    return write_run_result(
        output,
        operation="sensitivity",
        task_sha256=task_sha256,
        status=status,
        ok=status == "completed",
        summary=summary,
        warnings=warnings,
        failures=failures,
        certificate_id=None if certificate is None else certificate.get("certificate_id"),
        run_id=run_id,
    )


def execute_tolerance(
    request: ToleranceTaskPayload,
    output_dir: str | Path,
    *,
    registry: MaterialRegistry | None = None,
) -> dict[str, Any]:
    """Run a seeded thickness uncertainty study and report statistical yield."""

    started = time.perf_counter()
    output = prepare_output_directory(output_dir)
    run_id = f"run_{uuid.uuid4().hex}"
    normalized = {
        "schema_version": "tolerance-task-v1",
        "mode": "tolerance",
        "tolerance": request.model_dump(mode="json"),
    }
    task_sha256 = stable_payload_sha256(normalized)
    write_json(output / "NORMALIZED_TASK.json", normalized)
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    certificate: dict[str, Any] | None = None
    tolerance_result: dict[str, Any] | None = None
    robustness_report: dict[str, Any] | None = None
    try:
        simulation = simulation_task_from_dict(request.simulation.model_dump(mode="python"))
        layer_count = len(simulation.stack.layers)
        for uncertainty in request.uncertainties:
            if uncertainty.layer_index >= layer_count:
                raise ValueError(
                    f"uncertainty layer_index {uncertainty.layer_index} is outside the stack"
                )
        registry = registry or MaterialRegistry()
        workbench = TMMWorkbench(registry)
        certified = certify_simulation(workbench, simulation, AcceptanceSettings())
        certificate = certified.certificate
        write_json(output / "PHYSICS_ACCEPTANCE_CERTIFICATE.json", certificate)
        if not certificate.get("accepted"):
            failures.extend(certificate.get("failures", []))
            raise RuntimeError("nominal simulation did not pass physics acceptance")

        rng = np.random.default_rng(int(request.seed))
        values: list[float] = []
        samples: list[dict[str, Any]] = []
        pass_count = 0
        failed_count = 0
        uncertainties = {item.layer_index: item for item in request.uncertainties}
        for sample_index in range(int(request.sample_count)):
            global_bias = (
                0.0
                if request.global_correlated_bias_nm is None
                else float(rng.normal(0.0, float(request.global_correlated_bias_nm)))
            )
            draws: list[float] = []
            layers = []
            try:
                for index, layer in enumerate(simulation.stack.layers):
                    delta = global_bias
                    uncertainty = uncertainties.get(index)
                    if uncertainty is not None and uncertainty.distribution == "normal":
                        delta += float(rng.normal(0.0, float(uncertainty.sigma_nm)))
                    elif uncertainty is not None:
                        delta += float(
                            rng.uniform(
                                -float(uncertainty.half_width_nm),
                                float(uncertainty.half_width_nm),
                            )
                        )
                    thickness = float(layer.thickness_nm) + delta
                    draws.append(thickness)
                    layers.append(
                        replace(
                            layer,
                            thickness_nm=thickness,
                            optimizable=False,
                            min_thickness_nm=None,
                            max_thickness_nm=None,
                        )
                    )
                perturbed = replace(
                    simulation,
                    stack=replace(simulation.stack, layers=tuple(layers)),
                )
                perturbed.validate()
                forward = workbench.simulate(perturbed)
                metric_value = evaluate_metric(forward, request.metric)
                passed = metric_constraint_passes(
                    metric_value,
                    {"constraint": request.target.constraint, "value": request.target.value},
                )
                values.append(float(metric_value))
                pass_count += int(passed)
                samples.append(
                    {
                        "sample_index": sample_index,
                        "status": "completed",
                        "thicknesses_nm": draws,
                        "metric_value": float(metric_value),
                        "target_passed": bool(passed),
                    }
                )
            except Exception as exc:
                failed_count += 1
                samples.append(
                    {
                        "sample_index": sample_index,
                        "status": "failed",
                        "thicknesses_nm": draws,
                        "metric_value": None,
                        "target_passed": False,
                        "failure": failure_from_exception(exc).to_dict(),
                    }
                )
        if not values:
            raise RuntimeError("all tolerance samples failed")
        array = np.asarray(values, dtype=np.float64)
        total = int(request.sample_count)
        yield_value = float(pass_count / total)
        ci = wilson_interval(pass_count, total)
        stats = {
            "mean": float(np.mean(array)),
            "std": float(np.std(array)),
            "p01": float(np.quantile(array, 0.01)),
            "p05": float(np.quantile(array, 0.05)),
            "p50": float(np.quantile(array, 0.50)),
            "p95": float(np.quantile(array, 0.95)),
            "p99": float(np.quantile(array, 0.99)),
            "worst_case": (
                float(np.min(array))
                if request.target.constraint == "at_least"
                else float(np.max(array))
            ),
        }
        tolerance_result = {
            "schema_version": TOLERANCE_RESULT_SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "run_id": run_id,
            "task_sha256": task_sha256,
            "status": "completed",
            "metric": request.metric.model_dump(mode="json"),
            "target": request.target.model_dump(mode="json"),
            "uncertainties": [item.model_dump(mode="json") for item in request.uncertainties],
            "sample_count": total,
            "completed_sample_count": len(values),
            "failed_sample_count": failed_count,
            "seed": int(request.seed),
            "statistics": stats,
            "target_pass_probability": yield_value,
            "yield": yield_value,
            "yield_ci95": ci,
            "yield_ci_method": "wilson_score_interval",
            "samples": samples,
        }
        write_json(output / "TOLERANCE_RESULT.json", tolerance_result)
        robustness_report = {
            "schema_version": ROBUSTNESS_REPORT_SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "run_id": run_id,
            "task_sha256": task_sha256,
            "status": "evaluated",
            "physics_validity_is_separate": True,
            "nominal_physics_certificate_id": certificate.get("certificate_id"),
            "nominal_physics_accepted": bool(certificate.get("accepted")),
            "yield": yield_value,
            "yield_ci95": ci,
            "yield_ci_method": "wilson_score_interval",
            "worst_case_metric": stats["worst_case"],
            "sample_count": total,
            "failed_sample_count": failed_count,
            "seed": int(request.seed),
        }
        write_json(output / "ROBUSTNESS_REPORT.json", robustness_report)
    except Exception as exc:
        failure = failure_from_exception(exc).to_dict()
        if failure not in failures:
            failures.append(failure)

    status = "completed" if tolerance_result is not None else "failed"
    summary = build_result_summary(
        mode="tolerance",
        forward=None,
        certificate=certificate,
        warnings=warnings,
        run_id=run_id,
        task_sha256=task_sha256,
        run_status=status,
    )
    if tolerance_result is not None:
        summary["robustness"] = {
            "yield": tolerance_result["yield"],
            "yield_ci95": tolerance_result["yield_ci95"],
            "statistics": tolerance_result["statistics"],
            "sample_count": tolerance_result["sample_count"],
            "failed_sample_count": tolerance_result["failed_sample_count"],
        }
    write_json(output / "RESULT_SUMMARY.json", summary)
    write_json(
        output / "RUN_MANIFEST.json",
        {
            "schema_version": "veritmm-run-manifest-v1",
            "mode": "tolerance",
            "status": status,
            "run_id": run_id,
            "task_sha256": task_sha256,
            "wall_seconds": time.perf_counter() - started,
            "robustness_report_status": None
            if robustness_report is None
            else robustness_report["status"],
        },
    )
    return write_run_result(
        output,
        operation="tolerance",
        task_sha256=task_sha256,
        status=status,
        ok=status == "completed",
        summary=summary,
        warnings=warnings,
        failures=failures,
        certificate_id=None if certificate is None else certificate.get("certificate_id"),
        run_id=run_id,
    )


__all__ = [
    "ROBUSTNESS_REPORT_SCHEMA_VERSION",
    "SENSITIVITY_RESULT_SCHEMA_VERSION",
    "TOLERANCE_RESULT_SCHEMA_VERSION",
    "execute_sensitivity",
    "execute_tolerance",
    "wilson_interval",
]
