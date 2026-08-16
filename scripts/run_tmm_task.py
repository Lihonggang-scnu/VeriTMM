"""Run a standardized TMM simulation or differentiable optimization task."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tmm_engine import MaterialRegistry, TMMWorkbench  # noqa: E402
from tmm_engine.acceptance import (  # noqa: E402
    AcceptanceSettings,
    certify_simulation,
)
from tmm_engine.convergence import SpectralConvergenceSettings  # noqa: E402
from tmm_engine.task_io import load_task, write_normalized_task  # noqa: E402


def _robustness_score(optimizer: object, task: object, thicknesses: list[float]) -> tuple[float, dict]:
    values = np.asarray(thicknesses, dtype=np.float64)
    delta = float(task.optimizer.quantization_nm or 1.0)
    base_loss, _ = optimizer.evaluate(task, values)
    perturbed_losses = []
    perturbations = []
    for index, layer in enumerate(task.simulation.stack.layers):
        if not layer.optimizable:
            continue
        for sign in (-1.0, 1.0):
            candidate = values.copy()
            lo, hi = layer.bounds_nm(task.optimizer.thickness_window_nm)
            candidate[index] = np.clip(candidate[index] + sign * delta, lo, hi)
            loss, _ = optimizer.evaluate(task, candidate)
            perturbed_losses.append(float(loss))
            perturbations.append({"layer_index": index, "delta_nm": sign * delta, "loss": float(loss)})
    worst = max([float(base_loss), *perturbed_losses])
    scale = max(abs(float(base_loss)), 1e-4)
    score = float(np.exp(-max(0.0, worst - float(base_loss)) / scale))
    return score, {
        "method": "deterministic_one_layer_at_a_time_thickness_perturbation",
        "delta_nm": delta,
        "base_loss": float(base_loss),
        "perturbed_losses": perturbed_losses,
        "perturbations": perturbations,
        "worst_loss": worst,
    }


def _write_optimization_portfolio(
    output_dir: Path,
    task: object,
    optimization: object,
    optimizer: object,
    workbench: object,
    certificate_settings: AcceptanceSettings,
    max_candidates: int,
) -> dict:
    raw_all = list(optimization.candidate_designs)
    limit = max(1, int(max_candidates))
    raw = []
    for preferred_source in (None, "quantized_best", "initial", "optimized_best"):
        item = (
            raw_all[0]
            if preferred_source is None and raw_all
            else next((candidate for candidate in raw_all if candidate.get("source") == preferred_source), None)
        )
        if item is not None and item not in raw:
            raw.append(item)
        if len(raw) >= limit:
            break
    while len(raw) < limit:
        remaining = [item for item in raw_all if item not in raw]
        if not remaining:
            break
        if not raw:
            raw.append(remaining[0])
            continue
        chosen = max(
            remaining,
            key=lambda item: min(
                np.linalg.norm(
                    np.asarray(item["thicknesses_nm"], dtype=np.float64)
                    - np.asarray(existing["thicknesses_nm"], dtype=np.float64)
                )
                for existing in raw
            ),
        )
        raw.append(chosen)
    if not raw:
        return {}
    best_thickness = np.asarray(raw[0]["thicknesses_nm"], dtype=np.float64)
    spans = np.asarray(
        [
            max(layer.bounds_nm(task.optimizer.thickness_window_nm)[1] - layer.bounds_nm(task.optimizer.thickness_window_nm)[0], 1.0)
            for layer in task.simulation.stack.layers
        ],
        dtype=np.float64,
    )
    records = []
    for item in raw:
        candidate_id = str(item["candidate_id"])
        candidate_dir = output_dir / "candidates" / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        thicknesses = [float(value) for value in item["thicknesses_nm"]]
        candidate_result = replace(
            optimization,
            optimized_thicknesses_nm=thicknesses,
            quantized_thicknesses_nm=None,
            quantized_loss=None,
        )
        simulation, forward, validation = optimizer.validate_result(
            task, candidate_result, workbench
        )
        certified = certify_simulation(workbench, simulation, certificate_settings)
        (candidate_dir / "INDEPENDENT_VALIDATION.json").write_text(
            json.dumps(validation, indent=2), encoding="utf-8"
        )
        (candidate_dir / "PHYSICS_ACCEPTANCE_CERTIFICATE.json").write_text(
            json.dumps(certified.certificate, indent=2), encoding="utf-8"
        )
        if certified.result is not None:
            (candidate_dir / "SIMULATION_RESULT.json").write_text(
                json.dumps(certified.result.to_dict(), indent=2), encoding="utf-8"
            )
        robustness, robustness_audit = _robustness_score(
            optimizer, task, thicknesses
        )
        layer_count = len(task.simulation.stack.layers)
        quantized_bonus = 0.1 if item.get("source") == "quantized_best" else 0.0
        simplicity = min(1.0, 1.0 / (1.0 + 0.08 * layer_count) + quantized_bonus)
        distance = np.linalg.norm((np.asarray(thicknesses) - best_thickness) / spans)
        distinctiveness = float(min(1.0, distance / max(np.sqrt(len(spans)), 1.0)))
        records.append(
            {
                "candidate_id": candidate_id,
                "physics_status": str(certified.certificate["status"]),
                "target_attainment": validation.get("target_attainment", {}),
                "robustness_score": robustness,
                "simplicity_score": simplicity,
                "distinctiveness_score": distinctiveness,
                "certificate_id": certified.certificate.get("certificate_id"),
                "artifact_ids": [
                    str((candidate_dir / "INDEPENDENT_VALIDATION.json").relative_to(output_dir)),
                    str((candidate_dir / "PHYSICS_ACCEPTANCE_CERTIFICATE.json").relative_to(output_dir)),
                    str((candidate_dir / "SIMULATION_RESULT.json").relative_to(output_dir)),
                ],
                "metadata": {
                    "source": item.get("source"),
                    "thicknesses_nm": thicknesses,
                    "objective_loss": item.get("objective_loss"),
                    "robustness_audit": robustness_audit,
                    "simplicity_definition": "layer_count_with_quantized_manufacturing_bonus",
                    "distinctiveness_definition": "normalized_distance_from_best_objective_candidate",
                },
            }
        )
    admissible = [
        row
        for row in records
        if row["physics_status"] in {"physically_valid", "physically_valid_with_limits"}
    ]
    if not admissible:
        payload = {
            "status": "no_physically_admissible_candidate",
            "candidates": records,
            "selected_roles": {},
        }
    else:
        best = min(
            admissible,
            key=lambda row: float(row["metadata"].get("objective_loss") or float("inf")),
        )
        robust = max(admissible, key=lambda row: float(row["robustness_score"]))
        simple = max(admissible, key=lambda row: float(row["simplicity_score"]))
        distinctive = max(admissible, key=lambda row: float(row["distinctiveness_score"]))
        payload = {
            "status": "completed",
            "selection_policy": (
                "Physics admission is mandatory; objective performance, robustness, "
                "manufacturability, and structural distinctiveness are reported as separate roles."
            ),
            "candidates": records,
            "selected_roles": {
                "best_performance": best["candidate_id"],
                "most_robust": robust["candidate_id"],
                "easiest_to_manufacture": simple["candidate_id"],
                "structurally_distinctive": distinctive["candidate_id"],
            },
        }
    (output_dir / "DESIGN_PORTFOLIO.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def _write_spectra(path: Path, result: object) -> None:
    channels = result.channels
    channel_names = sorted(channels)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = ["wavelength_nm"]
        for key in channel_names:
            for observable in ("R", "T", "A", "E_system"):
                if observable in channels[key]:
                    header.append("%s|%s" % (key, observable))
        writer.writerow(header)
        for index, wavelength in enumerate(result.wavelengths_nm):
            row = [float(wavelength)]
            for key in channel_names:
                for observable in ("R", "T", "A", "E_system"):
                    if observable in channels[key]:
                        row.append(float(channels[key][observable][index]))
            writer.writerow(row)


def _plot(path: Path, result: object) -> bool:
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
        axis.plot(result.wavelengths_nm, values["R"], label="R %s" % key)
    axis.set(xlabel="Wavelength (nm)", ylabel="Reflectance", ylim=(-0.02, 1.02))
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--physics-python", default=None)
    parser.add_argument("--skip-certificate", action="store_true")
    parser.add_argument("--convergence-max-refinements", type=int, default=3)
    parser.add_argument("--convergence-pointwise-tolerance", type=float, default=5e-3)
    parser.add_argument("--convergence-integral-tolerance", type=float, default=1e-3)
    parser.add_argument("--portfolio-max-candidates", type=int, default=6)
    args = parser.parse_args()
    started = time.perf_counter()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mode, task = load_task(args.input)
    write_normalized_task(output_dir / "NORMALIZED_TASK.json", mode, task)
    if mode == "optimize":
        try:
            import torch  # noqa: F401
        except ImportError:
            if os.environ.get("VERITMM_PHYSICS_CHILD") == "1":
                raise
            from tmm_engine.physics_runtime import discover_physics_python

            physics_python = discover_physics_python(args.physics_python)
            environment = dict(os.environ)
            environment["VERITMM_PHYSICS_CHILD"] = "1"
            environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
            command = [
                str(physics_python),
                str(Path(__file__).resolve()),
                "--input",
                str(Path(args.input).resolve()),
                "--output-dir",
                str(output_dir),
                "--device",
                args.device,
            ]
            if args.skip_certificate:
                command.append("--skip-certificate")
            command.extend(
                [
                    "--convergence-max-refinements",
                    str(args.convergence_max_refinements),
                    "--convergence-pointwise-tolerance",
                    str(args.convergence_pointwise_tolerance),
                    "--convergence-integral-tolerance",
                    str(args.convergence_integral_tolerance),
                    "--portfolio-max-candidates",
                    str(args.portfolio_max_candidates),
                ]
            )
            return int(subprocess.run(command, env=environment, check=False).returncode)
    registry = MaterialRegistry()
    workbench = TMMWorkbench(registry)
    manifest = {
        "mode": mode,
        "status": "running",
        "input": str(Path(args.input).resolve()),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
    }

    try:
        certificate_settings = AcceptanceSettings(
            require_spectral_convergence=not args.skip_certificate,
            require_independent_solver=not args.skip_certificate,
            convergence=SpectralConvergenceSettings(
                max_refinements=args.convergence_max_refinements,
                max_pointwise_deviation=args.convergence_pointwise_tolerance,
                max_integral_deviation=args.convergence_integral_tolerance,
            ),
        )
        if mode == "simulate":
            certified = certify_simulation(workbench, task, certificate_settings)
            (output_dir / "PHYSICS_ACCEPTANCE_CERTIFICATE.json").write_text(
                json.dumps(certified.certificate, indent=2), encoding="utf-8"
            )
            if certified.result is None or not certified.certificate["accepted"]:
                raise RuntimeError("physics acceptance certificate rejected the simulation")
            forward = certified.result
            (output_dir / "SIMULATION_RESULT.json").write_text(
                json.dumps(forward.to_dict(), indent=2), encoding="utf-8"
            )
        else:
            from tmm_engine.optimization import DifferentiableThicknessOptimizer

            optimizer = DifferentiableThicknessOptimizer(registry, device=args.device)
            import torch

            manifest["torch_version"] = torch.__version__
            optimization = optimizer.optimize(task)
            simulation, forward, validation = optimizer.validate_result(task, optimization, workbench)
            (output_dir / "OPTIMIZATION_RESULT.json").write_text(
                json.dumps(optimization.to_dict(), indent=2), encoding="utf-8"
            )
            (output_dir / "INDEPENDENT_VALIDATION.json").write_text(
                json.dumps(validation, indent=2), encoding="utf-8"
            )
            if validation["status"] != "passed":
                raise RuntimeError("independent physics validation failed")
            certified = certify_simulation(workbench, simulation, certificate_settings)
            (output_dir / "PHYSICS_ACCEPTANCE_CERTIFICATE.json").write_text(
                json.dumps(certified.certificate, indent=2), encoding="utf-8"
            )
            if certified.result is None or not certified.certificate["accepted"]:
                raise RuntimeError("physics acceptance certificate rejected the optimized design")
            forward = certified.result
            portfolio = _write_optimization_portfolio(
                output_dir,
                task,
                optimization,
                optimizer,
                workbench,
                certificate_settings,
                args.portfolio_max_candidates,
            )
            manifest["portfolio_candidate_count"] = len(portfolio.get("candidates", []))
            manifest["portfolio_roles"] = portfolio.get("selected_roles", {})
        _write_spectra(output_dir / "SPECTRA.csv", forward)
        plot_written = _plot(output_dir / "SPECTRA.png", forward)
        manifest.update(
            {
                "status": "completed",
                "solver": forward.solver,
                "physics_audit": forward.audit,
                "acceptance_certificate_status": certified.certificate["status"],
                "acceptance_certificate_id": certified.certificate["certificate_id"],
                "plot_written": plot_written,
                "wall_seconds": time.perf_counter() - started,
            }
        )
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "wall_seconds": time.perf_counter() - started,
            }
        )
        (output_dir / "RUN_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        raise
    (output_dir / "RUN_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
