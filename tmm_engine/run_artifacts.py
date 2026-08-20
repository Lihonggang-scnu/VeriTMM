"""Compact machine-facing summaries and auditable artifact indexes."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .protocol.models import PROTOCOL_VERSION

RESULT_SUMMARY_SCHEMA_VERSION = "veritmm-result-summary-v1"
RUN_RESULT_SCHEMA_VERSION = "veritmm-run-result-v1"


_ARTIFACT_KINDS = {
    "NORMALIZED_TASK.json": ("normalized_task", "veritmm-normalized-task-v1"),
    "SIMULATION_RESULT.json": ("simulation_result", "veritmm-simulation-result-v1"),
    "OPTIMIZATION_RESULT.json": ("optimization_result", "veritmm-optimization-result-v1"),
    "INDEPENDENT_VALIDATION.json": (
        "independent_validation",
        "veritmm-independent-validation-v1",
    ),
    "PHYSICS_ACCEPTANCE_CERTIFICATE.json": (
        "physics_certificate",
        "physics-acceptance-certificate-v1",
    ),
    "PREFLIGHT_REPORT.json": ("preflight_report", "veritmm-preflight-v1"),
    "DESIGN_PORTFOLIO.json": ("design_portfolio", "veritmm-design-portfolio-v1"),
    "SPECTRA.csv": ("spectrum_table", "veritmm-spectrum-csv-v1"),
    "SPECTRA.png": ("spectrum_plot", "veritmm-spectrum-plot-v1"),
    "SPECTRA_PLOT_SKIPPED.txt": ("plot_diagnostic", "veritmm-text-diagnostic-v1"),
    "RUN_MANIFEST.json": ("legacy_run_manifest", "veritmm-run-manifest-v1"),
    "RESULT_SUMMARY.json": ("result_summary", RESULT_SUMMARY_SCHEMA_VERSION),
    "SWEEP_RESULT.json": ("sweep_result", "veritmm-sweep-result-v1"),
    "SWEEP_TABLE.csv": ("sweep_table", "veritmm-sweep-table-v1"),
    "SENSITIVITY_RESULT.json": (
        "sensitivity_result",
        "veritmm-sensitivity-result-v1",
    ),
    "TOLERANCE_RESULT.json": ("tolerance_result", "veritmm-tolerance-result-v1"),
    "ROBUSTNESS_REPORT.json": ("robustness_report", "veritmm-robustness-report-v1"),
}


def canonical_json_bytes(payload: Any) -> bytes:
    """Return the canonical UTF-8 representation used for stable task hashes."""

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def stable_payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
                default=str,
            ),
            encoding="utf-8",
        )
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def prepare_output_directory(output_dir: str | Path) -> Path:
    """Remove stale top-level protocol artifacts before a new run.

    Only paths owned by VeriTMM are removed. User files and unrelated
    subdirectories are never traversed or deleted.
    """

    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name in (*_ARTIFACT_KINDS, "RUN_RESULT.json"):
        path = root / name
        if path.is_file():
            path.unlink()
    candidate_root = root / "candidates"
    if candidate_root.is_dir() and candidate_root.resolve().parent == root:
        shutil.rmtree(candidate_root)
    return root


def _threshold_bands(
    wavelengths_nm: np.ndarray,
    values: np.ndarray,
    threshold: float,
    *,
    max_intervals: int = 20,
) -> dict[str, Any]:
    mask = np.asarray(values, dtype=np.float64) >= float(threshold)
    if not np.any(mask):
        return {"intervals_nm": [], "total_interval_count": 0, "truncated": False}
    starts = np.flatnonzero(mask & np.concatenate(([True], ~mask[:-1])))
    stops = np.flatnonzero(mask & np.concatenate((~mask[1:], [True])))
    intervals = [
        [float(wavelengths_nm[start]), float(wavelengths_nm[stop])]
        for start, stop in zip(starts, stops)
    ]
    return {
        "intervals_nm": intervals[:max_intervals],
        "total_interval_count": len(intervals),
        "truncated": len(intervals) > max_intervals,
    }


def build_result_summary(
    *,
    mode: str,
    forward: Any | None,
    certificate: Mapping[str, Any] | None,
    optimization: Any | None = None,
    warnings: Sequence[Any] = (),
    run_id: str | None = None,
    task_sha256: str | None = None,
    run_status: str | None = None,
) -> dict[str, Any]:
    """Build a compact summary so an agent need not ingest full spectra."""

    certificate = dict(certificate or {})
    summary: dict[str, Any] = {
        "schema_version": RESULT_SUMMARY_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "run_id": run_id,
        "task_sha256": task_sha256,
        "task_hash_scope": "normalized_operation_wrapper",
        "mode": str(mode),
        "status": str(run_status or certificate.get("status") or "not_certified"),
        "physics": {
            "accepted": bool(certificate.get("accepted", False)),
            "status": certificate.get("status"),
            "certificate_id": certificate.get("certificate_id"),
            "convergence_status": (certificate.get("spectral_convergence") or {}).get(
                "status"
            ),
            "cross_solver_status": (
                certificate.get("independent_solver_check") or {}
            ).get("status"),
            "cross_solver_max_absolute_difference": (
                certificate.get("independent_solver_check") or {}
            ).get("maximum_absolute_difference"),
            "energy_conservation_max_abs_error": (
                certificate.get("physics_audit") or {}
            ).get("energy_conservation_max_abs_error"),
        },
        "spectral_features": {},
        "materials": [],
        "warnings": list(warnings),
    }
    if forward is not None:
        wavelengths = np.asarray(forward.wavelengths_nm, dtype=np.float64)
        channels: dict[str, Any] = {}
        for channel_name, channel in sorted(forward.channels.items()):
            channel_summary: dict[str, Any] = {}
            for observable in ("R", "T", "A", "E_system"):
                if observable not in channel:
                    continue
                values = np.asarray(channel[observable], dtype=np.float64)
                finite = np.flatnonzero(np.isfinite(values))
                if finite.size == 0:
                    channel_summary[observable] = {
                        "finite": False,
                        "minimum": None,
                        "minimum_wavelength_nm": None,
                        "maximum": None,
                        "maximum_wavelength_nm": None,
                        "mean": None,
                        "bands_ge_0_90_nm": {
                            "intervals_nm": [],
                            "total_interval_count": 0,
                            "truncated": False,
                        },
                        "bands_ge_0_99_nm": {
                            "intervals_nm": [],
                            "total_interval_count": 0,
                            "truncated": False,
                        },
                    }
                    continue
                finite_values = values[finite]
                peak_index = int(finite[int(np.argmax(finite_values))])
                minimum_index = int(finite[int(np.argmin(finite_values))])
                channel_summary[observable] = {
                    "finite": bool(finite.size == values.size),
                    "minimum": float(values[minimum_index]),
                    "minimum_wavelength_nm": float(wavelengths[minimum_index]),
                    "maximum": float(values[peak_index]),
                    "maximum_wavelength_nm": float(wavelengths[peak_index]),
                    "mean": float(np.mean(finite_values)),
                    "bands_ge_0_90_nm": _threshold_bands(wavelengths, values, 0.90),
                    "bands_ge_0_99_nm": _threshold_bands(wavelengths, values, 0.99),
                }
            channels[str(channel_name)] = channel_summary
        summary["spectral_features"] = channels
        seen: set[tuple[Any, ...]] = set()
        materials = []
        for item in getattr(forward, "material_provenance", ()):
            record = dict(item)
            compact = {
                key: record.get(key)
                for key in (
                    "stack_position",
                    "provider",
                    "dataset_id",
                    "material",
                    "canonical_name",
                    "source",
                    "range_um",
                    "extrapolated",
                )
                if record.get(key) is not None
            }
            marker = tuple((key, json.dumps(value, sort_keys=True, default=str)) for key, value in sorted(compact.items()))
            if marker not in seen:
                seen.add(marker)
                materials.append(compact)
        summary["materials"] = materials
    if optimization is not None:
        payload = optimization.to_dict() if hasattr(optimization, "to_dict") else dict(optimization)
        summary["optimization"] = {
            key: payload.get(key)
            for key in (
                "status",
                "initial_loss",
                "optimized_loss",
                "quantized_loss",
                "optimized_thicknesses_nm",
                "quantized_thicknesses_nm",
            )
            if key in payload
        }
    return summary


def index_artifacts(output_dir: str | Path) -> list[dict[str, Any]]:
    """Index known artifacts without including RUN_RESULT.json itself."""

    root = Path(output_dir)
    records: list[dict[str, Any]] = []
    for name, (kind, schema_version) in sorted(_ARTIFACT_KINDS.items()):
        path = root / name
        if not path.is_file():
            continue
        records.append(
            {
                "kind": kind,
                "path": path.relative_to(root).as_posix(),
                "schema_version": schema_version,
                "sha256": file_sha256(path),
                "size_bytes": int(path.stat().st_size),
            }
        )
    candidate_root = root / "candidates"
    if candidate_root.is_dir():
        for path in sorted(candidate_root.rglob("*")):
            if not path.is_file() or path.name not in _ARTIFACT_KINDS:
                continue
            kind, schema_version = _ARTIFACT_KINDS[path.name]
            records.append(
                {
                    "kind": f"candidate_{kind}",
                    "path": path.relative_to(root).as_posix(),
                    "schema_version": schema_version,
                    "sha256": file_sha256(path),
                    "size_bytes": int(path.stat().st_size),
                }
            )
    children_root = root / "children"
    if children_root.is_dir():
        for path in sorted(children_root.rglob("*")):
            if not path.is_file() or path.name not in (*_ARTIFACT_KINDS, "RUN_RESULT.json"):
                continue
            if path.name == "RUN_RESULT.json":
                kind, schema_version = "child_run_result", RUN_RESULT_SCHEMA_VERSION
            else:
                base_kind, schema_version = _ARTIFACT_KINDS[path.name]
                kind = f"child_{base_kind}"
            records.append(
                {
                    "kind": kind,
                    "path": path.relative_to(root).as_posix(),
                    "schema_version": schema_version,
                    "sha256": file_sha256(path),
                    "size_bytes": int(path.stat().st_size),
                }
            )
    return records


def write_run_result(
    output_dir: str | Path,
    *,
    operation: str,
    task_sha256: str | None,
    status: str,
    ok: bool,
    summary: Mapping[str, Any] | None = None,
    warnings: Iterable[Any] = (),
    failures: Iterable[Mapping[str, Any]] = (),
    certificate_id: str | None = None,
    run_id: str | None = None,
    input_sha256: str | None = None,
) -> dict[str, Any]:
    """Write the single first-read artifact for an AI caller."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    failure_list = [dict(item) for item in failures]
    payload = {
        "schema_version": RUN_RESULT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "ok": bool(ok),
        "run_id": run_id or f"run_{uuid.uuid4().hex}",
        "task_sha256": task_sha256,
        "task_hash_scope": "normalized_operation_wrapper",
        "input_sha256": input_sha256,
        "operation": str(operation),
        "status": str(status),
        "summary": dict(summary or {}),
        "warnings": list(warnings),
        "failures": failure_list,
        "certificate_id": certificate_id,
        "artifacts": index_artifacts(root),
        "next_machine_actions": [
            action
            for failure in failure_list
            for action in list(failure.get("actions") or [])
        ],
        "cache_hit": False,
        "source_run_id": None,
        "artifact_provenance": None,
    }
    write_json(root / "RUN_RESULT.json", payload)
    return payload


__all__ = [
    "PROTOCOL_VERSION",
    "RESULT_SUMMARY_SCHEMA_VERSION",
    "RUN_RESULT_SCHEMA_VERSION",
    "build_result_summary",
    "canonical_json_bytes",
    "file_sha256",
    "index_artifacts",
    "prepare_output_directory",
    "stable_payload_sha256",
    "write_json",
    "write_run_result",
]
