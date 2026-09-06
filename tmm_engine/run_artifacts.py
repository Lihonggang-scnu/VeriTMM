"""Compact machine-facing summaries and auditable artifact indexes."""

from __future__ import annotations

import json
import shutil
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .archive.schema_registry import ARCHIVE_SCHEMA_VERSION
from .hashing import IDENTITY_SCHEME, canonical_json_dumps, file_sha256, stable_sha256
from .protocol.models import PROTOCOL_VERSION
from .protocol.responses import (
    DEFAULT_RESPONSE_DETAIL,
    RESPONSE_CONTEXT_FILENAME,
    RESPONSE_CONTEXT_SCHEMA_VERSION,
    RESPONSE_RETENTION_SCHEMA_VERSION,
    canonical_response_context,
    normalize_response_detail,
    project_response,
    response_profile,
    response_projection_source,
    validate_artifact_references,
    validate_projected_response,
)
from .reproducibility import reproducibility_block

RESULT_SUMMARY_SCHEMA_VERSION = "veritmm-result-summary-v2"
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
    "VERIFICATION_EVIDENCE.json": (
        "verification_evidence",
        "veritmm-verification-evidence-v1",
    ),
    "VERIFICATION_POLICY.json": (
        "verification_policy",
        "veritmm-verification-policy-v1",
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
    "TOLERANCE_RESULT.json": ("tolerance_result", "veritmm-tolerance-result-v2"),
    "ROBUSTNESS_REPORT.json": ("robustness_report", "veritmm-robustness-report-v2"),
    "BATCH_MANIFEST.json": (
        "research_batch_manifest",
        "veritmm-research-batch-manifest-v1",
    ),
    "BATCH_INDEX.jsonl": (
        "research_batch_index",
        "veritmm-research-batch-index-v1",
    ),
    "DATASET_MANIFEST.json": (
        "research_dataset_manifest",
        "veritmm-dataset-manifest-v1",
    ),
    "DATASET_INDEX.jsonl": (
        "research_dataset_index",
        "veritmm-dataset-index-v1",
    ),
    RESPONSE_CONTEXT_FILENAME: (
        "response_context",
        RESPONSE_CONTEXT_SCHEMA_VERSION,
    ),
}


class ResponseContextValidationError(ValueError):
    """Raised when a persisted canonical response source fails integrity checks."""


class ResponseDetailUnavailableError(ValueError):
    """Raised when a legacy compact run cannot supply a richer profile."""

    code = "response_detail_unavailable"

    def __init__(self, requested_detail: str, reason: str) -> None:
        self.requested_detail = normalize_response_detail(requested_detail)
        self.reason = str(reason)
        super().__init__(
            f"response detail {self.requested_detail!r} is unavailable: {self.reason}"
        )

    def to_response(self, *, operation: str = "inspect") -> dict[str, Any]:
        return {
            "schema_version": "veritmm-detail-unavailable-v1",
            "ok": False,
            "operation": operation,
            "status": "detail_unavailable",
            "error": {
                "code": self.code,
                "message": str(self),
                "recoverable": True,
                "requested_detail": self.requested_detail,
                "reason": self.reason,
                "actions": [
                    {
                        "action_id": "rerun_with_response_context",
                        "action_type": "rerun_required",
                        "description": (
                            "Re-run the task with VeriTMM v0.5.2+ to persist "
                            "RESPONSE_CONTEXT.json, then inspect again."
                        ),
                        "safety": "safe",
                    }
                ],
            },
        }


def stable_payload_sha256(payload: Any) -> str:
    """Stable content hash; the canonical implementation lives in hashing."""

    return stable_sha256(payload)


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


def prepare_output_directory(
    output_dir: str | Path, *, preserve_sweep_children: bool = False
) -> Path:
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
    children_root = root / "children"
    if (
        not preserve_sweep_children
        and children_root.is_dir()
        and children_root.resolve().parent == root
    ):
        shutil.rmtree(children_root)
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
    evidence_coverage = certificate.get("evidence_coverage")
    summary: dict[str, Any] = {
        "schema_version": RESULT_SUMMARY_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "run_id": run_id,
        "task_sha256": task_sha256,
        "identity_scheme": IDENTITY_SCHEME,
        "task_hash_scope": "normalized_operation_wrapper",
        "mode": str(mode),
        "status": str(run_status or certificate.get("status") or "not_certified"),
        "archive_schema_version": ARCHIVE_SCHEMA_VERSION,
        "evidence_coverage": evidence_coverage,
        "physics": {
            "accepted": bool(certificate.get("accepted", False)),
            "status": certificate.get("status"),
            "certificate_id": certificate.get("certificate_id"),
            "evidence_coverage": evidence_coverage,
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
            marker = tuple((key, canonical_json_dumps(value)) for key, value in sorted(compact.items()))
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


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _validate_identity_binding(
    payload: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    label: str,
) -> None:
    for key in ("run_id", "task_sha256"):
        if payload.get(key) != result.get(key):
            raise ResponseContextValidationError(
                f"{label}.{key} does not match RUN_RESULT.{key}"
            )


def validate_response_context(
    context: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    root: str | Path,
) -> bool:
    """Validate schema, run/task binding, retention, and the no-self-ref rule."""

    base = Path(root).resolve()
    if context.get("schema_version") != RESPONSE_CONTEXT_SCHEMA_VERSION:
        raise ResponseContextValidationError(
            "RESPONSE_CONTEXT.json has an unsupported schema_version"
        )
    _validate_identity_binding(context, result, label="RESPONSE_CONTEXT")
    retention = context.get("retention")
    if not isinstance(retention, Mapping):
        raise ResponseContextValidationError(
            "RESPONSE_CONTEXT.json retention must be an object"
        )
    if (
        retention.get("schema_version") != RESPONSE_RETENTION_SCHEMA_VERSION
        or retention.get("semantics") != "bounded_retained_metadata"
        or retention.get("bounded") is not True
        or retention.get("full_profile_scope")
        != "full_of_retained_bounded_metadata"
    ):
        raise ResponseContextValidationError(
            "RESPONSE_CONTEXT.json retention semantics are invalid"
        )
    for key in ("omitted_fields", "truncated_fields"):
        counts = retention.get(key)
        if not isinstance(counts, Mapping) or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in counts.values()
        ):
            raise ResponseContextValidationError(
                f"RESPONSE_CONTEXT.json retention.{key} is invalid"
            )
    source = context.get("source")
    if not isinstance(source, Mapping):
        raise ResponseContextValidationError(
            "RESPONSE_CONTEXT.json source must contain an object"
        )
    _validate_identity_binding(source, result, label="RESPONSE_CONTEXT.source")
    source_summary = source.get("summary")
    if not isinstance(source_summary, Mapping):
        raise ResponseContextValidationError(
            "RESPONSE_CONTEXT.json source.summary must contain an object"
        )
    _validate_identity_binding(
        source_summary,
        result,
        label="RESPONSE_CONTEXT.source.summary",
    )
    if response_profile(source) is not None:
        raise ResponseContextValidationError(
            "RESPONSE_CONTEXT.json source must be unprojected"
        )
    source_refs = source.get("artifacts", [])
    if not isinstance(source_refs, list):
        raise ResponseContextValidationError(
            "RESPONSE_CONTEXT.json source.artifacts must be a list"
        )
    try:
        validate_artifact_references(source_refs, root=base)
    except ValueError as exc:
        raise ResponseContextValidationError(str(exc)) from exc
    if any(str(item.get("path")) == RESPONSE_CONTEXT_FILENAME for item in source_refs):
        raise ResponseContextValidationError(
            "RESPONSE_CONTEXT.json must not reference itself"
        )

    result_refs = result.get("artifacts", [])
    if not isinstance(result_refs, list):
        raise ResponseContextValidationError("RUN_RESULT.artifacts must be a list")
    context_refs = [
        item
        for item in result_refs
        if isinstance(item, Mapping)
        and str(item.get("path")) == RESPONSE_CONTEXT_FILENAME
    ]
    if len(context_refs) != 1:
        raise ResponseContextValidationError(
            "RUN_RESULT must contain exactly one RESPONSE_CONTEXT.json artifact reference"
        )
    reference = context_refs[0]
    if (
        reference.get("kind") != "response_context"
        or reference.get("schema_version") != RESPONSE_CONTEXT_SCHEMA_VERSION
    ):
        raise ResponseContextValidationError(
            "RUN_RESULT response_context artifact reference is invalid"
        )
    return True


def _load_validated_run(
    output_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    root = Path(output_dir).resolve()
    result_path = root / "RUN_RESULT.json"
    if not result_path.is_file():
        return {}, None
    result = _read_json_object(result_path, "RUN_RESULT.json")
    try:
        validate_artifact_references(result, root=root)
    except ValueError as exc:
        raise ValueError(f"RUN_RESULT artifact integrity failure: {exc}") from exc
    summary = result.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("RUN_RESULT.summary must contain an object")
    _validate_identity_binding(summary, result, label="RUN_RESULT.summary")
    summary_path = root / "RESULT_SUMMARY.json"
    if summary_path.is_file():
        standalone = _read_json_object(summary_path, "RESULT_SUMMARY.json")
        _validate_identity_binding(standalone, result, label="RESULT_SUMMARY")
    context_path = root / RESPONSE_CONTEXT_FILENAME
    context = (
        _read_json_object(context_path, RESPONSE_CONTEXT_FILENAME)
        if context_path.is_file()
        else None
    )
    if context is not None:
        validate_response_context(context, result, root=root)
    return result, context


def validate_run_artifact_integrity(output_dir: str | Path) -> bool:
    """Fail closed when persisted run references or response context are stale."""

    result, _ = _load_validated_run(output_dir)
    if not result:
        raise ValueError("RUN_RESULT.json is missing")
    existing = response_profile(result)
    if existing is not None:
        validate_projected_response(result, detail=existing)
    return True


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
    detail: str = DEFAULT_RESPONSE_DETAIL,
) -> dict[str, Any]:
    """Write the single first-read artifact for an AI caller.

    The operation-specific JSON artifacts are written by their execution
    modules before this function is called.  ``RUN_RESULT.json`` is therefore
    a projected index by default; requesting ``standard`` or ``full`` only
    changes this envelope and never removes the detailed artifacts.
    """

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    context_path = root / RESPONSE_CONTEXT_FILENAME
    if context_path.is_file():
        context_path.unlink()
    effective_run_id = run_id or f"run_{uuid.uuid4().hex}"
    summary_payload = dict(summary or {})
    summary_payload.setdefault("schema_version", RESULT_SUMMARY_SCHEMA_VERSION)
    summary_payload.setdefault("protocol_version", PROTOCOL_VERSION)
    summary_payload["run_id"] = effective_run_id
    summary_payload["task_sha256"] = task_sha256
    summary_payload.setdefault("identity_scheme", IDENTITY_SCHEME)
    summary_payload.setdefault("archive_schema_version", ARCHIVE_SCHEMA_VERSION)
    summary_payload.setdefault("task_hash_scope", "normalized_operation_wrapper")
    summary_payload.setdefault("mode", str(operation))
    summary_payload.setdefault("status", str(status))
    # RESULT_SUMMARY is normalized here so the standalone artifact, canonical
    # source, and RUN_RESULT all originate from this exact mapping.
    write_json(root / "RESULT_SUMMARY.json", summary_payload)
    failure_list = [dict(item) for item in failures]
    operation_artifacts = index_artifacts(root)
    validate_artifact_references(operation_artifacts, root=root)
    base_payload = {
        "schema_version": RUN_RESULT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "ok": bool(ok),
        "run_id": effective_run_id,
        "task_sha256": task_sha256,
        "identity_scheme": IDENTITY_SCHEME,
        "task_hash_scope": "normalized_operation_wrapper",
        "archive_schema_version": ARCHIVE_SCHEMA_VERSION,
        "input_sha256": input_sha256,
        "operation": str(operation),
        "status": str(status),
        "summary": summary_payload,
        "warnings": list(warnings),
        "failures": failure_list,
        "certificate_id": certificate_id,
        "artifacts": operation_artifacts,
        "next_machine_actions": [
            action
            for failure in failure_list
            for action in list(failure.get("actions") or [])
        ],
        "cache_hit": False,
        "source_run_id": None,
        "artifact_provenance": None,
        "reproducibility": reproducibility_block(),
    }
    context = canonical_response_context(
        base_payload, artifact_refs=operation_artifacts
    )
    write_json(context_path, context)
    artifacts = index_artifacts(root)
    validate_artifact_references(artifacts, root=root)
    payload = {**base_payload, "artifacts": artifacts}
    projected = project_response(
        response_projection_source(payload, context["retention"]),
        detail=detail,
    )
    write_json(root / "RUN_RESULT.json", projected)
    validate_run_artifact_integrity(root)
    return projected


def load_run_result(
    output_dir: str | Path,
    *,
    detail: str = DEFAULT_RESPONSE_DETAIL,
    force_context: bool = False,
) -> dict[str, Any]:
    """Load a persisted run at one profile without re-projecting compact data."""

    root = Path(output_dir).resolve()
    result, context = _load_validated_run(root)
    if not result:
        return {}
    profile = normalize_response_detail(detail)
    existing = response_profile(result)
    if existing is not None:
        validate_projected_response(result, detail=existing)
    if context is None and profile != "compact":
        raise ResponseDetailUnavailableError(
            profile,
            "this legacy run has no validated RESPONSE_CONTEXT.json canonical source",
        )
    if existing == profile and not force_context:
        return deepcopy(result)
    source_payload, retention = load_run_result_source(root, detail=profile)
    return project_response(
        response_projection_source(source_payload, retention),
        detail=profile,
    )


def load_run_result_source(
    output_dir: str | Path,
    *,
    detail: str = DEFAULT_RESPONSE_DETAIL,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load one validated unprojected source plus honest retention semantics."""

    root = Path(output_dir).resolve()
    result, context = _load_validated_run(root)
    if not result:
        return {}, {}
    profile = normalize_response_detail(detail)
    if context is not None:
        source = context.get("source")
        retention = context.get("retention")
        if not isinstance(source, Mapping) or not isinstance(retention, Mapping):
            raise ResponseContextValidationError(
                "validated response context lost its source or retention metadata"
            )
        source_payload = deepcopy(dict(source))
        source_payload["artifacts"] = deepcopy(result.get("artifacts", []))
        return source_payload, deepcopy(dict(retention))

    if profile != "compact":
        raise ResponseDetailUnavailableError(
            profile,
            "this legacy run has no validated RESPONSE_CONTEXT.json canonical source",
        )
    source_payload = deepcopy(result)
    source_payload.pop("response", None)
    source_summary = source_payload.get("summary")
    if isinstance(source_summary, dict):
        source_summary.pop("response", None)
    retention = {
        "schema_version": RESPONSE_RETENTION_SCHEMA_VERSION,
        "semantics": "legacy_compact_only",
        "bounded": True,
        "full_profile_scope": "unavailable_without_response_context",
        "large_operation_arrays_retained": False,
        "limits": {},
        "omitted_fields": {},
        "truncated_fields": {},
    }
    return source_payload, retention


__all__ = [
    "PROTOCOL_VERSION",
    "ResponseContextValidationError",
    "ResponseDetailUnavailableError",
    "RESULT_SUMMARY_SCHEMA_VERSION",
    "RUN_RESULT_SCHEMA_VERSION",
    "build_result_summary",
    "file_sha256",
    "index_artifacts",
    "load_run_result",
    "load_run_result_source",
    "prepare_output_directory",
    "stable_payload_sha256",
    "validate_response_context",
    "validate_run_artifact_integrity",
    "write_json",
    "write_run_result",
]
