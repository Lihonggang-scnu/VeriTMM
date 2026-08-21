"""Context-efficient projections for machine-facing VeriTMM responses.

The numerical and artifact-producing paths deliberately keep their existing
payloads.  This module owns the one projection policy used at the protocol
boundary so a large sweep, tolerance study, benchmark, or optimizer cannot
accidentally make a first-read response proportional to its detailed data.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Literal, TypeAlias

ResponseDetail: TypeAlias = Literal["compact", "standard", "full"]

RESPONSE_SCHEMA_VERSION = "veritmm-response-v1"
RESPONSE_CONTEXT_SCHEMA_VERSION = "veritmm-response-context-v2"
RESPONSE_RETENTION_SCHEMA_VERSION = "veritmm-response-retention-v1"
RESPONSE_CONTEXT_FILENAME = "RESPONSE_CONTEXT.json"
DEFAULT_RESPONSE_DETAIL: ResponseDetail = "compact"
RESPONSE_DETAILS: tuple[ResponseDetail, ...] = ("compact", "standard", "full")

# These limits are protocol limits, not numerical limits.  The full response
# profile keeps complete scalar/mapping metadata, but bulky scientific arrays
# remain in the existing operation artifacts for every profile.
COMPACT_TARGET_BYTES = 16 * 1024
COMPACT_MAX_BYTES = 32 * 1024
COMPACT_MAX_ITEMS = 24
COMPACT_MAX_ARTIFACT_REFS = 32
COMPACT_MAX_SPECTRAL_CHANNELS = 12
COMPACT_MAX_MATERIALS = 12
STANDARD_MAX_ITEMS = 256
STANDARD_MAX_ARTIFACT_REFS = 256
FULL_MAX_ARTIFACT_REFS = 512
CANONICAL_MAX_ITEMS = 256
CANONICAL_MAX_ARRAY_ITEMS = 64
CANONICAL_MAX_ARTIFACT_REFS = 64
CANONICAL_MAX_STRING_CHARS = 4096
CANONICAL_MAX_METADATA_PATHS = 64
LINEAGE_MAX_RECORDS = 16
SMALL_INLINE_ARRAY_ITEMS = 8
FULL_MAX_GENERIC_ARRAY_ITEMS = 64
COMPACT_MAX_STRING_CHARS = 256
MAX_RESPONSE_METADATA_PATHS = 32
MAX_COMPACT_FAILURES = 8
MAX_COMPACT_ACTIONS = 4
MAX_COMPACT_FAILURE_ACTIONS = 1
MAX_COMPACT_PATCH_ITEMS = 2
MAX_COMPACT_WARNINGS = 16

_BULKY_ARRAY_TOKENS = frozenset(
    {
        "spectra",
        "spectrum",
        "wavelength",
        "wavelengths",
        "spectralgrid",
        "samples",
        "sample",
        "history",
        "losshistory",
        "losses",
        "objectivehistory",
        "optimizationhistory",
        "optimizerhistory",
        "traininghistory",
        "perturbation",
        "perturbations",
        "offset",
        "offsets",
        "draw",
        "draws",
        "children",
        "case",
        "cases",
        "independentaudit",
        "provenance",
        "trajectory",
        "trajectories",
    }
)
_CHANNEL_ARRAY_KEYS = frozenset(
    {
        "r",
        "t",
        "a",
        "reflectance",
        "transmittance",
        "absorptance",
    }
)
_WAVELENGTH_KEYS = frozenset(
    {"wavelength", "wavelengths", "wavelengthgrid", "wavelengthsgrid"}
)
_COMPACT_RUN_RECORD_FIELDS = (
    "run_id",
    "experiment_id",
    "parent_run_id",
    "task_sha256",
    "operation",
    "status",
    "created_at",
    "completed_at",
    "protocol_version",
    "package_version",
    "certificate_id",
    "artifact_root",
    "cache_hit",
    "source_run_id",
    "tags",
)
_COMPACT_RECORD_LIST_KEYS = frozenset({"runs", "ancestors", "descendants"})
_LINEAGE_RECORD_FIELDS = ("run_id", "operation", "status", "parent_run_id", "created_at")
_COMPACT_FAILURE_FIELDS = (
    "code",
    "message",
    "recoverable",
    "suggested_solver_family",
    "severity",
    "requires_user_choice",
    "actions",
)
_COMPACT_ACTION_FIELDS = ("action_id", "action_type", "description", "safety", "patch")
_COMPACT_CACHE_PROVENANCE_FIELDS = (
    "mode",
    "source_run_id",
    "source_parent_run_id",
)
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_RETENTION_INPUT = "__response_retention__"


class ContextBudgetError(ValueError):
    """Raised when a compact response violates its deterministic budget."""


class _ProjectionState:
    def __init__(self) -> None:
        self.omitted: dict[str, int] = {}
        self.truncated: dict[str, int] = {}
        self.invalid_artifacts: int = 0
        self.artifact_backed: bool = False
        self.artifact_kinds: set[str] = set()

    def record_omitted(self, path: str, count: int = 1) -> None:
        key = _bounded_string(path or "$", limit=256)
        self.omitted[key] = self.omitted.get(key, 0) + max(1, int(count))

    def record_truncated(self, path: str, count: int) -> None:
        key = _bounded_string(path or "$", limit=256)
        self.truncated[key] = self.truncated.get(key, 0) + max(1, int(count))


def normalize_response_detail(detail: str | None) -> ResponseDetail:
    """Validate and normalize a response profile name."""

    value = DEFAULT_RESPONSE_DETAIL if detail is None else str(detail).lower()
    if value not in RESPONSE_DETAILS:
        supported = ", ".join(RESPONSE_DETAILS)
        raise ValueError(f"unknown response detail {detail!r}; expected one of: {supported}")
    return value  # type: ignore[return-value]


def _normalized_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def _has_wavelength_key(value: Mapping[str, Any]) -> bool:
    return any(
        _normalized_key(key) in _WAVELENGTH_KEYS
        or _normalized_key(key).startswith("wavelength")
        for key in value
    )


def _array_alias(
    key: Any,
    value: Any,
    parent: Mapping[str, Any] | None = None,
) -> str | None:
    """Return a stable alias for a bulky scientific array, if identifiable."""

    if not isinstance(value, (list, tuple)):
        return None
    normalized = _normalized_key(key)
    if normalized == "artifacts":
        return None
    if _provenance_key(key):
        return "provenance"
    if normalized in _CHANNEL_ARRAY_KEYS:
        # R/T/A can also be a short interval summary.  Treat it as a full
        # channel array only when it is long or paired with a wavelength grid.
        if len(value) > SMALL_INLINE_ARRAY_ITEMS or (
            parent is not None and _has_wavelength_key(parent)
        ):
            return "channel_values"
        return None
    if any(token in normalized for token in _BULKY_ARRAY_TOKENS):
        return normalized
    return None


def _forbidden_compact_array_key(
    key: Any,
    value: Any = None,
    parent: Mapping[str, Any] | None = None,
) -> bool:
    """Compatibility name for the shared all-profile bulky-array detector."""

    return _array_alias(key, value, parent) is not None


def _provenance_key(key: Any) -> bool:
    normalized = _normalized_key(key)
    return "provenance" in normalized and normalized != "artifactprovenance"


def _is_run_envelope(payload: Mapping[str, Any]) -> bool:
    required = {"run_id", "operation", "status", "summary", "artifacts"}
    return required <= set(payload)


def response_profile(payload: Mapping[str, Any]) -> ResponseDetail | None:
    """Return the already-projected profile at a response boundary, if any."""

    if not isinstance(payload, Mapping):
        return None
    metadata: Any = None
    summary = payload.get("summary")
    if isinstance(summary, Mapping):
        metadata = summary.get("response")
    if metadata is None:
        metadata = payload.get("response")
    if not isinstance(metadata, Mapping):
        return None
    if metadata.get("schema_version") != RESPONSE_SCHEMA_VERSION:
        return None
    profile = metadata.get("profile")
    if profile not in RESPONSE_DETAILS:
        raise ValueError(f"invalid projected response profile: {profile!r}")
    return profile  # type: ignore[return-value]


def is_projected_response(
    payload: Mapping[str, Any], *, detail: str | None = None
) -> bool:
    """Return whether a payload already carries the requested response profile."""

    existing = response_profile(payload)
    if existing is None:
        return False
    if detail is None:
        return True
    return existing == normalize_response_detail(detail)


def _path_label(path: str, key: Any) -> str:
    token = str(key)
    return f"{path}.{token}" if path else token


def _bounded_string(value: str, *, limit: int = COMPACT_MAX_STRING_CHARS) -> str:
    if len(value) <= limit:
        return value
    return f"{value[: limit - 20]}...[truncated]"


def _artifact_ref_shape(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    kind = value.get("kind")
    path = value.get("path")
    schema_version = value.get("schema_version")
    digest = value.get("sha256")
    size = value.get("size_bytes")
    if not isinstance(kind, str) or not kind:
        return False
    if not isinstance(path, str) or not path or "\\" in path:
        return False
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts:
        return False
    if not isinstance(schema_version, str) or not schema_version:
        return False
    if not isinstance(digest, str) or _HEX_64.fullmatch(digest) is None:
        return False
    return isinstance(size, int) and not isinstance(size, bool) and size >= 0


def validate_artifact_references(
    payload_or_refs: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    *,
    root: str | Path | None = None,
) -> bool:
    """Validate relative, hashed artifact references.

    ``root`` enables filesystem validation of the referenced path, byte size,
    and SHA-256.  Without it this validates the portable reference shape.
    """

    if isinstance(payload_or_refs, Mapping):
        refs = payload_or_refs.get("artifacts", [])
    else:
        refs = payload_or_refs
    if not isinstance(refs, Iterable) or isinstance(refs, (str, bytes, Mapping)):
        raise ValueError("artifact references must be an iterable of objects")
    base = None if root is None else Path(root).resolve()
    for reference in refs:
        if not _artifact_ref_shape(reference):
            raise ValueError(f"invalid artifact reference: {reference!r}")
        path = str(reference["path"])
        if base is None:
            continue
        candidate = (base / PurePosixPath(path)).resolve()
        try:
            candidate.relative_to(base)
        except ValueError as exc:
            raise ValueError(f"artifact reference escapes its root: {path}") from exc
        if not candidate.is_file():
            raise ValueError(f"artifact reference does not point to a file: {path}")
        if int(reference["size_bytes"]) != int(candidate.stat().st_size):
            raise ValueError(f"artifact reference size is stale: {path}")
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if digest != reference["sha256"]:
            raise ValueError(f"artifact reference hash is stale: {path}")
    return True


def _externalization_marker(
    key: Any,
    value: Any,
    detail: ResponseDetail,
    state: _ProjectionState,
    path: str,
    *,
    alias: str | None = None,
    detail_available_via_profile: bool = False,
) -> dict[str, Any]:
    """Replace a bulky field without making an unreachable artifact claim."""

    count = len(value) if isinstance(value, (Mapping, list, tuple)) else 1
    state.record_omitted(path, count)
    name = str(key)
    normalized_alias = _normalized_key(alias or name)
    supported_kinds: set[str]
    if any(token in normalized_alias for token in ("spectrum", "wavelength", "channelvalues")):
        supported_kinds = {"simulation_result", "spectrum_table"}
    elif any(token in normalized_alias for token in ("sample", "draw", "offset", "perturbation")):
        supported_kinds = {"tolerance_result", "robustness_report"}
    elif any(token in normalized_alias for token in ("history", "loss", "objective", "optimization")):
        supported_kinds = {"optimization_result", "design_portfolio", "robustness_report"}
    elif "children" in normalized_alias:
        supported_kinds = {"sweep_result", "sweep_table"}
    elif any(token in normalized_alias for token in ("case", "independentaudit")):
        supported_kinds = {"benchmark_result"}
    elif "trajectory" in normalized_alias:
        supported_kinds = {"agent_benchmark_result"}
    elif "provenance" in normalized_alias:
        supported_kinds = {"normalized_task", "preflight_report", "legacy_run_manifest"}
    else:
        supported_kinds = set()
    field_artifact_backed = bool(state.artifact_kinds & supported_kinds)
    return {
        f"{name}_count": count,
        f"{name}_artifact_backed": field_artifact_backed,
        f"{name}_detail_available_via_profile": bool(detail_available_via_profile),
    }


def _project_compact_record(
    value: Mapping[str, Any], state: _ProjectionState, path: str
) -> dict[str, Any]:
    """Keep history/list records decision-critical and drop user metadata."""

    return {
        key: _project_value(value[key], "compact", state, _path_label(path, key))
        for key in _COMPACT_RUN_RECORD_FIELDS
        if key in value
    }


def _is_lineage_payload(value: Mapping[str, Any]) -> bool:
    return (
        value.get("schema_version") == "veritmm-lineage-v1"
        and isinstance(value.get("run"), Mapping)
        and isinstance(value.get("children"), list)
    )


def _is_experiment_record_list(value: Any) -> bool:
    """Recognize store records without treating sweep child rows as lineage."""

    if not isinstance(value, list):
        return False
    records = [item for item in value if isinstance(item, Mapping)]
    if not records:
        return True
    return all(
        "run_id" in item
        and "child_run_id" not in item
        and ("operation" in item or "status" in item)
        for item in records
    )


def _project_lineage_records(
    value: list[Any],
    state: _ProjectionState,
    path: str,
    *,
    total_count: int | None = None,
) -> tuple[list[dict[str, Any]], int, int]:
    total = max(len(value), int(total_count or 0))
    selected = value[:LINEAGE_MAX_RECORDS]
    truncated = max(0, total - len(selected))
    if truncated:
        state.record_truncated(path, truncated)
    records = [
        {
            key: copy.deepcopy(item[key])
            for key in _LINEAGE_RECORD_FIELDS
            if key in item
        }
        for item in selected
        if isinstance(item, Mapping)
    ]
    return records, total, truncated


def _project_compact_spectral_features(
    value: Mapping[str, Any], state: _ProjectionState, path: str
) -> dict[str, Any]:
    """Keep extrema and means while externalizing interval enumerations."""

    result: dict[str, Any] = {}
    channels = sorted(value.items(), key=lambda item: str(item[0]))
    selected = channels[:COMPACT_MAX_SPECTRAL_CHANNELS]
    if len(channels) > len(selected):
        state.record_truncated(path, len(channels) - len(selected))
    for channel_name, observables in selected:
        if not isinstance(observables, Mapping):
            continue
        channel_result: dict[str, Any] = {}
        for observable, metrics in observables.items():
            if not isinstance(metrics, Mapping):
                continue
            metric_result = {
                key: copy.deepcopy(metrics[key])
                for key in (
                    "finite",
                    "minimum",
                    "minimum_wavelength_nm",
                    "maximum",
                    "maximum_wavelength_nm",
                    "mean",
                )
                if key in metrics
            }
            for band_key in ("bands_ge_0_90_nm", "bands_ge_0_99_nm"):
                band = metrics.get(band_key)
                if not isinstance(band, Mapping):
                    continue
                intervals = band.get("intervals_nm")
                if isinstance(intervals, list) and intervals:
                    state.record_omitted(
                        f"{path}.{channel_name}.{observable}.{band_key}.intervals_nm",
                        len(intervals),
                    )
                metric_result[band_key] = {
                    key: copy.deepcopy(band[key])
                    for key in ("total_interval_count", "truncated")
                    if key in band
                }
            channel_result[str(observable)] = metric_result
        result[str(channel_name)] = channel_result
    if len(channels) > len(selected):
        result["channel_count"] = len(channels)
        result["included_channel_count"] = len(selected)
    return result


def _project_compact_materials(
    value: list[Any], state: _ProjectionState, path: str
) -> list[dict[str, Any]]:
    """Collapse repeated layer-level provenance into unique material datasets."""

    fields = (
        "provider",
        "dataset_id",
        "material",
        "canonical_name",
        "source",
        "range_um",
        "extrapolated",
    )
    grouped: dict[str, dict[str, Any]] = {}
    for item in value:
        if not isinstance(item, Mapping):
            continue
        compact = {key: copy.deepcopy(item[key]) for key in fields if key in item}
        marker = json.dumps(compact, sort_keys=True, ensure_ascii=False, default=str)
        record = grouped.setdefault(marker, {**compact, "stack_occurrence_count": 0})
        record["stack_occurrence_count"] += 1
    records = list(grouped.values())
    selected = records[:COMPACT_MAX_MATERIALS]
    if len(records) > len(selected):
        state.record_truncated(path, len(records) - len(selected))
    if any(isinstance(item, Mapping) and "stack_position" in item for item in value):
        state.record_omitted(f"{path}.stack_position", len(value))
    return selected


def _project_action(value: Mapping[str, Any], state: _ProjectionState, path: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in _COMPACT_ACTION_FIELDS:
        if key not in value:
            continue
        if key == "patch":
            patch = value.get(key)
            if isinstance(patch, list):
                result[key] = _project_list(
                    patch,
                    "compact",
                    state,
                    _path_label(path, key),
                    MAX_COMPACT_PATCH_ITEMS,
                )
            else:
                result[key] = copy.deepcopy(patch)
        elif key == "description" and isinstance(value[key], str):
            result[key] = _bounded_string(value[key], limit=256)
        else:
            result[key] = _project_value(value[key], "compact", state, _path_label(path, key))
    return result


def _project_failure(value: Mapping[str, Any], state: _ProjectionState, path: str) -> dict[str, Any]:
    """Keep typed navigation while moving diagnostic context to artifacts."""

    result: dict[str, Any] = {}
    for key in _COMPACT_FAILURE_FIELDS:
        if key not in value:
            continue
        if key == "actions":
            actions = value.get(key)
            if isinstance(actions, list):
                result[key] = [
                    _project_action(item, state, _path_label(path, "actions"))
                    if isinstance(item, Mapping)
                    else _project_value(item, "compact", state, _path_label(path, "actions"))
                    for item in actions[:MAX_COMPACT_FAILURE_ACTIONS]
                ]
                if len(actions) > MAX_COMPACT_FAILURE_ACTIONS:
                    state.record_truncated(
                        _path_label(path, key),
                        len(actions) - MAX_COMPACT_FAILURE_ACTIONS,
                    )
            continue
        if key == "message" and isinstance(value[key], str):
            result[key] = _bounded_string(value[key], limit=384)
        else:
            result[key] = _project_value(
                value[key], "compact", state, _path_label(path, key)
            )
    if "context" in value:
        state.record_omitted(_path_label(path, "context"))
    return result


def _project_warning(value: Any, state: _ProjectionState, path: str) -> Any:
    if not isinstance(value, Mapping):
        return _project_value(value, "compact", state, path)
    result: dict[str, Any] = {}
    for key in ("code", "severity", "message"):
        if key in value:
            result[key] = _project_value(value[key], "compact", state, _path_label(path, key))
    if "context" in value:
        state.record_omitted(_path_label(path, "context"))
    return result


def _project_cache_provenance(value: Any, state: _ProjectionState, path: str) -> Any:
    if not isinstance(value, Mapping):
        if isinstance(value, list):
            state.record_omitted(path, len(value))
            return None
        return _project_value(value, "compact", state, path)
    return {
        key: _project_value(value[key], "compact", state, _path_label(path, key))
        for key in _COMPACT_CACHE_PROVENANCE_FIELDS
        if key in value
    }


def _project_artifacts(
    value: Any,
    detail: ResponseDetail,
    state: _ProjectionState,
    path: str,
) -> Any:
    if not isinstance(value, list):
        raise ValueError("artifacts must be a list of artifact reference objects")
    validate_artifact_references(value)
    limit = {
        "compact": COMPACT_MAX_ARTIFACT_REFS,
        "standard": STANDARD_MAX_ARTIFACT_REFS,
        "full": None,
    }[detail]
    if limit is None:
        limit = FULL_MAX_ARTIFACT_REFS
    records = [item for item in value if isinstance(item, Mapping)]
    # Root artifacts are the stable progressive-disclosure entry points.  A
    # child/candidate artifact is discoverable through its parent artifact and
    # should not make RUN_RESULT grow with study size.
    root_records = [item for item in records if "/" not in str(item.get("path", ""))]
    supplemental_records = [
        item
        for item in records
        if "/" in str(item.get("path", ""))
        and not str(item.get("path", "")).startswith("children/")
    ]
    selected = (root_records + supplemental_records) or records
    selected = selected[:limit]
    omitted = max(0, len(records) - len(selected))
    if omitted:
        state.record_truncated(path, omitted)
    return [copy.deepcopy(item) for item in selected]


def _project_list(
    value: list[Any],
    detail: ResponseDetail,
    state: _ProjectionState,
    path: str,
    explicit_limit: int | None = None,
) -> list[Any]:
    if detail == "full" and explicit_limit is None:
        return copy.deepcopy(value)
    limit = explicit_limit or (COMPACT_MAX_ITEMS if detail == "compact" else STANDARD_MAX_ITEMS)
    selected = value[:limit]
    if len(value) > limit:
        state.record_truncated(path, len(value) - limit)
    return [
        _project_value(item, detail, state, f"{path}[{index}]")
        for index, item in enumerate(selected)
    ]


def _project_mapping(
    value: Mapping[str, Any], detail: ResponseDetail, state: _ProjectionState, path: str
) -> dict[str, Any]:
    items = [
        (key, item)
        for key, item in value.items()
        if str(key) != _SOURCE_RETENTION_INPUT
    ]
    if detail != "full":
        limit = COMPACT_MAX_ITEMS if detail == "compact" else STANDARD_MAX_ITEMS
    else:
        limit = None
    lineage_payload = _is_lineage_payload(value)
    if limit is not None and len(items) > limit:
        priority = {
            "schema_version",
            "protocol_version",
            "ok",
            "status",
            "run_id",
            "operation",
            "summary",
            "physics",
            "failures",
            "artifacts",
            "next_machine_actions",
        }
        items = sorted(items, key=lambda pair: (str(pair[0]) not in priority, str(pair[0])))[:limit]
        state.record_truncated(path, len(value) - limit)
    result: dict[str, Any] = {}
    for key, item in items:
        if str(key) == _SOURCE_RETENTION_INPUT:
            continue
        label = _path_label(path, key)
        normalized = _normalized_key(key)
        if _provenance_key(key) and isinstance(item, (list, tuple, Mapping)):
            result.update(
                _externalization_marker(
                    key, item, detail, state, label, alias="provenance"
                )
            )
            continue
        if detail == "compact" and normalized == "context":
            state.record_omitted(label)
            result["context_artifact_backed"] = bool(state.artifact_backed)
            result["context_detail_available_via_profile"] = True
            continue
        if detail == "compact" and normalized == "artifactprovenance":
            compact = _project_cache_provenance(item, state, label)
            if compact is not None:
                result[key] = compact
            continue
        if (
            detail == "compact"
            and normalized == "record"
            and isinstance(item, Mapping)
            and "run_id" in item
            and ("operation" in item or "status" in item)
        ):
            result[key] = _project_compact_record(item, state, label)
            if "user_metadata" in item:
                state.record_omitted(_path_label(label, "user_metadata"))
            if "hypothesis" in item:
                state.record_omitted(_path_label(label, "hypothesis"))
            if "change_reason" in item:
                state.record_omitted(_path_label(label, "change_reason"))
            continue
        if normalized == "artifacts":
            result[key] = _project_artifacts(item, detail, state, label)
            continue
        if (
            detail == "compact"
            and lineage_payload
            and normalized in {"children", "ancestors"}
            and _is_experiment_record_list(item)
        ):
            declared_total = value.get(f"{key}_count")
            records, total, truncated = _project_lineage_records(
                item,
                state,
                label,
                total_count=(
                    declared_total
                    if isinstance(declared_total, int)
                    and not isinstance(declared_total, bool)
                    and declared_total >= 0
                    else None
                ),
            )
            result[key] = records
            result[f"{key}_count"] = total
            result[f"{key}_included_count"] = len(records)
            result[f"{key}_truncated_count"] = truncated
            result[f"{key}_truncated"] = bool(truncated)
            continue
        if detail == "compact" and normalized == "spectralfeatures" and isinstance(
            item, Mapping
        ):
            result[key] = _project_compact_spectral_features(item, state, label)
            continue
        if detail == "compact" and normalized == "materials" and isinstance(item, list):
            result[key] = _project_compact_materials(item, state, label)
            continue
        if normalized in {"failures", "nextmachineactions"} and isinstance(item, list):
            limit = MAX_COMPACT_FAILURES if normalized == "failures" else MAX_COMPACT_ACTIONS
            if detail == "compact":
                selected = item[:limit]
                if len(item) > limit:
                    state.record_truncated(label, len(item) - limit)
            else:
                selected = item
            if normalized == "failures" and detail == "compact":
                result[key] = [
                    _project_failure(row, state, f"{label}[{index}]")
                    if isinstance(row, Mapping)
                    else _project_value(row, detail, state, f"{label}[{index}]")
                    for index, row in enumerate(selected)
                ]
            elif normalized == "nextmachineactions" and detail == "compact":
                result[key] = [
                    _project_action(row, state, f"{label}[{index}]")
                    if isinstance(row, Mapping)
                    else _project_value(row, detail, state, f"{label}[{index}]")
                    for index, row in enumerate(selected)
                ]
            else:
                result[key] = _project_list(item, detail, state, label, limit if detail == "standard" else None)
            continue
        if normalized == "warnings" and isinstance(item, list):
            if detail == "compact":
                selected = item[:MAX_COMPACT_WARNINGS]
                if len(item) > MAX_COMPACT_WARNINGS:
                    state.record_truncated(label, len(item) - MAX_COMPACT_WARNINGS)
                result[key] = [
                    _project_warning(row, state, f"{label}[{index}]")
                    for index, row in enumerate(selected)
                ]
            else:
                result[key] = _project_list(item, detail, state, label)
            continue
        if isinstance(item, (list, tuple)):
            array_alias = _array_alias(key, item, value)
            if array_alias is not None:
                result.update(
                    _externalization_marker(
                        key, item, detail, state, label, alias=array_alias
                    )
                )
                continue
            if detail == "compact" and normalized in _COMPACT_RECORD_LIST_KEYS:
                selected = item[:COMPACT_MAX_ITEMS]
                if len(item) > COMPACT_MAX_ITEMS:
                    state.record_truncated(label, len(item) - COMPACT_MAX_ITEMS)
                result[key] = [
                    _project_compact_record(row, state, f"{label}[{index}]")
                    if isinstance(row, Mapping)
                    else _project_value(row, detail, state, f"{label}[{index}]")
                    for index, row in enumerate(selected)
                ]
                continue
            if (
                detail == "full"
                and len(item) > FULL_MAX_GENERIC_ARRAY_ITEMS
                and not path.startswith("record.user_metadata")
                and normalized not in {"actions", "nextmachineactions", "failures", "warnings", "patch", "tags"}
            ):
                result.update(
                    _externalization_marker(
                        key, item, detail, state, label, alias=normalized
                    )
                )
                continue
            result[key] = _project_list(list(item), detail, state, label)
        else:
            result[key] = _project_value(item, detail, state, label)
    return result


def _project_value(value: Any, detail: ResponseDetail, state: _ProjectionState, path: str) -> Any:
    if isinstance(value, Mapping):
        return _project_mapping(value, detail, state, path)
    if isinstance(value, list):
        return _project_list(value, detail, state, path)
    if isinstance(value, tuple):
        return _project_list(list(value), detail, state, path)
    if isinstance(value, str) and detail == "compact":
        return _bounded_string(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return copy.deepcopy(value)
    # Existing callers occasionally pass enum/dataclass-like scalar values.
    # Keep the protocol JSON boundary deterministic without importing the
    # numerical modules into this small policy module.
    return str(value)


class _CanonicalState:
    def __init__(self) -> None:
        self.omitted: dict[str, int] = {}
        self.truncated: dict[str, int] = {}

    def record_omitted(self, path: str, count: int) -> None:
        key = _bounded_string(path or "$", limit=256)
        self.omitted[key] = self.omitted.get(key, 0) + max(1, int(count))

    def record_truncated(self, path: str, count: int) -> None:
        key = _bounded_string(path or "$", limit=256)
        self.truncated[key] = self.truncated.get(key, 0) + max(1, int(count))


def _bounded_count_metadata(
    values: Mapping[str, int], *, limit: int
) -> dict[str, int]:
    selected = sorted(values.items())[:limit]
    result = {key: int(count) for key, count in selected}
    if len(values) > len(selected):
        selected_keys = {key for key, _ in selected}
        result["$additional_paths"] = sum(
            int(count) for key, count in values.items() if key not in selected_keys
        )
    return result


def _canonical_artifact_refs(
    value: Any, state: _CanonicalState, path: str
) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("canonical artifact references must be a list")
    validate_artifact_references(value)
    records = [dict(item) for item in value]
    root_records = [item for item in records if "/" not in str(item.get("path", ""))]
    selected = root_records or records
    if len(selected) > CANONICAL_MAX_ARTIFACT_REFS:
        state.record_truncated(path, len(selected) - CANONICAL_MAX_ARTIFACT_REFS)
    return [copy.deepcopy(item) for item in selected[:CANONICAL_MAX_ARTIFACT_REFS]]


def _canonical_marker(key: Any, value: Any) -> dict[str, Any]:
    name = str(key)
    count = len(value) if isinstance(value, (Mapping, list, tuple)) else 1
    return {
        f"{name}_count": count,
        f"{name}_retained_in_context": False,
        f"{name}_bounded": True,
    }


def _canonical_value(
    value: Any,
    *,
    state: _CanonicalState,
    key: Any = "",
    parent: Mapping[str, Any] | None = None,
    path: str = "",
) -> Any:
    """Keep a bounded, unprojected source for later profile reconstruction."""

    if isinstance(value, Mapping):
        items = list(value.items())
        if len(items) > CANONICAL_MAX_ITEMS:
            priority = {
                "schema_version",
                "protocol_version",
                "ok",
                "status",
                "run_id",
                "operation",
                "summary",
                "physics",
                "failures",
                "artifacts",
                "next_machine_actions",
            }
            items = sorted(
                items,
                key=lambda pair: (str(pair[0]) not in priority, str(pair[0])),
            )[:CANONICAL_MAX_ITEMS]
            state.record_truncated(path, len(value) - len(items))
        return {
            str(item_key): _canonical_value(
                item,
                state=state,
                key=item_key,
                parent=value,
                path=_path_label(path, item_key),
            )
            for item_key, item in items
        }
    if isinstance(value, (list, tuple)):
        normalized = _normalized_key(key)
        if normalized == "artifacts":
            return _canonical_artifact_refs(value, state, path)
        if _array_alias(key, value, parent) is not None:
            state.record_omitted(path, len(value))
            return _canonical_marker(key, value)
        if len(value) > CANONICAL_MAX_ARRAY_ITEMS:
            state.record_omitted(path, len(value))
            return _canonical_marker(key, value)
        return [
            _canonical_value(
                item,
                state=state,
                key=key,
                parent=parent,
                path=f"{path}[{index}]",
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, str):
        if len(value) > CANONICAL_MAX_STRING_CHARS:
            state.record_truncated(path, len(value) - CANONICAL_MAX_STRING_CHARS)
        return _bounded_string(value, limit=CANONICAL_MAX_STRING_CHARS)
    if value is None or isinstance(value, (bool, int, float)):
        return copy.deepcopy(value)
    return str(value)


def canonical_response_context(
    payload: Mapping[str, Any],
    *,
    artifact_refs: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the bounded source artifact used by later ``inspect`` profiles.

    The source intentionally has no ``response`` metadata.  It is therefore
    safe to project later, and it can be written before ``RESPONSE_CONTEXT``
    itself is added to the artifact index without creating a hash cycle.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("response context payload must be a mapping")
    state = _CanonicalState()
    source = _canonical_value(payload, state=state)
    if not isinstance(source, dict):  # pragma: no cover - mapping input guarantees this
        raise TypeError("response context source must be a mapping")
    source.pop("response", None)
    summary = source.get("summary")
    if isinstance(summary, dict):
        summary.pop("response", None)
    if artifact_refs is not None:
        source["artifacts"] = _canonical_artifact_refs(
            list(artifact_refs), state, "artifacts"
        )
    retention = {
        "schema_version": RESPONSE_RETENTION_SCHEMA_VERSION,
        "semantics": "bounded_retained_metadata",
        "bounded": True,
        "full_profile_scope": "full_of_retained_bounded_metadata",
        "large_operation_arrays_retained": False,
        "limits": {
            "mapping_items": CANONICAL_MAX_ITEMS,
            "array_items": CANONICAL_MAX_ARRAY_ITEMS,
            "artifact_references": CANONICAL_MAX_ARTIFACT_REFS,
            "string_characters": CANONICAL_MAX_STRING_CHARS,
        },
        "omitted_fields": _bounded_count_metadata(
            state.omitted, limit=CANONICAL_MAX_METADATA_PATHS
        ),
        "truncated_fields": _bounded_count_metadata(
            state.truncated, limit=CANONICAL_MAX_METADATA_PATHS
        ),
    }
    return {
        "schema_version": RESPONSE_CONTEXT_SCHEMA_VERSION,
        "run_id": source.get("run_id"),
        "task_sha256": source.get("task_sha256"),
        "retention": retention,
        "source": source,
    }


def response_projection_source(
    payload: Mapping[str, Any], retention: Mapping[str, Any]
) -> dict[str, Any]:
    """Attach trusted retained-source metadata for one projection call."""

    result = copy.deepcopy(dict(payload))
    result[_SOURCE_RETENTION_INPUT] = copy.deepcopy(dict(retention))
    return result


def rebase_response_context(
    context: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, Any]:
    """Update run-scoped identities in a copied canonical response source."""

    rebased = copy.deepcopy(dict(context))
    if rebased.get("schema_version") != RESPONSE_CONTEXT_SCHEMA_VERSION:
        raise ValueError(
            "cached response context has an unsupported schema_version"
        )
    source = rebased.get("source")
    if not isinstance(source, dict):
        raise ValueError("response context source must be an object")
    for key in (
        "schema_version",
        "protocol_version",
        "ok",
        "run_id",
        "task_sha256",
        "task_hash_scope",
        "input_sha256",
        "operation",
        "status",
        "certificate_id",
        "cache_hit",
        "source_run_id",
        "artifact_provenance",
    ):
        if key in result:
            source[key] = copy.deepcopy(result[key])
    source_summary = source.get("summary")
    result_summary = result.get("summary")
    if isinstance(source_summary, dict) and isinstance(result_summary, Mapping):
        for key in (
            "run_id",
            "task_sha256",
            "status",
            "mode",
            "cache_hit",
            "source_run_id",
            "artifact_provenance",
        ):
            if key in result_summary:
                source_summary[key] = copy.deepcopy(result_summary[key])
        source_summary.pop("response", None)
    rebased["run_id"] = source.get("run_id")
    rebased["task_sha256"] = source.get("task_sha256")
    rebased["source"] = source
    return rebased


def _mapping_records(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _artifact_summary(
    refs: Any, projected_refs: Any, state: _ProjectionState
) -> dict[str, Any]:
    records = _mapping_records(refs)
    included = _mapping_records(projected_refs)
    invalid_count = sum(not _artifact_ref_shape(item) for item in records)
    state.invalid_artifacts += invalid_count
    by_kind: dict[str, int] = {}
    for item in records:
        kind = str(item.get("kind", "unknown"))
        by_kind[kind] = by_kind.get(kind, 0) + 1
    return {
        "total_count": len(records),
        "included_count": len(included),
        "omitted_count": max(0, len(records) - len(included)),
        "by_kind": dict(sorted(by_kind.items())),
        "all_references_valid_shape": all(_artifact_ref_shape(item) for item in records),
        "invalid_reference_count": invalid_count,
    }


def _response_metadata(
    detail: ResponseDetail,
    state: _ProjectionState,
    original_refs: Any,
    projected_refs: Any,
    retention: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "schema_version": RESPONSE_SCHEMA_VERSION,
        "profile": detail,
        "available_profiles": list(RESPONSE_DETAILS),
        "artifact_backed": bool(state.artifact_backed),
        "detail_available_via_profile": detail != "full",
        "artifact_summary": _artifact_summary(original_refs, projected_refs, state),
    }
    if detail == "compact":
        metadata["context_budget"] = {
            "guard": "veritmm-compact-context-v2",
            "target_bytes": COMPACT_TARGET_BYTES,
            "max_bytes": COMPACT_MAX_BYTES,
        }
    if retention is not None:
        metadata["source_retention"] = {
            key: copy.deepcopy(retention[key])
            for key in (
                "schema_version",
                "semantics",
                "bounded",
                "full_profile_scope",
                "large_operation_arrays_retained",
                "limits",
            )
            if key in retention
        }
    if state.omitted:
        metadata["omitted_fields"] = _bounded_count_metadata(
            state.omitted, limit=MAX_RESPONSE_METADATA_PATHS
        )
    if state.truncated:
        metadata["truncated_fields"] = _bounded_count_metadata(
            state.truncated, limit=MAX_RESPONSE_METADATA_PATHS
        )
    return metadata


def guard_context_budget(
    payload: Mapping[str, Any],
    *,
    detail: str | None = DEFAULT_RESPONSE_DETAIL,
    max_bytes: int = COMPACT_MAX_BYTES,
) -> int:
    """Assert compact structural and byte budgets; return encoded byte size."""

    profile = normalize_response_detail(detail)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    if profile != "compact":
        return len(encoded)
    violations: list[str] = []

    def scan(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            lineage_payload = _is_lineage_payload(value)
            for key, item in value.items():
                label = _path_label(path, key)
                lineage_records = (
                    lineage_payload
                    and _normalized_key(key) in {"children", "ancestors"}
                    and _is_experiment_record_list(item)
                )
                if (
                    not lineage_records
                    and _forbidden_compact_array_key(key, item, value)
                    and isinstance(item, (list, tuple))
                ):
                    violations.append(label)
                scan(item, label)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                scan(item, f"{path}[{index}]")

    scan(payload, "")
    if violations:
        raise ContextBudgetError(
            "compact response contains forbidden large-payload arrays: "
            + ", ".join(sorted(violations))
        )
    if len(encoded) > int(max_bytes):
        raise ContextBudgetError(
            f"compact response exceeds the {int(max_bytes)}-byte context budget "
            f"({len(encoded)} bytes)"
        )
    return len(encoded)


def validate_projected_response(
    payload: Mapping[str, Any],
    *,
    detail: str | None = None,
) -> bool:
    """Validate an existing projection before it crosses a machine boundary."""

    if not isinstance(payload, Mapping):
        raise TypeError("projected response must be a mapping")
    expected_profile = None if detail is None else normalize_response_detail(detail)
    summary = payload.get("summary")
    nested = summary.get("response") if isinstance(summary, Mapping) else None
    top = payload.get("response")
    if isinstance(nested, Mapping) and isinstance(top, Mapping):
        raise ValueError("projected response contains mixed top-level and nested metadata")
    run_envelope = _is_run_envelope(payload)
    metadata = nested if run_envelope else top
    if not isinstance(metadata, Mapping):
        raise ValueError("projected response metadata is missing or misplaced")
    if run_envelope and top is not None:
        raise ValueError("run response metadata must be nested under summary")
    if not run_envelope and nested is not None:
        raise ValueError("non-run response metadata must be top-level")
    if metadata.get("schema_version") != RESPONSE_SCHEMA_VERSION:
        raise ValueError("projected response metadata schema_version is invalid")
    profile = metadata.get("profile")
    if profile not in RESPONSE_DETAILS:
        raise ValueError("projected response profile is invalid")
    if expected_profile is not None and profile != expected_profile:
        raise ValueError(
            f"projected response profile {profile!r} does not match {expected_profile!r}"
        )
    if list(metadata.get("available_profiles") or []) != list(RESPONSE_DETAILS):
        raise ValueError("projected response available_profiles is invalid")
    if metadata.get("detail_available_via_profile") is not (profile != "full"):
        raise ValueError("projected response detail availability is inconsistent")

    refs = payload.get("artifacts", [])
    if not isinstance(refs, list):
        raise ValueError("projected response artifacts must be a list")
    validate_artifact_references(refs)
    artifact_summary = metadata.get("artifact_summary")
    if not isinstance(artifact_summary, Mapping):
        raise ValueError("projected response artifact_summary is missing")
    counts = {
        key: artifact_summary.get(key)
        for key in ("total_count", "included_count", "omitted_count")
    }
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in counts.values()
    ):
        raise ValueError("projected response artifact counts are invalid")
    total = int(counts["total_count"])
    included = int(counts["included_count"])
    omitted = int(counts["omitted_count"])
    if included != len(refs) or total != included + omitted:
        raise ValueError("projected response artifact counts are inconsistent")
    by_kind = artifact_summary.get("by_kind")
    if not isinstance(by_kind, Mapping) or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in by_kind.values()
    ):
        raise ValueError("projected response artifact kind counts are invalid")
    if sum(int(value) for value in by_kind.values()) != total:
        raise ValueError("projected response artifact kind counts are inconsistent")
    if artifact_summary.get("all_references_valid_shape") is not True:
        raise ValueError("projected response reports invalid artifact references")
    if artifact_summary.get("invalid_reference_count") != 0:
        raise ValueError("projected response invalid artifact count is nonzero")
    if metadata.get("artifact_backed") is not (total > 0):
        raise ValueError("projected response artifact_backed is inconsistent")
    for key in ("omitted_fields", "truncated_fields"):
        values = metadata.get(key, {})
        if not isinstance(values, Mapping) or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in values.values()
        ):
            raise ValueError(f"projected response {key} metadata is invalid")
    if profile == "compact":
        expected_budget = {
            "guard": "veritmm-compact-context-v2",
            "target_bytes": COMPACT_TARGET_BYTES,
            "max_bytes": COMPACT_MAX_BYTES,
        }
        if metadata.get("context_budget") != expected_budget:
            raise ValueError("projected compact response context_budget is invalid")
        guard_context_budget(payload, detail="compact")
    return True


def project_response(
    payload: Mapping[str, Any],
    *,
    detail: str | None = DEFAULT_RESPONSE_DETAIL,
) -> dict[str, Any]:
    """Project one response using the shared compact/standard/full policy.

    Projection is pure: callers retain the original detailed mapping for
    artifact persistence or legacy use.  The returned mapping is JSON-safe and
    carries a nested ``response`` metadata object so the existing v1 envelope
    schema remains valid for old consumers.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("response payload must be a mapping")
    profile = normalize_response_detail(detail)
    existing_profile = response_profile(payload)
    if existing_profile is not None:
        if existing_profile != profile:
            raise ValueError(
                "cannot re-project an already projected response from "
                f"{existing_profile!r} to {profile!r}; load its canonical response context"
            )
        validate_projected_response(payload, detail=profile)
        result = copy.deepcopy(dict(payload))
        return result
    state = _ProjectionState()
    original_refs = payload.get("artifacts", [])
    if not isinstance(original_refs, list):
        raise ValueError("response artifacts must be a list")
    validate_artifact_references(original_refs)
    valid_refs = [
        item for item in _mapping_records(original_refs) if _artifact_ref_shape(item)
    ]
    state.artifact_kinds = {str(item["kind"]) for item in valid_refs}
    state.artifact_backed = bool(valid_refs)
    result = _project_mapping(payload, profile, state, "")
    projected_refs = result.get("artifacts", [])
    retention = payload.get(_SOURCE_RETENTION_INPUT)
    if retention is not None and not isinstance(retention, Mapping):
        raise ValueError("response retention metadata must be an object")
    metadata = _response_metadata(
        profile,
        state,
        original_refs,
        projected_refs,
        retention=retention,
    )
    if _is_run_envelope(payload):
        summary = result.get("summary")
        if not isinstance(summary, dict):
            summary = {}
            result["summary"] = summary
        summary["response"] = metadata
    else:
        result["response"] = metadata
    validate_projected_response(result, detail=profile)
    return result


def compact_response(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Convenience alias for the default machine-facing profile."""

    return project_response(payload, detail="compact")


__all__ = [
    "CANONICAL_MAX_ARTIFACT_REFS",
    "COMPACT_MAX_BYTES",
    "COMPACT_TARGET_BYTES",
    "ContextBudgetError",
    "DEFAULT_RESPONSE_DETAIL",
    "RESPONSE_CONTEXT_FILENAME",
    "RESPONSE_CONTEXT_SCHEMA_VERSION",
    "RESPONSE_DETAILS",
    "RESPONSE_RETENTION_SCHEMA_VERSION",
    "RESPONSE_SCHEMA_VERSION",
    "ResponseDetail",
    "canonical_response_context",
    "compact_response",
    "guard_context_budget",
    "is_projected_response",
    "normalize_response_detail",
    "project_response",
    "rebase_response_context",
    "response_projection_source",
    "response_profile",
    "validate_artifact_references",
    "validate_projected_response",
]
