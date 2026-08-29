"""Resumable bounded batch orchestration for research evaluations."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from pydantic import Field, JsonValue, StrictBool, StrictInt, StrictStr, model_validator

from ..protocol.models import ResponseMetadata
from ..protocol.responses import COMPACT_MAX_BYTES, guard_context_budget, project_response
from ..run_artifacts import (
    index_artifacts,
    stable_payload_sha256,
    write_json,
)
from .contracts import DesignCandidate, ResearchModel, content_id
from .evaluator import EvaluationRecord, ResearchArtifactRef

if TYPE_CHECKING:
    from .evaluator import ResearchEvaluator

BATCH_REQUEST_SCHEMA_VERSION = "veritmm-research-batch-request-v1"
BATCH_RESULT_SCHEMA_VERSION = "veritmm-research-batch-result-v1"
BATCH_MANIFEST_SCHEMA_VERSION = "veritmm-research-batch-manifest-v1"
BATCH_INDEX_SCHEMA_VERSION = "veritmm-research-batch-index-v1"
BATCH_PREVIEW_LIMIT = 12


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BatchEvaluationRequest(ResearchModel):
    """Stable ordered candidate batch bound to one design and objective set."""

    schema_version: Literal[BATCH_REQUEST_SCHEMA_VERSION] = BATCH_REQUEST_SCHEMA_VERSION
    batch_id: StrictStr = ""
    design_space_id: StrictStr
    objective_set_id: StrictStr
    candidates: tuple[DesignCandidate, ...] = Field(min_length=1)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _valid_request(self) -> "BatchEvaluationRequest":
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("batch candidate IDs must be unique")
        if any(
            candidate.design_space_id != self.design_space_id
            for candidate in self.candidates
        ):
            raise ValueError("every batch candidate must belong to design_space_id")
        expected = content_id(
            "batch",
            self.model_dump(mode="json", exclude={"batch_id"}),
        )
        if self.batch_id and self.batch_id != expected:
            raise ValueError("batch_id does not match request content")
        object.__setattr__(self, "batch_id", expected)
        return self


class BatchEvaluationPreview(ResearchModel):
    """Bounded first-read identity and status for one evaluated candidate."""

    candidate_id: StrictStr
    status: Literal["completed", "failed"]
    physics_accepted: StrictBool
    feasible: StrictBool | None
    total_score: float | None
    run_id: StrictStr | None
    cache_hit: StrictBool


class BatchEvaluationResult(ResearchModel):
    """Compact batch response; complete records remain in ``BATCH_INDEX.jsonl``."""

    schema_version: Literal[BATCH_RESULT_SCHEMA_VERSION] = BATCH_RESULT_SCHEMA_VERSION
    ok: StrictBool
    batch_id: StrictStr
    status: Literal["completed", "partial", "failed"]
    executor: StrictStr
    candidate_count: StrictInt
    completed_count: StrictInt
    failed_count: StrictInt
    feasible_count: StrictInt
    preview: tuple[BatchEvaluationPreview, ...]
    preview_count: StrictInt
    truncated_count: StrictInt
    artifact_root: StrictStr
    artifacts: tuple[ResearchArtifactRef, ...]
    response: ResponseMetadata

    @model_validator(mode="after")
    def _valid_counts(self) -> "BatchEvaluationResult":
        if self.candidate_count != self.completed_count + self.failed_count:
            raise ValueError("batch result counts are inconsistent")
        if self.preview_count != len(self.preview):
            raise ValueError("batch preview_count is inconsistent")
        if self.truncated_count != self.candidate_count - self.preview_count:
            raise ValueError("batch truncated_count is inconsistent")
        if self.preview_count > BATCH_PREVIEW_LIMIT:
            raise ValueError("batch preview exceeds its bounded limit")
        if self.response.profile != "compact":
            raise ValueError("batch response must use the compact profile")
        return self


@runtime_checkable
class BatchExecutor(Protocol):
    """Replaceable execution strategy for an ordered set of pending candidates."""

    name: str

    def execute(
        self,
        evaluator: "ResearchEvaluator",
        candidates: tuple[DesignCandidate, ...],
        *,
        output_root: Path,
    ) -> Iterable[EvaluationRecord]:
        """Yield exactly one record per candidate in input order."""


class SequentialBatchExecutor:
    """Deterministic reference executor with per-candidate failure isolation."""

    name = "sequential"

    def execute(
        self,
        evaluator: "ResearchEvaluator",
        candidates: tuple[DesignCandidate, ...],
        *,
        output_root: Path,
    ) -> Iterable[EvaluationRecord]:
        for candidate in candidates:
            yield evaluator.evaluate(candidate, output_root=output_root)


class ChunkedVerifiedBatchExecutor:
    """Chunk proposal forwarding while retaining independent verification."""

    name = "chunked_verified"

    def __init__(self, batch_size: int) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.batch_size = int(batch_size)

    def execute(
        self,
        evaluator: "ResearchEvaluator",
        candidates: tuple[DesignCandidate, ...],
        *,
        output_root: Path,
    ) -> Iterable[EvaluationRecord]:
        for start in range(0, len(candidates), self.batch_size):
            chunk = candidates[start : start + self.batch_size]
            # Batching improves throughput but every candidate still receives
            # an independent certificate. Batch execution does not change
            # verification decisions.
            batch_forward = getattr(evaluator, "_batch_forward_proposals", None)
            if callable(batch_forward):
                batch_forward(chunk)
            for candidate in chunk:
                yield evaluator.evaluate(candidate, output_root=output_root)


def build_batch_result(
    *,
    batch_id: str,
    records: Iterable[EvaluationRecord],
    executor_name: str,
    artifact_root: str | Path,
    artifacts: Iterable[ResearchArtifactRef | Mapping[str, Any]] = (),
) -> BatchEvaluationResult:
    """Build and enforce the bounded first-read response for any batch size."""

    record_list = list(records)
    completed = sum(record.status == "completed" for record in record_list)
    failed = len(record_list) - completed
    feasible = sum(record.feasible is True for record in record_list)
    status: Literal["completed", "partial", "failed"]
    if failed == 0:
        status = "completed"
    elif completed == 0:
        status = "failed"
    else:
        status = "partial"
    refs = [
        item.model_dump(mode="json")
        if isinstance(item, ResearchArtifactRef)
        else dict(item)
        for item in artifacts
    ]
    preview = [
        {
            "candidate_id": record.candidate_id,
            "status": record.status,
            "physics_accepted": record.physics_accepted,
            "feasible": record.feasible,
            "total_score": record.total_score,
            "run_id": record.run_id,
            "cache_hit": record.cache_hit,
        }
        for record in record_list[:BATCH_PREVIEW_LIMIT]
    ]
    raw = {
        "schema_version": BATCH_RESULT_SCHEMA_VERSION,
        "ok": status == "completed",
        "batch_id": batch_id,
        "status": status,
        "executor": executor_name,
        "candidate_count": len(record_list),
        "completed_count": completed,
        "failed_count": failed,
        "feasible_count": feasible,
        "preview": preview,
        "preview_count": len(preview),
        "truncated_count": len(record_list) - len(preview),
        "artifact_root": str(Path(artifact_root).resolve()),
        "artifacts": refs,
    }
    projected = project_response(raw, detail="compact")
    guard_context_budget(projected, detail="compact", max_bytes=COMPACT_MAX_BYTES)
    return BatchEvaluationResult.model_validate_json(
        json.dumps(projected, ensure_ascii=False, allow_nan=False)
    )


def evaluate_batch(
    evaluator: "ResearchEvaluator",
    request: BatchEvaluationRequest | Mapping[str, Any],
    *,
    executor: BatchExecutor,
    resume: bool,
    output_dir: str | Path | None,
) -> BatchEvaluationResult:
    """Execute or resume a batch with a fail-closed append-only ledger."""

    request = (
        request
        if isinstance(request, BatchEvaluationRequest)
        else BatchEvaluationRequest.model_validate(request)
    )
    if request.design_space_id != evaluator.design_space.design_space_id:
        raise ValueError("batch request design_space_id does not match evaluator")
    if request.objective_set_id != evaluator.objectives.objective_set_id:
        raise ValueError("batch request objective_set_id does not match evaluator")
    if not isinstance(executor, BatchExecutor):
        raise TypeError("executor must implement the BatchExecutor protocol")
    executor_name = getattr(executor, "name", "")
    if not isinstance(executor_name, str) or not executor_name.strip():
        raise ValueError("batch executor name must be a non-empty string")

    root = (
        Path(output_dir)
        if output_dir is not None
        else Path(evaluator.config.output_root)
        / "batches"
        / f"b_{request.batch_id.rsplit('_', 1)[-1][:16]}"
    ).resolve()
    manifest_path = root / "BATCH_MANIFEST.json"
    index_path = root / "BATCH_INDEX.jsonl"
    request_sha256 = stable_payload_sha256(request.model_dump(mode="json"))
    if root.exists() and any(root.iterdir()):
        if not resume:
            raise FileExistsError("batch output already exists; enable resume to continue")
        manifest = _load_manifest(manifest_path)
        _validate_manifest(manifest, request, request_sha256, index_path, root)
        records = _load_index(index_path, request, root)
    else:
        root.mkdir(parents=True, exist_ok=True)
        _initialize_index(index_path)
        records = {}
        _write_manifest(
            manifest_path,
            request,
            request_sha256=request_sha256,
            executor_name=executor_name,
            status="running",
            processed_count=0,
        )

    pending = [
        (index, candidate)
        for index, candidate in enumerate(request.candidates)
        if index not in records
    ]
    iterator = iter(
        executor.execute(
            evaluator,
            tuple(candidate for _, candidate in pending),
            output_root=root / "evaluations",
        )
    )
    try:
        for index, candidate in pending:
            try:
                record = next(iterator)
            except StopIteration as exc:
                raise ValueError("batch executor returned too few records") from exc
            _validate_record(record, candidate, evaluator)
            _append_index(index_path, request.batch_id, index, record)
            records[index] = record
            _write_manifest(
                manifest_path,
                request,
                request_sha256=request_sha256,
                executor_name=executor_name,
                status="running",
                processed_count=len(records),
            )
        try:
            next(iterator)
        except StopIteration:
            pass
        else:
            raise ValueError("batch executor returned too many records")
    except Exception:
        _write_manifest(
            manifest_path,
            request,
            request_sha256=request_sha256,
            executor_name=executor_name,
            status="interrupted",
            processed_count=len(records),
        )
        raise

    ordered = tuple(records[index] for index in range(len(request.candidates)))
    completed_count = sum(record.status == "completed" for record in ordered)
    final_status = (
        "completed"
        if completed_count == len(ordered)
        else ("failed" if completed_count == 0 else "partial")
    )
    index_ref = _artifact_ref(index_path, root)
    _write_manifest(
        manifest_path,
        request,
        request_sha256=request_sha256,
        executor_name=executor_name,
        status=final_status,
        processed_count=len(ordered),
        completed_count=completed_count,
        failed_count=len(ordered) - completed_count,
        index_artifact=index_ref,
    )
    artifact_refs = tuple(
        ResearchArtifactRef.model_validate(item)
        for item in index_artifacts(root)
        if item["kind"] in {"research_batch_manifest", "research_batch_index"}
    )
    from ..protocol.responses import validate_artifact_references

    validate_artifact_references(
        [item.model_dump(mode="python") for item in artifact_refs], root=root
    )
    if {item.kind for item in artifact_refs} != {
        "research_batch_manifest",
        "research_batch_index",
    }:
        raise ValueError("batch manifest and index artifact references are required")
    return build_batch_result(
        batch_id=request.batch_id,
        records=ordered,
        executor_name=executor_name,
        artifact_root=root,
        artifacts=artifact_refs,
    )


def _initialize_index(path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(b"")
    temporary.replace(path)


def _append_index(
    path: Path, batch_id: str, index: int, record: EvaluationRecord
) -> None:
    payload = {
        "schema_version": BATCH_INDEX_SCHEMA_VERSION,
        "batch_id": batch_id,
        "index": index,
        "candidate_id": record.candidate_id,
        "record": record.model_dump(mode="json"),
    }
    line = (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_APPEND | os.O_WRONLY)
    try:
        written = os.write(descriptor, line)
        if written != len(line):  # pragma: no cover - defensive short write
            raise OSError("short write while appending batch index")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_index(
    path: Path,
    request: BatchEvaluationRequest,
    root: Path,
) -> dict[int, EvaluationRecord]:
    if not path.is_file():
        raise ValueError("batch index is missing")
    records: dict[int, EvaluationRecord] = {}
    candidate_ids: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"batch index is unreadable: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"batch index line {line_number} is corrupt") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"batch index line {line_number} must be an object")
        if payload.get("schema_version") != BATCH_INDEX_SCHEMA_VERSION:
            raise ValueError("batch index schema_version is invalid")
        if payload.get("batch_id") != request.batch_id:
            raise ValueError("batch index is bound to a different batch")
        index = payload.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("batch index entry index must be an integer")
        if not 0 <= index < len(request.candidates):
            raise ValueError("batch index entry is outside the request")
        expected_id = request.candidates[index].candidate_id
        if payload.get("candidate_id") != expected_id:
            raise ValueError("batch index candidate identity mismatch")
        if index in records or expected_id in candidate_ids:
            raise ValueError("batch index contains a duplicate candidate")
        try:
            record = EvaluationRecord.model_validate_json(
                json.dumps(payload.get("record"), ensure_ascii=False, allow_nan=False)
            )
        except Exception as exc:
            raise ValueError(f"batch index record {index} is invalid") from exc
        if record.candidate_id != expected_id:
            raise ValueError("batch index record candidate identity mismatch")
        _validate_record_artifacts(record)
        records[index] = record
        candidate_ids.add(expected_id)
    return records


def _validate_record(
    record: EvaluationRecord,
    candidate: DesignCandidate,
    evaluator: "ResearchEvaluator",
) -> None:
    if not isinstance(record, EvaluationRecord):
        raise TypeError("batch executor must yield EvaluationRecord instances")
    if record.candidate_id != candidate.candidate_id:
        raise ValueError("batch executor changed candidate ordering or identity")
    if record.design_space_id != evaluator.design_space.design_space_id:
        raise ValueError("batch record design-space identity mismatch")
    if record.objective_set_id != evaluator.objectives.objective_set_id:
        raise ValueError("batch record objective-set identity mismatch")
    _validate_record_artifacts(record)


def _validate_record_artifacts(record: EvaluationRecord) -> None:
    if not record.artifacts:
        return
    if record.artifact_root is None:
        raise ValueError("batch record artifact root is missing")
    from ..protocol.responses import validate_artifact_references

    validate_artifact_references(
        [item.model_dump(mode="python") for item in record.artifacts],
        root=record.artifact_root,
    )


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"batch manifest is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("batch manifest must contain an object")
    return payload


def _validate_manifest(
    manifest: Mapping[str, Any],
    request: BatchEvaluationRequest,
    request_sha256: str,
    index_path: Path,
    root: Path,
) -> None:
    expected = {
        "schema_version": BATCH_MANIFEST_SCHEMA_VERSION,
        "batch_id": request.batch_id,
        "request_sha256": request_sha256,
        "design_space_id": request.design_space_id,
        "objective_set_id": request.objective_set_id,
        "candidate_count": len(request.candidates),
        "candidate_ids": [candidate.candidate_id for candidate in request.candidates],
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"batch manifest binding mismatch: {key}")
    index_artifact = manifest.get("index_artifact")
    if index_artifact is not None:
        from ..protocol.responses import validate_artifact_references

        validate_artifact_references((index_artifact,), root=root)
        if index_artifact.get("path") != index_path.relative_to(root).as_posix():
            raise ValueError("batch manifest index path is invalid")


def _write_manifest(
    path: Path,
    request: BatchEvaluationRequest,
    *,
    request_sha256: str,
    executor_name: str,
    status: str,
    processed_count: int,
    completed_count: int | None = None,
    failed_count: int | None = None,
    index_artifact: Mapping[str, Any] | None = None,
) -> None:
    payload = {
        "schema_version": BATCH_MANIFEST_SCHEMA_VERSION,
        "batch_id": request.batch_id,
        "request_sha256": request_sha256,
        "design_space_id": request.design_space_id,
        "objective_set_id": request.objective_set_id,
        "candidate_count": len(request.candidates),
        "candidate_ids": [candidate.candidate_id for candidate in request.candidates],
        "executor": executor_name,
        "status": status,
        "processed_count": processed_count,
        "completed_count": completed_count,
        "failed_count": failed_count,
        "updated_at": _utc_now(),
        "metadata": request.metadata,
        "index_artifact": None if index_artifact is None else dict(index_artifact),
    }
    write_json(path, payload)


def _artifact_ref(path: Path, root: Path) -> dict[str, Any]:
    from ..run_artifacts import file_sha256

    return {
        "kind": "research_batch_index",
        "path": path.relative_to(root).as_posix(),
        "schema_version": BATCH_INDEX_SCHEMA_VERSION,
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


__all__ = [
    "BATCH_INDEX_SCHEMA_VERSION",
    "BATCH_MANIFEST_SCHEMA_VERSION",
    "BATCH_REQUEST_SCHEMA_VERSION",
    "BATCH_RESULT_SCHEMA_VERSION",
    "BatchEvaluationPreview",
    "BatchEvaluationRequest",
    "BatchEvaluationResult",
    "BatchExecutor",
    "ChunkedVerifiedBatchExecutor",
    "SequentialBatchExecutor",
    "build_batch_result",
    "evaluate_batch",
]
