"""Verified, resumable dataset generation over the public research evaluator."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    Field,
    JsonValue,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from .. import __version__
from ..archive.schema_registry import ARCHIVE_SCHEMA_VERSION
from ..protocol.models import ResponseMetadata
from ..protocol.responses import (
    COMPACT_MAX_BYTES,
    guard_context_budget,
    project_response,
    validate_artifact_references,
)
from ..run_artifacts import (
    file_sha256,
    index_artifacts,
    stable_payload_sha256,
    write_json,
)
from ..schemas import LayerSpec, MediumSpec, SimulationTask
from .batch import BATCH_INDEX_SCHEMA_VERSION, BatchEvaluationRequest, BatchExecutor
from .contracts import DesignCandidate, ResearchModel, content_id
from .design_space import DesignSpace
from .evaluator import EvaluationRecord, ResearchArtifactRef, ResearchEvaluator
from .sampling import SamplingPlan, sample_candidates

DATASET_CONFIG_SCHEMA_VERSION = "veritmm-dataset-config-v1"
DATASET_RECORD_SCHEMA_VERSION = "veritmm-dataset-record-v1"
DATASET_MANIFEST_SCHEMA_VERSION = "veritmm-dataset-manifest-v1"
DATASET_INDEX_SCHEMA_VERSION = "veritmm-dataset-index-v1"
DATASET_RESULT_SCHEMA_VERSION = "veritmm-dataset-result-v1"
MAX_DATASET_PREVIEW = 24
VersionIdentityStatus = Literal["verified", "legacy_inconsistent"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DatasetConfig(ResearchModel):
    """Persistence, resume, cache, and compact-preview policy."""

    schema_version: Literal[DATASET_CONFIG_SCHEMA_VERSION] = DATASET_CONFIG_SCHEMA_VERSION
    output_root: StrictStr
    resume: StrictBool = True
    cache: StrictBool = True
    preview_limit: StrictInt = 12

    @field_validator("output_root")
    @classmethod
    def _valid_output_root(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("output_root must not be empty")
        return value

    @field_validator("preview_limit")
    @classmethod
    def _valid_preview_limit(cls, value: int) -> int:
        if not 1 <= value <= MAX_DATASET_PREVIEW:
            raise ValueError(
                f"preview_limit must be between 1 and {MAX_DATASET_PREVIEW}"
            )
        return value


class DatasetMaterialIdentity(ResearchModel):
    """Material selector for an incident medium, finite layer, or exit medium."""

    position: StrictStr
    layer_index: StrictInt | None = None
    material: StrictStr | None = None
    provider: StrictStr | None = None
    dataset_id: StrictStr | StrictInt | None = None
    constant_n: StrictFloat | None = None
    constant_k: StrictFloat
    thickness_nm: StrictFloat | None = None

    @model_validator(mode="after")
    def _valid_identity(self) -> "DatasetMaterialIdentity":
        if (self.material is None) == (self.constant_n is None):
            raise ValueError("material identity requires material or constant_n")
        if self.position == "layer":
            if self.layer_index is None or self.thickness_nm is None:
                raise ValueError("finite layer identity requires index and thickness")
        elif self.position not in {"incident", "exit"}:
            raise ValueError("material identity position is invalid")
        elif self.layer_index is not None or self.thickness_nm is not None:
            raise ValueError("semi-infinite medium cannot have layer geometry")
        return self


class DatasetWavelengthConfig(ResearchModel):
    """Compact wavelength-grid identity without an inline wavelength array."""

    mode: Literal["linspace", "explicit"]
    start_nm: StrictFloat
    stop_nm: StrictFloat
    point_count: StrictInt
    configuration_sha256: StrictStr


class DatasetRecord(ResearchModel):
    """One compact sample row bound to managed execution and verification."""

    schema_version: Literal[DATASET_RECORD_SCHEMA_VERSION] = DATASET_RECORD_SCHEMA_VERSION
    archive_schema_version: StrictInt = ARCHIVE_SCHEMA_VERSION
    dataset_id: StrictStr
    plan_id: StrictStr
    candidate_id: StrictStr
    sample_index: StrictInt
    seed: StrictInt
    design_variables: dict[StrictStr, StrictInt | StrictFloat | StrictStr]
    normalized_design: tuple[StrictFloat, ...]
    task_sha256: StrictStr | None
    run_id: StrictStr | None
    material_catalog_sha256: StrictStr
    material_identities: tuple[DatasetMaterialIdentity, ...] = Field(min_length=1)
    wavelength: DatasetWavelengthConfig
    requested_outputs: tuple[StrictStr, ...]
    selected_outputs: tuple[StrictStr, ...]
    evaluation_status: Literal["completed", "failed"]
    verification_status: Literal["accepted", "rejected", "failed"]
    physics_accepted: StrictBool
    certificate_id: StrictStr | None
    veritmm_version: StrictStr
    version_identity_status: VersionIdentityStatus = "verified"
    artifact_root: StrictStr | None = None
    artifacts: tuple[ResearchArtifactRef, ...] = ()
    provenance: dict[str, JsonValue] = Field(default_factory=dict)
    failure_codes: tuple[StrictStr, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _infer_version_identity_status(cls, value: Any) -> Any:
        """Annotate legacy records without rewriting their recorded version."""

        if not isinstance(value, Mapping):
            return value
        payload = dict(value)
        # Pydantic strict JSON validation does not coerce JSON arrays to tuple
        # fields on newer releases; normalize persisted arrays before applying
        # the record validators so old and new dataset indexes remain readable.
        for key in (
            "normalized_design",
            "material_identities",
            "requested_outputs",
            "selected_outputs",
            "artifacts",
            "failure_codes",
        ):
            if isinstance(payload.get(key), list):
                payload[key] = tuple(payload[key])
        if "version_identity_status" not in payload:
            payload["version_identity_status"] = (
                "verified"
                if payload.get("veritmm_version") == __version__
                else "legacy_inconsistent"
            )
        return payload

    @model_validator(mode="after")
    def _valid_verification(self) -> "DatasetRecord":
        expected_version_status: VersionIdentityStatus = (
            "verified"
            if self.veritmm_version == __version__
            else "legacy_inconsistent"
        )
        if self.version_identity_status != expected_version_status:
            raise ValueError(
                "version_identity_status does not match veritmm_version"
            )
        accepted = self.verification_status == "accepted"
        if accepted:
            if (
                self.evaluation_status != "completed"
                or not self.physics_accepted
                or self.certificate_id is None
                or self.run_id is None
                or self.task_sha256 is None
            ):
                raise ValueError("accepted dataset record requires verified run identity")
        elif self.physics_accepted or self.certificate_id is not None:
            raise ValueError("unverified dataset record cannot retain acceptance identity")
        if self.sample_index < 0:
            raise ValueError("sample_index must be non-negative")
        if self.artifacts and self.artifact_root is None:
            raise ValueError("dataset record artifacts require artifact_root")
        return self


class DatasetManifest(ResearchModel):
    """Bounded persisted dataset identity and ledger summary."""

    schema_version: Literal[DATASET_MANIFEST_SCHEMA_VERSION] = (
        DATASET_MANIFEST_SCHEMA_VERSION
    )
    archive_schema_version: StrictInt = ARCHIVE_SCHEMA_VERSION
    response_profile: Literal["compact"] = "compact"
    dataset_id: StrictStr
    plan_id: StrictStr
    strategy: StrictStr
    design_space_id: StrictStr
    objective_set_id: StrictStr
    batch_id: StrictStr
    candidate_count: StrictInt
    candidate_order_sha256: StrictStr
    status: Literal["running", "interrupted", "completed", "partial", "failed"]
    processed_count: StrictInt
    accepted_count: StrictInt
    rejected_count: StrictInt
    failed_count: StrictInt
    veritmm_version: StrictStr
    created_at: StrictStr
    updated_at: StrictStr
    provenance: dict[str, JsonValue] = Field(default_factory=dict)
    index_artifact: ResearchArtifactRef | None = None

    @model_validator(mode="after")
    def _valid_counts(self) -> "DatasetManifest":
        if self.processed_count != (
            self.accepted_count + self.rejected_count + self.failed_count
        ):
            raise ValueError("dataset manifest counts are inconsistent")
        if not 0 <= self.processed_count <= self.candidate_count:
            raise ValueError("dataset manifest processed_count is invalid")
        return self


class DatasetRecordPreview(ResearchModel):
    candidate_id: StrictStr
    sample_index: StrictInt
    verification_status: Literal["accepted", "rejected", "failed"]
    physics_accepted: StrictBool
    run_id: StrictStr | None
    certificate_id: StrictStr | None


class DatasetGenerationResult(ResearchModel):
    """Compact artifact-backed first read for a generated dataset."""

    schema_version: Literal[DATASET_RESULT_SCHEMA_VERSION] = DATASET_RESULT_SCHEMA_VERSION
    ok: StrictBool
    dataset_id: StrictStr
    plan_id: StrictStr
    status: Literal["completed", "partial", "failed"]
    record_count: StrictInt
    accepted_count: StrictInt
    rejected_count: StrictInt
    failed_count: StrictInt
    preview: tuple[DatasetRecordPreview, ...]
    preview_count: StrictInt
    truncated_count: StrictInt
    artifact_root: StrictStr
    artifacts: tuple[ResearchArtifactRef, ...]
    response: ResponseMetadata

    @model_validator(mode="after")
    def _valid_result(self) -> "DatasetGenerationResult":
        if self.record_count != (
            self.accepted_count + self.rejected_count + self.failed_count
        ):
            raise ValueError("dataset result counts are inconsistent")
        if self.preview_count != len(self.preview):
            raise ValueError("dataset result preview_count is inconsistent")
        if self.truncated_count != self.record_count - self.preview_count:
            raise ValueError("dataset result truncated_count is inconsistent")
        if self.response.profile != "compact":
            raise ValueError("dataset generation result must be compact")
        return self


class _DatasetSequentialExecutor:
    """Batch-compatible evaluator adapter that enforces DatasetConfig.cache."""

    name = "dataset-sequential"

    def __init__(self, cache: bool) -> None:
        self.cache = bool(cache)

    def execute(
        self,
        evaluator: ResearchEvaluator,
        candidates: tuple[DesignCandidate, ...],
        *,
        output_root: Path,
    ) -> Iterable[EvaluationRecord]:
        for candidate in candidates:
            yield evaluator.evaluate(
                candidate,
                cache=self.cache,
                experiment_metadata={"dataset_generation": True},
                output_root=output_root,
            )


class DatasetFactory:
    """Generate verified datasets exclusively through ``evaluate_many()``."""

    def __init__(self, design_space: DesignSpace, evaluator: ResearchEvaluator) -> None:
        if not isinstance(design_space, DesignSpace):
            raise TypeError("design_space must be a DesignSpace")
        if not isinstance(evaluator, ResearchEvaluator):
            raise TypeError("evaluator must be a ResearchEvaluator")
        if design_space.design_space_id != evaluator.design_space.design_space_id:
            raise ValueError("DatasetFactory design space does not match evaluator")
        self.design_space = design_space
        self.evaluator = evaluator

    def generate(
        self,
        plan: SamplingPlan,
        config: DatasetConfig,
        *,
        executor: BatchExecutor | None = None,
    ) -> DatasetGenerationResult:
        """Sample, batch-evaluate, and persist a strictly bound dataset."""

        candidates = sample_candidates(self.design_space, plan)
        order_sha256 = stable_payload_sha256(
            [candidate.candidate_id for candidate in candidates]
        )
        batch_request = BatchEvaluationRequest(
            design_space_id=self.design_space.design_space_id,
            objective_set_id=self.evaluator.objectives.objective_set_id,
            candidates=candidates,
            metadata={"sampling_plan_id": plan.plan_id},
        )
        dataset_id = content_id(
            "dataset",
            {
                "plan_id": plan.plan_id,
                "design_space_id": self.design_space.design_space_id,
                "objective_set_id": self.evaluator.objectives.objective_set_id,
                "batch_id": batch_request.batch_id,
                "candidate_order_sha256": order_sha256,
            },
        )
        root = Path(config.output_root).expanduser().resolve()
        manifest_path = root / "DATASET_MANIFEST.json"
        index_path = root / "DATASET_INDEX.jsonl"
        if root.exists() and any(root.iterdir()):
            if not config.resume:
                raise FileExistsError(
                    "dataset output already exists; enable resume to continue"
                )
            manifest = _load_dataset_manifest(manifest_path)
            _validate_manifest_binding(
                manifest,
                dataset_id=dataset_id,
                plan=plan,
                batch_id=batch_request.batch_id,
                design_space_id=self.design_space.design_space_id,
                objective_set_id=self.evaluator.objectives.objective_set_id,
                candidate_count=len(candidates),
                candidate_order_sha256=order_sha256,
                root=root,
            )
            records = _load_dataset_index(
                index_path,
                dataset_id=dataset_id,
                candidates=candidates,
            )
            created_at = manifest.created_at
        else:
            root.mkdir(parents=True, exist_ok=True)
            _initialize_index(index_path)
            records = {}
            created_at = _utc_now()
            _persist_manifest(
                manifest_path,
                dataset_id=dataset_id,
                plan=plan,
                batch_id=batch_request.batch_id,
                design_space_id=self.design_space.design_space_id,
                objective_set_id=self.evaluator.objectives.objective_set_id,
                candidate_count=len(candidates),
                candidate_order_sha256=order_sha256,
                records=records.values(),
                status="running",
                created_at=created_at,
            )

        if len(records) < len(candidates):
            active_executor = executor or _DatasetSequentialExecutor(config.cache)
            try:
                batch_result = self.evaluator.evaluate_many(
                    batch_request,
                    executor=active_executor,
                    resume=config.resume,
                    output_dir=root / "evaluations",
                )
                evaluations = _load_batch_records(
                    batch_result.artifact_root,
                    batch_request,
                )
                for index, candidate in enumerate(candidates):
                    if index in records:
                        continue
                    dataset_record = self._dataset_record(
                        dataset_id,
                        plan,
                        candidate,
                        evaluations[index],
                    )
                    _validate_dataset_record_artifacts(dataset_record)
                    _append_dataset_index(index_path, index, dataset_record)
                    records[index] = dataset_record
                    _persist_manifest(
                        manifest_path,
                        dataset_id=dataset_id,
                        plan=plan,
                        batch_id=batch_request.batch_id,
                        design_space_id=self.design_space.design_space_id,
                        objective_set_id=self.evaluator.objectives.objective_set_id,
                        candidate_count=len(candidates),
                        candidate_order_sha256=order_sha256,
                        records=records.values(),
                        status="running",
                        created_at=created_at,
                    )
            except Exception:
                _persist_manifest(
                    manifest_path,
                    dataset_id=dataset_id,
                    plan=plan,
                    batch_id=batch_request.batch_id,
                    design_space_id=self.design_space.design_space_id,
                    objective_set_id=self.evaluator.objectives.objective_set_id,
                    candidate_count=len(candidates),
                    candidate_order_sha256=order_sha256,
                    records=records.values(),
                    status="interrupted",
                    created_at=created_at,
                )
                raise

        ordered = tuple(records[index] for index in range(len(candidates)))
        final_status = _dataset_status(ordered)
        index_ref = _artifact_ref(index_path, root)
        _persist_manifest(
            manifest_path,
            dataset_id=dataset_id,
            plan=plan,
            batch_id=batch_request.batch_id,
            design_space_id=self.design_space.design_space_id,
            objective_set_id=self.evaluator.objectives.objective_set_id,
            candidate_count=len(candidates),
            candidate_order_sha256=order_sha256,
            records=ordered,
            status=final_status,
            created_at=created_at,
            index_artifact=index_ref,
        )
        artifact_refs = tuple(
            ResearchArtifactRef.model_validate(item)
            for item in index_artifacts(root)
            if item["kind"] in {"research_dataset_manifest", "research_dataset_index"}
        )
        validate_artifact_references(
            [item.model_dump(mode="python") for item in artifact_refs], root=root
        )
        if {item.kind for item in artifact_refs} != {
            "research_dataset_manifest",
            "research_dataset_index",
        }:
            raise ValueError("dataset manifest and index artifact references are required")
        return build_dataset_result(
            dataset_id=dataset_id,
            plan_id=plan.plan_id,
            records=ordered,
            artifact_root=root,
            artifacts=artifact_refs,
            preview_limit=config.preview_limit,
        )

    def _dataset_record(
        self,
        dataset_id: str,
        plan: SamplingPlan,
        candidate: DesignCandidate,
        evaluation: EvaluationRecord,
    ) -> DatasetRecord:
        if evaluation.candidate_id != candidate.candidate_id:
            raise ValueError("evaluation candidate identity does not match sampling order")
        task = self.design_space.to_simulation_task(candidate)
        accepted = (
            evaluation.status == "completed"
            and evaluation.physics_accepted
            and evaluation.certificate_id is not None
        )
        verification_status: Literal["accepted", "rejected", "failed"]
        if accepted:
            verification_status = "accepted"
        elif evaluation.run_id is not None and not evaluation.physics_accepted:
            verification_status = "rejected"
        else:
            verification_status = "failed"
        selected_outputs = tuple(
            dict.fromkeys(
                [
                    item.observable
                    for item in (
                        *self.evaluator.objectives.objectives,
                        *self.evaluator.objectives.constraints,
                    )
                ]
            )
        )
        failure_codes = tuple(
            str(item.get("code"))
            for item in evaluation.failures
            if isinstance(item.get("code"), str)
        )
        if candidate.sample_index is None:
            raise ValueError("sampled candidate is missing sample_index provenance")
        return DatasetRecord(
            dataset_id=dataset_id,
            plan_id=plan.plan_id,
            candidate_id=candidate.candidate_id,
            sample_index=candidate.sample_index,
            seed=plan.seed,
            design_variables=candidate.values,
            normalized_design=candidate.normalized_design,
            task_sha256=evaluation.task_sha256,
            run_id=evaluation.run_id,
            material_catalog_sha256=evaluation.material_catalog_sha256,
            material_identities=_material_identities(task),
            wavelength=_wavelength_config(task),
            requested_outputs=tuple(task.requested_outputs),
            selected_outputs=selected_outputs,
            evaluation_status=evaluation.status,
            verification_status=verification_status,
            physics_accepted=accepted,
            certificate_id=evaluation.certificate_id if accepted else None,
            veritmm_version=__version__,
            archive_schema_version=ARCHIVE_SCHEMA_VERSION,
            artifact_root=evaluation.artifact_root,
            artifacts=evaluation.artifacts,
            provenance={
                "sampling_strategy": plan.strategy,
                "sampling_plan_id": plan.plan_id,
                "design_space_id": self.design_space.design_space_id,
                "objective_set_id": self.evaluator.objectives.objective_set_id,
                "cache_hit": evaluation.cache_hit,
                "source_run_id": evaluation.source_run_id,
                "artifact_provenance": evaluation.artifact_provenance,
            },
            failure_codes=failure_codes,
        )


def build_dataset_result(
    *,
    dataset_id: str,
    plan_id: str,
    records: Iterable[DatasetRecord],
    artifact_root: str | Path,
    artifacts: Iterable[ResearchArtifactRef | Mapping[str, Any]] = (),
    preview_limit: int = 12,
) -> DatasetGenerationResult:
    """Build a compact response whose preview is independent of dataset size."""

    if not 1 <= preview_limit <= MAX_DATASET_PREVIEW:
        raise ValueError("preview_limit is outside the bounded dataset limit")
    record_list = list(records)
    accepted = sum(item.verification_status == "accepted" for item in record_list)
    rejected = sum(item.verification_status == "rejected" for item in record_list)
    failed = len(record_list) - accepted - rejected
    status = _dataset_status(record_list)
    refs = [
        item.model_dump(mode="json")
        if isinstance(item, ResearchArtifactRef)
        else dict(item)
        for item in artifacts
    ]
    preview = [
        {
            "candidate_id": record.candidate_id,
            "sample_index": record.sample_index,
            "verification_status": record.verification_status,
            "physics_accepted": record.physics_accepted,
            "run_id": record.run_id,
            "certificate_id": record.certificate_id,
        }
        for record in record_list[:preview_limit]
    ]
    raw = {
        "schema_version": DATASET_RESULT_SCHEMA_VERSION,
        "ok": status == "completed",
        "dataset_id": dataset_id,
        "plan_id": plan_id,
        "status": status,
        "record_count": len(record_list),
        "accepted_count": accepted,
        "rejected_count": rejected,
        "failed_count": failed,
        "preview": preview,
        "preview_count": len(preview),
        "truncated_count": len(record_list) - len(preview),
        "artifact_root": str(Path(artifact_root).resolve()),
        "artifacts": refs,
    }
    projected = project_response(raw, detail="compact")
    guard_context_budget(projected, detail="compact", max_bytes=COMPACT_MAX_BYTES)
    return DatasetGenerationResult.model_validate_json(
        json.dumps(projected, ensure_ascii=False, allow_nan=False)
    )


def _material_identities(task: SimulationTask) -> tuple[DatasetMaterialIdentity, ...]:
    identities = [_medium_identity("incident", task.stack.incident)]
    identities.extend(
        _layer_identity(index, layer) for index, layer in enumerate(task.stack.layers)
    )
    identities.append(_medium_identity("exit", task.stack.exit))
    return tuple(identities)


def _medium_identity(position: str, medium: MediumSpec) -> DatasetMaterialIdentity:
    return DatasetMaterialIdentity(
        position=position,
        material=medium.material,
        provider=medium.provider,
        dataset_id=medium.dataset_id,
        constant_n=medium.constant_n,
        constant_k=medium.constant_k,
    )


def _layer_identity(index: int, layer: LayerSpec) -> DatasetMaterialIdentity:
    return DatasetMaterialIdentity(
        position="layer",
        layer_index=index,
        material=layer.material,
        provider=layer.provider,
        dataset_id=layer.dataset_id,
        constant_n=layer.constant_n,
        constant_k=layer.constant_k,
        thickness_nm=layer.thickness_nm,
    )


def _wavelength_config(task: SimulationTask) -> DatasetWavelengthConfig:
    grid = task.spectrum
    if grid.values_nm is not None:
        values = tuple(float(value) for value in grid.values_nm)
        return DatasetWavelengthConfig(
            mode="explicit",
            start_nm=min(values),
            stop_nm=max(values),
            point_count=len(values),
            configuration_sha256=stable_payload_sha256(values),
        )
    if grid.start_nm is None or grid.stop_nm is None or grid.points is None:
        raise ValueError("simulation wavelength configuration is incomplete")
    return DatasetWavelengthConfig(
        mode="linspace",
        start_nm=grid.start_nm,
        stop_nm=grid.stop_nm,
        point_count=grid.points,
        configuration_sha256=stable_payload_sha256(
            {
                "start_nm": grid.start_nm,
                "stop_nm": grid.stop_nm,
                "points": grid.points,
            }
        ),
    )


def _load_batch_records(
    artifact_root: str | Path,
    request: BatchEvaluationRequest,
) -> dict[int, EvaluationRecord]:
    root = Path(artifact_root).resolve()
    path = root / "BATCH_INDEX.jsonl"
    records: dict[int, EvaluationRecord] = {}
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
            raise ValueError("batch index entry must be an object")
        index = payload.get("index")
        if (
            payload.get("schema_version") != BATCH_INDEX_SCHEMA_VERSION
            or payload.get("batch_id") != request.batch_id
            or isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(request.candidates)
        ):
            raise ValueError("batch index binding is invalid")
        if index in records:
            raise ValueError("batch index contains duplicate sample indices")
        expected_id = request.candidates[index].candidate_id
        if payload.get("candidate_id") != expected_id:
            raise ValueError("batch index candidate order mismatch")
        try:
            record = EvaluationRecord.model_validate_json(
                json.dumps(payload.get("record"), ensure_ascii=False, allow_nan=False)
            )
        except Exception as exc:
            raise ValueError("batch evaluation record is invalid") from exc
        if record.candidate_id != expected_id:
            raise ValueError("batch evaluation candidate identity mismatch")
        records[index] = record
    if len(records) != len(request.candidates):
        raise ValueError("batch index does not contain every requested candidate")
    return records


def _initialize_index(path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(b"")
    temporary.replace(path)


def _append_dataset_index(path: Path, index: int, record: DatasetRecord) -> None:
    payload = {
        "schema_version": DATASET_INDEX_SCHEMA_VERSION,
        "dataset_id": record.dataset_id,
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
            raise OSError("short write while appending dataset index")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_dataset_index(
    path: Path,
    *,
    dataset_id: str,
    candidates: tuple[DesignCandidate, ...],
) -> dict[int, DatasetRecord]:
    if not path.is_file():
        raise ValueError("dataset index is missing")
    records: dict[int, DatasetRecord] = {}
    candidate_ids: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"dataset index is unreadable: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"dataset index line {line_number} is corrupt") from exc
        if not isinstance(payload, dict):
            raise ValueError("dataset index entry must be an object")
        index = payload.get("index")
        if payload.get("schema_version") != DATASET_INDEX_SCHEMA_VERSION:
            raise ValueError("dataset index schema_version is invalid")
        if payload.get("dataset_id") != dataset_id:
            raise ValueError("dataset index is bound to a different dataset")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(candidates)
        ):
            raise ValueError("dataset index sample index is invalid")
        expected_id = candidates[index].candidate_id
        if payload.get("candidate_id") != expected_id:
            raise ValueError("dataset index candidate ordering mismatch")
        if index in records or expected_id in candidate_ids:
            raise ValueError("dataset index contains duplicate records")
        try:
            record = DatasetRecord.model_validate_json(
                json.dumps(payload.get("record"), ensure_ascii=False, allow_nan=False)
            )
        except Exception as exc:
            raise ValueError(f"dataset record {index} is invalid") from exc
        if (
            record.dataset_id != dataset_id
            or record.candidate_id != expected_id
            or record.sample_index != index
        ):
            raise ValueError("dataset record identity binding mismatch")
        _validate_dataset_record_artifacts(record)
        records[index] = record
        candidate_ids.add(expected_id)
    return records


def _load_dataset_manifest(path: Path) -> DatasetManifest:
    try:
        return DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"dataset manifest is invalid or unreadable: {exc}") from exc


def _validate_manifest_binding(
    manifest: DatasetManifest,
    *,
    dataset_id: str,
    plan: SamplingPlan,
    batch_id: str,
    design_space_id: str,
    objective_set_id: str,
    candidate_count: int,
    candidate_order_sha256: str,
    root: Path,
) -> None:
    expected = {
        "dataset_id": dataset_id,
        "plan_id": plan.plan_id,
        "strategy": plan.strategy,
        "design_space_id": design_space_id,
        "objective_set_id": objective_set_id,
        "batch_id": batch_id,
        "candidate_count": candidate_count,
        "candidate_order_sha256": candidate_order_sha256,
        "veritmm_version": __version__,
        "archive_schema_version": ARCHIVE_SCHEMA_VERSION,
    }
    for key, value in expected.items():
        if getattr(manifest, key) != value:
            raise ValueError(f"dataset manifest binding mismatch: {key}")
    if manifest.index_artifact is not None:
        validate_artifact_references(
            (manifest.index_artifact.model_dump(mode="python"),), root=root
        )


def _persist_manifest(
    path: Path,
    *,
    dataset_id: str,
    plan: SamplingPlan,
    batch_id: str,
    design_space_id: str,
    objective_set_id: str,
    candidate_count: int,
    candidate_order_sha256: str,
    records: Iterable[DatasetRecord],
    status: Literal["running", "interrupted", "completed", "partial", "failed"],
    created_at: str,
    index_artifact: Mapping[str, Any] | None = None,
) -> DatasetManifest:
    record_list = list(records)
    accepted = sum(item.verification_status == "accepted" for item in record_list)
    rejected = sum(item.verification_status == "rejected" for item in record_list)
    failed = len(record_list) - accepted - rejected
    manifest = DatasetManifest(
        dataset_id=dataset_id,
        plan_id=plan.plan_id,
        strategy=plan.strategy,
        design_space_id=design_space_id,
        objective_set_id=objective_set_id,
        batch_id=batch_id,
        candidate_count=candidate_count,
        candidate_order_sha256=candidate_order_sha256,
        status=status,
        processed_count=len(record_list),
        accepted_count=accepted,
        rejected_count=rejected,
        failed_count=failed,
        veritmm_version=__version__,
        archive_schema_version=ARCHIVE_SCHEMA_VERSION,
        created_at=created_at,
        updated_at=_utc_now(),
        provenance={
            "generator": "DatasetFactory",
            "sampling_plan_schema_version": plan.schema_version,
            "evaluation_path": "ResearchEvaluator.evaluate_many",
            "physics_validity_source": "EvaluationRecord",
        },
        index_artifact=(
            None
            if index_artifact is None
            else ResearchArtifactRef.model_validate(index_artifact)
        ),
    )
    write_json(path, manifest.model_dump(mode="json"))
    return manifest


def _validate_dataset_record_artifacts(record: DatasetRecord) -> None:
    if not record.artifacts:
        return
    if record.artifact_root is None:
        raise ValueError("dataset record artifact_root is missing")
    validate_artifact_references(
        [item.model_dump(mode="python") for item in record.artifacts],
        root=record.artifact_root,
    )


def _dataset_status(
    records: Iterable[DatasetRecord],
) -> Literal["completed", "partial", "failed"]:
    record_list = list(records)
    accepted = sum(item.verification_status == "accepted" for item in record_list)
    if accepted == len(record_list):
        return "completed"
    if accepted == 0:
        return "failed"
    return "partial"


def _artifact_ref(path: Path, root: Path) -> dict[str, Any]:
    return {
        "kind": "research_dataset_index",
        "path": path.relative_to(root).as_posix(),
        "schema_version": DATASET_INDEX_SCHEMA_VERSION,
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


__all__ = [
    "DATASET_CONFIG_SCHEMA_VERSION",
    "DATASET_INDEX_SCHEMA_VERSION",
    "DATASET_MANIFEST_SCHEMA_VERSION",
    "DATASET_RECORD_SCHEMA_VERSION",
    "DATASET_RESULT_SCHEMA_VERSION",
    "DatasetConfig",
    "DatasetFactory",
    "DatasetGenerationResult",
    "DatasetManifest",
    "DatasetMaterialIdentity",
    "DatasetRecord",
    "DatasetRecordPreview",
    "DatasetWavelengthConfig",
    "build_dataset_result",
]
