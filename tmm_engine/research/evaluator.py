"""Verified objective extraction over VeriTMM managed simulation runs.

The evaluator deliberately has no solver, workbench, or certifier dependency.
Every candidate follows the existing managed execution path, and optimizer
scores remain separate from the resulting physics acceptance certificate.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Literal, Mapping

import numpy as np
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

from ..execution import ExecutionSettings
from ..experiment_store import ExperimentStore
from ..managed_execution import execute_managed_task, material_catalog_identity
from ..protocol.models import RunResultEnvelope
from ..protocol.responses import validate_artifact_references
from ..run_artifacts import validate_run_artifact_integrity
from ..schemas import SimulationTask, dataclass_to_dict
from .contracts import DesignCandidate, ResearchModel
from .design_space import DesignSpace
from .objectives import (
    ConstraintSpec,
    ConstraintStatus,
    ObjectiveScore,
    ObjectiveSet,
    ObjectiveSpec,
    ObjectiveValue,
)

EVALUATION_RECORD_SCHEMA_VERSION = "veritmm-research-evaluation-v1"
EVALUATOR_CONFIG_SCHEMA_VERSION = "veritmm-research-evaluator-config-v1"


class ResearchArtifactRef(ResearchModel):
    """Portable integrity reference rooted at ``EvaluationRecord.artifact_root``."""

    kind: StrictStr
    path: StrictStr
    schema_version: StrictStr
    sha256: StrictStr
    size_bytes: StrictInt

    @model_validator(mode="after")
    def _valid_ref(self) -> "ResearchArtifactRef":
        validate_artifact_references((self.model_dump(mode="python"),))
        return self


class EvaluatorConfig(ResearchModel):
    """Serializable policy for managed research evaluations."""

    schema_version: Literal[EVALUATOR_CONFIG_SCHEMA_VERSION] = (
        EVALUATOR_CONFIG_SCHEMA_VERSION
    )
    output_root: StrictStr
    cache: StrictBool = True
    detail: Literal["compact"] = "compact"
    experiment_id: StrictStr | None = None
    parent_run_id: StrictStr | None = None
    tags: tuple[StrictStr, ...] = ()
    hypothesis: StrictStr | None = None
    change_reason: StrictStr | None = None
    user_metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("output_root")
    @classmethod
    def _nonempty_root(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("output_root must not be empty")
        return value


class EvaluationRecord(ResearchModel):
    """Compact verified research result without inline spectral arrays.

    ``total_score`` and ``feasible`` describe only the declared research
    objective.  They never confer physics validity; ``physics_accepted`` is
    copied exclusively from a validated VeriTMM certificate.
    """

    schema_version: Literal[EVALUATION_RECORD_SCHEMA_VERSION] = (
        EVALUATION_RECORD_SCHEMA_VERSION
    )
    response_profile: Literal["compact"] = "compact"
    candidate_id: StrictStr
    design_space_id: StrictStr
    objective_set_id: StrictStr
    status: Literal["completed", "failed"]
    failure_stage: Literal[
        "candidate_validation",
        "managed_execution",
        "artifact_integrity",
        "certificate_validation",
        "objective_extraction",
    ] | None = None
    objective_values: tuple[ObjectiveValue, ...] = ()
    objective_scores: tuple[ObjectiveScore, ...] = ()
    constraint_statuses: tuple[ConstraintStatus, ...] = ()
    total_score: StrictFloat | None = None
    feasible: StrictBool | None = None
    physics_accepted: StrictBool = False
    certificate_id: StrictStr | None = None
    run_id: StrictStr | None = None
    task_sha256: StrictStr | None = None
    material_catalog_sha256: StrictStr
    cache_hit: StrictBool = False
    source_run_id: StrictStr | None = None
    artifact_provenance: dict[str, JsonValue] | None = None
    artifact_root: StrictStr | None = None
    artifacts: tuple[ResearchArtifactRef, ...] = ()
    warnings: tuple[dict[str, JsonValue], ...] = ()
    failures: tuple[dict[str, JsonValue], ...] = ()

    @model_validator(mode="after")
    def _coherent_status(self) -> "EvaluationRecord":
        if self.status == "completed":
            if self.failure_stage is not None or self.failures:
                raise ValueError("completed evaluation cannot contain a failure stage")
            if not self.physics_accepted or self.certificate_id is None:
                raise ValueError("completed evaluation requires an accepted certificate")
            if self.run_id is None or self.task_sha256 is None or self.artifact_root is None:
                raise ValueError("completed evaluation requires run and artifact identity")
            if self.total_score is None or self.feasible is None:
                raise ValueError("completed evaluation requires score and feasibility")
            if not self.objective_values or not self.objective_scores:
                raise ValueError("completed evaluation requires objective values and scores")
        else:
            if self.failure_stage is None or not self.failures:
                raise ValueError("failed evaluation requires an actionable failure")
            if self.objective_values or self.objective_scores or self.constraint_statuses:
                raise ValueError("failed evaluation cannot expose unverified objective data")
            if self.total_score is not None or self.feasible is not None:
                raise ValueError("failed evaluation cannot expose score or feasibility")
        if self.cache_hit and self.source_run_id is None:
            raise ValueError("cache hit requires source_run_id provenance")
        if self.artifacts and self.artifact_root is None:
            raise ValueError("artifact references require artifact_root")
        return self


class ResearchEvaluator:
    """Evaluate design candidates through ``execute_managed_task('simulate')``."""

    def __init__(
        self,
        design_space: DesignSpace,
        objectives: ObjectiveSet,
        config: EvaluatorConfig,
        *,
        execution_settings: ExecutionSettings | None = None,
        store: ExperimentStore | None = None,
    ) -> None:
        if not isinstance(design_space, DesignSpace):
            raise TypeError("design_space must be a DesignSpace")
        if not isinstance(objectives, ObjectiveSet):
            raise TypeError("objectives must be an ObjectiveSet")
        self.design_space = design_space
        self.objectives = objectives
        self.config = config
        self.execution_settings = execution_settings or ExecutionSettings()
        self.store = store

    def evaluate(
        self,
        candidate: DesignCandidate | Mapping[str, Any],
        *,
        cache: bool | None = None,
        experiment_metadata: Mapping[str, Any] | None = None,
        output_root: str | Path | None = None,
    ) -> EvaluationRecord:
        """Evaluate one candidate and fail closed at every trust boundary."""

        catalog_sha256 = material_catalog_identity()
        candidate_id = self._candidate_id(candidate)
        envelope: dict[str, Any] | None = None
        invocation: Path | None = None
        task: SimulationTask | None = None
        validated: DesignCandidate | None = None
        try:
            validated = self.design_space.validate_candidate(candidate)
            candidate_id = validated.candidate_id
            task = self.design_space.to_simulation_task(validated)
        except Exception as exc:
            return self._failure_record(
                candidate_id=candidate_id,
                material_catalog_sha256=catalog_sha256,
                stage="candidate_validation",
                code="candidate_invalid",
                message=str(exc),
            )

        try:
            invocation = self._new_invocation_root(
                validated.candidate_id,
                root=Path(output_root) if output_root is not None else None,
            )
            user_metadata = dict(self.config.user_metadata)
            user_metadata.update(dict(experiment_metadata or {}))
            user_metadata["research"] = {
                "candidate_id": validated.candidate_id,
                "design_space_id": self.design_space.design_space_id,
                "objective_set_id": self.objectives.objective_set_id,
            }
            envelope = execute_managed_task(
                "simulate",
                task,
                invocation,
                execution_settings=self.execution_settings,
                store=self.store,
                experiment_id=self.config.experiment_id,
                parent_run_id=self.config.parent_run_id,
                tags=self.config.tags,
                hypothesis=self.config.hypothesis,
                change_reason=self.config.change_reason,
                user_metadata=user_metadata,
                cache=self.config.cache if cache is None else cache,
                detail="compact",
            )
        except Exception as exc:
            return self._failure_record(
                candidate_id=validated.candidate_id,
                material_catalog_sha256=catalog_sha256,
                stage="managed_execution",
                code="managed_execution_failed",
                message=str(exc),
                artifact_root=invocation,
            )

        try:
            RunResultEnvelope.model_validate(envelope)
            common = self._envelope_fields(envelope, invocation, catalog_sha256)
            refs = self._validated_refs(envelope, invocation)
        except Exception as exc:
            return self._failure_record(
                candidate_id=validated.candidate_id,
                stage="artifact_integrity",
                code="artifact_integrity_failed",
                message=str(exc),
                material_catalog_sha256=catalog_sha256,
                artifact_root=invocation,
            )

        common["artifacts"] = refs
        if envelope.get("ok") is not True or envelope.get("status") != "completed":
            return self._failure_record(
                candidate_id=validated.candidate_id,
                stage="managed_execution",
                code="simulation_not_successful",
                message=f"managed simulation status was {envelope.get('status')!r}",
                **common,
            )
        try:
            certificate = self._load_accepted_certificate(envelope, invocation, refs, task)
        except Exception as exc:
            return self._failure_record(
                candidate_id=validated.candidate_id,
                stage="certificate_validation",
                code="certificate_invalid",
                message=str(exc),
                **common,
            )

        common["physics_accepted"] = True
        common["certificate_id"] = str(certificate["certificate_id"])
        try:
            simulation = self._load_simulation_result(invocation, refs)
            values, scores, constraints, total, feasible = self._extract_objectives(
                task, simulation
            )
        except Exception as exc:
            return self._failure_record(
                candidate_id=validated.candidate_id,
                stage="objective_extraction",
                code="objective_unavailable",
                message=str(exc),
                **common,
            )

        common.pop("envelope_failures", None)
        return EvaluationRecord(
            candidate_id=validated.candidate_id,
            design_space_id=self.design_space.design_space_id,
            objective_set_id=self.objectives.objective_set_id,
            status="completed",
            objective_values=values,
            objective_scores=scores,
            constraint_statuses=constraints,
            total_score=total,
            feasible=feasible,
            **common,
        )

    def evaluate_many(
        self,
        request: Any,
        *,
        executor: Any = None,
        resume: bool = True,
        output_dir: str | Path | None = None,
        batch_size: int | None = None,
    ) -> Any:
        """Evaluate a batch through the replaceable batch-executor contract.

        ``batch_size`` selects the chunked verified executor.  It may batch an
        optional differentiable proposal forward, but the reference solver and
        physics certificate still run independently for every candidate.
        """

        from .batch import ChunkedVerifiedBatchExecutor, SequentialBatchExecutor, evaluate_batch

        if batch_size is not None and executor is not None:
            raise ValueError("batch_size cannot be combined with an explicit executor")
        if batch_size is not None:
            active_executor = ChunkedVerifiedBatchExecutor(batch_size)
        else:
            active_executor = SequentialBatchExecutor() if executor is None else executor

        return evaluate_batch(
            self,
            request,
            executor=active_executor,
            resume=resume,
            output_dir=output_dir,
        )

    def _batch_forward_proposals(self, candidates: tuple[DesignCandidate, ...]) -> dict[str, Any]:
        """Optionally execute a batched differentiable proposal forward.

        This method is deliberately advisory: its output is never used to
        construct an EvaluationRecord or certificate.  Missing PyTorch,
        mixed material stacks, incoherent layers, or incompatible grids simply
        fall back to the independent verified path.
        """

        if not candidates:
            return {"used": False, "reason": "empty_chunk"}
        try:
            import torch

            from ..differentiable import DifferentiableTMM
            from ..material_registry import MaterialRegistry
            from ..workbench import TMMWorkbench
        except ImportError:
            return {"used": False, "reason": "torch_unavailable"}

        try:
            tasks = [
                self.design_space.to_simulation_task(candidate)
                for candidate in candidates
            ]
            reference = tasks[0]
            if (
                reference.stack.has_incoherent_layers
                or len(reference.illumination.angles_deg) != 1
                or len(reference.illumination.polarizations) != 1
                or set(reference.requested_outputs) - {"R", "T", "A"}
            ):
                return {"used": False, "reason": "incompatible_task_shape"}
            wavelengths = reference.spectrum.wavelengths_nm()
            if any(
                task.stack.has_incoherent_layers
                or task.spectrum.wavelengths_nm().shape != wavelengths.shape
                or not np.array_equal(task.spectrum.wavelengths_nm(), wavelengths)
                or task.illumination != reference.illumination
                or len(task.stack.layers) != len(reference.stack.layers)
                for task in tasks[1:]
            ):
                return {"used": False, "reason": "incompatible_candidate_pool"}

            workbench = TMMWorkbench(MaterialRegistry())
            resolved: list[list[np.ndarray]] = []
            for task in tasks:
                media, _, _ = workbench._resolve_stack(task)
                resolved.append([np.asarray(item, dtype=np.complex128) for item in media])
            first_media = resolved[0]
            if any(
                len(media) != len(first_media)
                or any(not np.array_equal(item, reference_item) for item, reference_item in zip(media, first_media, strict=True))
                for media in resolved[1:]
            ):
                return {"used": False, "reason": "material_sequence_varies"}

            thicknesses_um = torch.tensor(
                [[float(layer.thickness_nm) * 1e-3 for layer in task.stack.layers] for task in tasks],
                dtype=torch.float64,
            )
            nk = torch.tensor(
                np.stack(first_media, axis=0), dtype=torch.complex128
            ).unsqueeze(0).expand(len(tasks), -1, -1).contiguous()
            wavelengths_um = torch.tensor(wavelengths * 1e-3, dtype=torch.float64)
            model = DifferentiableTMM(
                polarization=str(reference.illumination.polarizations[0])
            )
            with torch.no_grad():
                model(
                    thicknesses_um,
                    nk,
                    wavelengths_um,
                    theta_rad=np.deg2rad(float(reference.illumination.angles_deg[0])),
                )
            return {"used": True, "candidate_count": len(tasks)}
        except Exception as exc:  # pragma: no cover - optional acceleration path
            return {"used": False, "reason": f"batch_forward_unavailable:{type(exc).__name__}"}

    def _new_invocation_root(self, candidate_id: str, *, root: Path | None) -> Path:
        parent = (
            root
            if root is not None
            else Path(self.config.output_root).expanduser() / "evaluations"
        )
        parent.mkdir(parents=True, exist_ok=True)
        safe_id = "".join(char for char in candidate_id if char.isalnum())
        safe_id = (safe_id.rsplit("candidate", 1)[-1][:8] or "candidate")
        for _ in range(10):
            # Keep substantial headroom for atomic artifact temporary names on
            # Windows while retaining independently random invocation identity.
            invocation = parent / f"e_{safe_id}_{uuid.uuid4().hex[:12]}"
            try:
                invocation.mkdir()
                return invocation
            except FileExistsError:  # pragma: no cover - UUID collision guard
                continue
        raise RuntimeError("could not allocate a unique evaluation directory")

    @staticmethod
    def _candidate_id(candidate: DesignCandidate | Mapping[str, Any]) -> str:
        if isinstance(candidate, DesignCandidate):
            return candidate.candidate_id
        value = candidate.get("candidate_id") if isinstance(candidate, Mapping) else None
        return str(value) if isinstance(value, str) and value else "invalid_candidate"

    def _envelope_fields(
        self,
        envelope: Mapping[str, Any],
        invocation: Path,
        catalog_sha256: str,
    ) -> dict[str, Any]:
        provenance = envelope.get("artifact_provenance")
        return {
            "material_catalog_sha256": catalog_sha256,
            "run_id": self._optional_string(envelope.get("run_id")),
            "task_sha256": self._optional_string(envelope.get("task_sha256")),
            "certificate_id": self._optional_string(envelope.get("certificate_id")),
            "cache_hit": envelope.get("cache_hit") is True,
            "source_run_id": self._optional_string(envelope.get("source_run_id")),
            "artifact_provenance": (
                dict(provenance) if isinstance(provenance, Mapping) else None
            ),
            "artifact_root": str(invocation.resolve()),
            "warnings": self._json_mappings(envelope.get("warnings")),
            "envelope_failures": self._json_mappings(envelope.get("failures")),
        }

    def _failure_record(
        self,
        *,
        candidate_id: str,
        stage: Literal[
            "candidate_validation",
            "managed_execution",
            "artifact_integrity",
            "certificate_validation",
            "objective_extraction",
        ],
        code: str,
        message: str,
        material_catalog_sha256: str | None = None,
        run_id: str | None = None,
        task_sha256: str | None = None,
        certificate_id: str | None = None,
        cache_hit: bool = False,
        source_run_id: str | None = None,
        artifact_provenance: Mapping[str, Any] | None = None,
        artifact_root: str | Path | None = None,
        artifacts: Iterable[ResearchArtifactRef] = (),
        warnings: Iterable[Mapping[str, Any]] = (),
        envelope_failures: Iterable[Mapping[str, Any]] = (),
        physics_accepted: bool = False,
    ) -> EvaluationRecord:
        failures = [dict(item) for item in envelope_failures]
        failures.append({"code": code, "message": self._bounded_message(message)})
        return EvaluationRecord(
            candidate_id=candidate_id,
            design_space_id=self.design_space.design_space_id,
            objective_set_id=self.objectives.objective_set_id,
            status="failed",
            failure_stage=stage,
            physics_accepted=physics_accepted,
            certificate_id=certificate_id,
            run_id=run_id,
            task_sha256=task_sha256,
            material_catalog_sha256=(
                material_catalog_sha256 or material_catalog_identity()
            ),
            cache_hit=cache_hit,
            source_run_id=source_run_id,
            artifact_provenance=(
                None if artifact_provenance is None else dict(artifact_provenance)
            ),
            artifact_root=None if artifact_root is None else str(Path(artifact_root).resolve()),
            artifacts=tuple(artifacts),
            warnings=tuple(dict(item) for item in warnings),
            failures=tuple(failures),
        )

    @staticmethod
    def _validated_refs(
        envelope: Mapping[str, Any], invocation: Path
    ) -> tuple[ResearchArtifactRef, ...]:
        validate_run_artifact_integrity(invocation)
        validate_artifact_references(envelope, root=invocation)
        raw_refs = envelope.get("artifacts")
        if not isinstance(raw_refs, list):
            raise ValueError("managed run artifacts must be a list")
        return tuple(ResearchArtifactRef.model_validate(item) for item in raw_refs)

    @classmethod
    def _load_accepted_certificate(
        cls,
        envelope: Mapping[str, Any],
        root: Path,
        refs: tuple[ResearchArtifactRef, ...],
        task: SimulationTask,
    ) -> dict[str, Any]:
        certificate_ref = cls._one_artifact(refs, "physics_certificate")
        certificate = cls._read_json_object(root / PurePosixPath(certificate_ref.path))
        if certificate.get("schema_version") != "physics-acceptance-certificate-v1":
            raise ValueError("certificate schema_version is invalid")
        if certificate.get("accepted") is not True:
            raise ValueError("physics certificate was not accepted")
        certificate_id = certificate.get("certificate_id")
        if not isinstance(certificate_id, str) or not certificate_id:
            raise ValueError("physics certificate_id is missing")
        if certificate_id != envelope.get("certificate_id"):
            raise ValueError("physics certificate_id does not match the run envelope")

        unsigned = dict(certificate)
        unsigned.pop("certificate_id", None)
        expected_certificate_id = hashlib.sha256(
            json.dumps(
                unsigned,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        if certificate_id != expected_certificate_id:
            raise ValueError("physics certificate_id does not match certificate content")
        task_id = hashlib.sha256(
            json.dumps(
                dataclass_to_dict(task),
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        if certificate.get("task_sha256") != task_id:
            raise ValueError("physics certificate is bound to a different simulation task")
        return certificate

    @classmethod
    def _load_simulation_result(
        cls, root: Path, refs: tuple[ResearchArtifactRef, ...]
    ) -> dict[str, Any]:
        result_ref = cls._one_artifact(refs, "simulation_result")
        return cls._read_json_object(root / PurePosixPath(result_ref.path))

    @staticmethod
    def _one_artifact(
        refs: tuple[ResearchArtifactRef, ...], kind: str
    ) -> ResearchArtifactRef:
        matching = [item for item in refs if item.kind == kind]
        if len(matching) != 1:
            raise ValueError(f"managed run requires exactly one {kind} artifact")
        return matching[0]

    @staticmethod
    def _read_json_object(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"research source artifact is unreadable: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("research source artifact must contain a JSON object")
        return payload

    def _extract_objectives(
        self, task: SimulationTask, simulation: Mapping[str, Any]
    ) -> tuple[
        tuple[ObjectiveValue, ...],
        tuple[ObjectiveScore, ...],
        tuple[ConstraintStatus, ...],
        float,
        bool,
    ]:
        wavelengths = self._finite_vector(simulation.get("wavelengths_nm"), "wavelengths_nm")
        channels = simulation.get("channels")
        if not isinstance(channels, Mapping):
            raise ValueError("simulation result channels are unavailable")

        objective_values: list[ObjectiveValue] = []
        objective_scores: list[ObjectiveScore] = []
        total = 0.0
        for objective in self.objectives.objectives:
            value = self._aggregate_spec(task, wavelengths, channels, objective)
            score = self._objective_score(objective, value)
            weighted = objective.weight * score
            objective_values.append(
                ObjectiveValue(objective_name=objective.name, value=value)
            )
            objective_scores.append(
                ObjectiveScore(
                    objective_name=objective.name,
                    value=value,
                    score=score,
                    weighted_score=weighted,
                )
            )
            total += weighted

        statuses: list[ConstraintStatus] = []
        for constraint in self.objectives.constraints:
            value = self._aggregate_spec(task, wavelengths, channels, constraint)
            satisfied = (
                value + constraint.tolerance >= constraint.threshold
                if constraint.relation == "at_least"
                else value - constraint.tolerance <= constraint.threshold
            )
            statuses.append(
                ConstraintStatus(
                    constraint_name=constraint.name,
                    relation=constraint.relation,
                    value=value,
                    threshold=constraint.threshold,
                    tolerance=constraint.tolerance,
                    satisfied=satisfied,
                )
            )
        return (
            tuple(objective_values),
            tuple(objective_scores),
            tuple(statuses),
            float(total),
            all(item.satisfied for item in statuses),
        )

    @classmethod
    def _aggregate_spec(
        cls,
        task: SimulationTask,
        wavelengths: np.ndarray,
        channels: Mapping[str, Any],
        spec: ObjectiveSpec | ConstraintSpec,
    ) -> float:
        if spec.observable not in task.requested_outputs:
            raise ValueError(
                f"observable {spec.observable!r} was not requested by the simulation task"
            )
        channel_key = f"angle={float(spec.angle_deg):g}|pol={spec.polarization}"
        channel = channels.get(channel_key)
        if not isinstance(channel, Mapping):
            raise ValueError(f"simulation channel is unavailable: {channel_key}")
        values = cls._finite_vector(channel.get(spec.observable), spec.observable)
        if values.shape != wavelengths.shape:
            raise ValueError("observable length does not match wavelength grid")
        mask = (wavelengths >= spec.wavelength_min_nm) & (
            wavelengths <= spec.wavelength_max_nm
        )
        if not np.any(mask):
            raise ValueError(
                f"wavelength band has no sampled points: "
                f"[{spec.wavelength_min_nm}, {spec.wavelength_max_nm}] nm"
            )
        selected = values[mask]
        if spec.aggregation == "mean":
            return float(np.mean(selected))
        if spec.aggregation == "min":
            return float(np.min(selected))
        return float(np.max(selected))

    @staticmethod
    def _objective_score(spec: ObjectiveSpec, value: float) -> float:
        if spec.direction == "maximize":
            return value
        if spec.direction == "minimize":
            return -value
        if spec.target is None:  # pragma: no cover - ObjectiveSpec invariant
            raise ValueError("target objective is missing target")
        return -abs(value - spec.target)

    @staticmethod
    def _finite_vector(value: Any, field_name: str) -> np.ndarray:
        try:
            vector = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be a numeric array") from exc
        if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
            raise ValueError(f"{field_name} must be a finite non-empty vector")
        return vector

    @staticmethod
    def _json_mappings(value: Any) -> tuple[dict[str, Any], ...]:
        if not isinstance(value, (list, tuple)):
            return ()
        return tuple(dict(item) for item in value if isinstance(item, Mapping))

    @staticmethod
    def _optional_string(value: Any) -> str | None:
        return value if isinstance(value, str) and value else None

    @staticmethod
    def _bounded_message(value: str, limit: int = 1024) -> str:
        text = str(value)
        return text if len(text) <= limit else f"{text[: limit - 14]}...[truncated]"


__all__ = [
    "EVALUATION_RECORD_SCHEMA_VERSION",
    "EVALUATOR_CONFIG_SCHEMA_VERSION",
    "EvaluationRecord",
    "EvaluatorConfig",
    "ResearchArtifactRef",
    "ResearchEvaluator",
]
