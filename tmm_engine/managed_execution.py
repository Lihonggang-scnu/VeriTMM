"""Study-aware execution wrapper around the deterministic physics runtime."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import __version__
from .execution import ExecutionSettings, execute_task
from .experiment_store import ExperimentStore
from .protocol.models import (
    PROTOCOL_VERSION,
    SensitivityTaskPayload,
    SweepTaskPayload,
    ToleranceTaskPayload,
)
from .protocol.responses import (
    DEFAULT_RESPONSE_DETAIL,
    RESPONSE_CONTEXT_SCHEMA_VERSION,
)
from .run_artifacts import (
    file_sha256,
    stable_payload_sha256,
)
from .schemas import OptimizationTask, SimulationTask, dataclass_to_dict
from .sweep import SweepExecutionSettings, execute_sweep


def material_catalog_identity() -> str:
    """Hash every bundled material source used by the local registry."""

    root = Path(__file__).resolve().parent
    records: list[dict[str, Any]] = []
    for path in sorted([root / "rii_cache.db", *(root / "materials").glob("*.csv")]):
        if path.is_file():
            records.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": file_sha256(path),
                }
            )
    return stable_payload_sha256(records)


def normalized_operation(mode: str, task: Any) -> dict[str, Any]:
    if mode in {"simulate", "optimize"}:
        return {
            "mode": mode,
            "simulation" if mode == "simulate" else "optimization": dataclass_to_dict(task),
        }
    if mode == "sweep" and isinstance(task, SweepTaskPayload):
        return {
            "schema_version": "sweep-task-v1",
            "mode": "sweep",
            "sweep": task.model_dump(mode="json"),
        }
    if mode == "sensitivity" and isinstance(task, SensitivityTaskPayload):
        return {
            "schema_version": "sensitivity-task-v1",
            "mode": "sensitivity",
            "sensitivity": task.model_dump(mode="json"),
        }
    if mode == "tolerance" and isinstance(task, ToleranceTaskPayload):
        return {
            "schema_version": "tolerance-task-v1",
            "mode": "tolerance",
            "tolerance": task.model_dump(mode="json"),
        }
    raise TypeError(f"unsupported managed task: {mode}/{type(task).__name__}")


def _settings_payload(
    mode: str,
    execution_settings: ExecutionSettings,
    *,
    resume: bool,
) -> dict[str, Any]:
    payload = asdict(execution_settings)
    payload["mode"] = mode
    # Resume affects orchestration, not numerical identity.
    payload["study_protocol"] = "veritmm-study-v1"
    payload["response_context_schema"] = RESPONSE_CONTEXT_SCHEMA_VERSION
    return payload


def _prepare_sweep_child_records(
    store: ExperimentStore,
    *,
    envelope: Mapping[str, Any],
    output_dir: Path,
    experiment_id: str,
    execution_identity_sha256: str,
) -> list[dict[str, Any]]:
    path = output_dir / "SWEEP_RESULT.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    parent_run_id = str(envelope["run_id"])
    records: list[dict[str, Any]] = []
    for child in payload.get("children", []):
        child_run_id = child.get("child_run_id")
        relative_root = child.get("artifact_root")
        if not child_run_id:
            if relative_root and child.get("status") in {"failed", "pending"}:
                continue
            raise ValueError("sweep child without run_id must be explicitly failed or pending")
        if not relative_root:
            raise ValueError("sweep child with run_id requires artifact_root")
        child_root = (output_dir / str(relative_root)).resolve()
        children_root = (output_dir / "children").resolve()
        try:
            child_root.relative_to(children_root)
        except ValueError as exc:
            raise ValueError("sweep child artifact_root must stay under children/") from exc
        if child_root == children_root or child_root.parent != children_root:
            raise ValueError("sweep child artifact_root must name one direct child directory")
        if not child_root.is_dir():
            raise FileNotFoundError(f"sweep child artifact directory missing: {relative_root}")
        archived = store.archive_artifacts(child_root, str(child_run_id))
        child_result_path = child_root / "RUN_RESULT.json"
        if child_result_path.is_file():
            child_envelope = json.loads(child_result_path.read_text(encoding="utf-8"))
        else:
            child_envelope = {
                "run_id": child_run_id,
                "task_sha256": child.get("child_task_sha256"),
                "operation": "simulate",
                "status": child.get("status", "unknown"),
                "protocol_version": PROTOCOL_VERSION,
                "certificate_id": None,
            }
        records.append(
            store.envelope_record_fields(
                child_envelope,
                artifact_root=archived,
                experiment_id=experiment_id,
                parent_run_id=parent_run_id,
                execution_identity_sha256=stable_payload_sha256(
                    {
                        "parent_identity": execution_identity_sha256,
                        "child_task_sha256": child.get("child_task_sha256"),
                    }
                ),
                tags=("sweep_child",),
                user_metadata={"sweep_index": child.get("index")},
            )
        )
    return records


def _record_managed_group(
    store: ExperimentStore,
    *,
    mode: str,
    envelope: Mapping[str, Any],
    output_dir: Path,
    archived_parent: Path,
    experiment_id: str,
    parent_run_id: str | None,
    execution_identity_sha256: str,
    tags: Iterable[str],
    hypothesis: str | None,
    change_reason: str | None,
    user_metadata: Mapping[str, Any] | None,
) -> None:
    staged_run_ids = [str(envelope["run_id"])]
    try:
        if mode == "sweep":
            sweep_path = output_dir / "SWEEP_RESULT.json"
            if sweep_path.is_file():
                sweep_payload = json.loads(sweep_path.read_text(encoding="utf-8"))
                staged_run_ids.extend(
                    str(child["child_run_id"])
                    for child in sweep_payload.get("children", [])
                    if isinstance(child, Mapping) and child.get("child_run_id")
                )
        records = [
            store.envelope_record_fields(
                envelope,
                artifact_root=archived_parent,
                experiment_id=experiment_id,
                parent_run_id=parent_run_id,
                execution_identity_sha256=execution_identity_sha256,
                tags=tags,
                hypothesis=hypothesis,
                change_reason=change_reason,
                user_metadata=user_metadata,
            )
        ]
        if mode == "sweep":
            records.extend(
                _prepare_sweep_child_records(
                    store,
                    envelope=envelope,
                    output_dir=output_dir,
                    experiment_id=experiment_id,
                    execution_identity_sha256=execution_identity_sha256,
                )
            )
        store.record_run_batch(records)
    except Exception:
        store.discard_unindexed_artifacts(staged_run_ids)
        raise


def execute_managed_task(
    mode: str,
    task: SimulationTask
    | OptimizationTask
    | SweepTaskPayload
    | SensitivityTaskPayload
    | ToleranceTaskPayload,
    output_dir: str | Path,
    *,
    input_path: str | Path | None = None,
    execution_settings: ExecutionSettings | None = None,
    store: ExperimentStore | None = None,
    experiment_id: str | None = None,
    parent_run_id: str | None = None,
    tags: Iterable[str] = (),
    hypothesis: str | None = None,
    change_reason: str | None = None,
    user_metadata: Mapping[str, Any] | None = None,
    cache: bool = True,
    resume: bool = False,
    detail: str = DEFAULT_RESPONSE_DETAIL,
) -> dict[str, Any]:
    """Execute, cache, archive and index one public task invocation."""

    settings = execution_settings or ExecutionSettings()
    if resume and mode != "sweep":
        raise ValueError("--resume is currently supported only for sweep tasks")
    ExperimentStore.assert_no_path_redirection(output_dir)
    output_path = Path(output_dir).absolute()
    if output_path.is_dir():
        ExperimentStore.assert_artifact_tree_no_links(output_path)
    output = output_path.resolve()
    normalized = normalized_operation(mode, task)
    identity = ExperimentStore.execution_identity(
        normalized,
        package_version=__version__,
        protocol_version=PROTOCOL_VERSION,
        material_catalog_sha256=material_catalog_identity(),
        execution_settings=_settings_payload(mode, settings, resume=resume),
    )
    exp_id = experiment_id or (store.new_experiment_id() if store else "")
    if store is not None and cache and not resume:
        source = store.find_cache_source(identity)
        if source is not None:
            new_run_id = store.new_run_id()
            envelope = store.materialize_cache_hit(
                source,
                output,
                new_run_id=new_run_id,
                detail=detail,
            )
            archived = store.archive_artifacts(output, new_run_id)
            _record_managed_group(
                store,
                mode=mode,
                envelope=envelope,
                output_dir=output,
                archived_parent=archived,
                experiment_id=exp_id,
                parent_run_id=parent_run_id,
                execution_identity_sha256=identity,
                tags=tags,
                hypothesis=hypothesis,
                change_reason=change_reason,
                user_metadata=user_metadata,
            )
            return envelope

    if mode in {"simulate", "optimize"}:
        envelope = execute_task(
            mode,
            task,
            output,
            input_path=input_path,
            settings=settings,
            detail=detail,
        )
    elif mode == "sweep" and isinstance(task, SweepTaskPayload):
        envelope = execute_sweep(
            task,
            output,
            settings=SweepExecutionSettings(
                child_execution=settings,
                resume=resume,
            ),
            detail=detail,
        )
    elif mode == "sensitivity" and isinstance(task, SensitivityTaskPayload):
        from .scientific_analysis import execute_sensitivity

        envelope = execute_sensitivity(task, output, device=settings.device, detail=detail)
    elif mode == "tolerance" and isinstance(task, ToleranceTaskPayload):
        from .scientific_analysis import execute_tolerance

        envelope = execute_tolerance(task, output, detail=detail)
    else:
        raise TypeError(f"unsupported managed task: {mode}/{type(task).__name__}")

    if store is not None:
        archived = store.archive_artifacts(output, str(envelope["run_id"]))
        _record_managed_group(
            store,
            mode=mode,
            envelope=envelope,
            output_dir=output,
            archived_parent=archived,
            experiment_id=exp_id,
            parent_run_id=parent_run_id,
            execution_identity_sha256=identity,
            tags=tags,
            hypothesis=hypothesis,
            change_reason=change_reason,
            user_metadata=user_metadata,
        )
    return envelope


__all__ = [
    "execute_managed_task",
    "material_catalog_identity",
    "normalized_operation",
]
