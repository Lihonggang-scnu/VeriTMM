"""Formal, resumable finite parameter sweeps for planar TMM simulations."""

from __future__ import annotations

import csv
import itertools
import json
import uuid
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .archive.schema_registry import ARCHIVE_SCHEMA_VERSION
from .capabilities import failure_from_exception
from .execution import ExecutionSettings, execute_task
from .protocol.models import PROTOCOL_VERSION, SweepTaskPayload
from .protocol.responses import DEFAULT_RESPONSE_DETAIL
from .run_artifacts import (
    build_result_summary,
    prepare_output_directory,
    stable_payload_sha256,
    write_json,
    write_run_result,
)
from .study_metrics import evaluate_metric
from .task_io import simulation_task_from_dict

SWEEP_RESULT_SCHEMA_VERSION = "veritmm-sweep-result-v1"


@dataclass(frozen=True)
class SweepExecutionSettings:
    child_execution: ExecutionSettings = ExecutionSettings(write_plot=False)
    resume: bool = False
    stop_after_children: int | None = None


def _set_pointer(payload: dict[str, Any], pointer: str, value: float | int) -> None:
    tokens = [token.replace("~1", "/").replace("~0", "~") for token in pointer.split("/")[1:]]
    current: Any = payload
    for token in tokens[:-1]:
        current = current[int(token)] if isinstance(current, list) else current[token]
    leaf = tokens[-1]
    if isinstance(current, list):
        current[int(leaf)] = value
    else:
        current[leaf] = value


def expand_sweep(sweep: SweepTaskPayload) -> list[dict[str, Any]]:
    """Expand axes in declaration order and values in supplied order."""

    axes = list(sweep.parameters)
    rows: list[dict[str, Any]] = []
    for index, values in enumerate(itertools.product(*(axis.values for axis in axes))):
        parameters = [
            {"path": axis.path, "value": value}
            for axis, value in zip(axes, values, strict=True)
        ]
        simulation = deepcopy(sweep.base_simulation.model_dump(mode="python"))
        for item in parameters:
            _set_pointer(simulation, item["path"], item["value"])
        normalized = {"mode": "simulate", "simulation": simulation}
        rows.append(
            {
                "index": index,
                "parameters": parameters,
                "simulation": simulation,
                "child_task_sha256": stable_payload_sha256(normalized),
            }
        )
    return rows


def _existing_children(output: Path, task_sha256: str) -> dict[int, dict[str, Any]]:
    path = output / "SWEEP_RESULT.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("task_sha256") != task_sha256:
        raise ValueError("resume sweep task hash differs from existing study")
    return {
        int(item["index"]): item
        for item in payload.get("children", [])
        if item.get("ok") is True and item.get("status") == "completed"
    }


def _write_table(path: Path, children: list[dict[str, Any]], metric_names: list[str]) -> None:
    parameter_paths = sorted(
        {item["path"] for child in children for item in child.get("parameters", [])}
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["index", "child_run_id", "status", "ok", *parameter_paths, *metric_names]
        )
        for child in sorted(children, key=lambda item: int(item["index"])):
            parameters = {item["path"]: item["value"] for item in child.get("parameters", [])}
            metrics = child.get("metrics", {})
            writer.writerow(
                [
                    child["index"],
                    child.get("child_run_id"),
                    child.get("status"),
                    child.get("ok"),
                    *(parameters.get(path) for path in parameter_paths),
                    *(metrics.get(name) for name in metric_names),
                ]
            )


def execute_sweep(
    sweep: SweepTaskPayload,
    output_dir: str | Path,
    *,
    settings: SweepExecutionSettings | None = None,
    run_id: str | None = None,
    detail: str = DEFAULT_RESPONSE_DETAIL,
) -> dict[str, Any]:
    """Run or resume a finite sweep while preserving every child outcome."""

    config = settings or SweepExecutionSettings()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    normalized = {
        "schema_version": "sweep-task-v1",
        "mode": "sweep",
        "sweep": sweep.model_dump(mode="json"),
    }
    task_sha256 = stable_payload_sha256(normalized)
    previous = _existing_children(output, task_sha256) if config.resume else {}
    previous_parent = None
    existing_run_result = output / "RUN_RESULT.json"
    if config.resume and existing_run_result.is_file():
        previous_parent = json.loads(existing_run_result.read_text(encoding="utf-8")).get("run_id")
    parent_run_id = run_id or previous_parent or f"run_{uuid.uuid4().hex}"
    prepare_output_directory(output, preserve_sweep_children=config.resume)
    write_json(output / "NORMALIZED_TASK.json", normalized)

    children: list[dict[str, Any]] = []
    executed_now = 0
    interrupted = False
    for row in expand_sweep(sweep):
        index = int(row["index"])
        cached = previous.get(index)
        if cached is not None and cached.get("child_task_sha256") == row["child_task_sha256"]:
            children.append(cached)
            continue
        if config.stop_after_children is not None and executed_now >= config.stop_after_children:
            interrupted = True
            children.append(
                {
                    **{key: row[key] for key in ("index", "parameters", "child_task_sha256")},
                    "status": "pending",
                    "ok": False,
                    "metrics": {},
                    "failures": [],
                }
            )
            continue
        child_dir = output / "children" / f"{index:06d}_{row['child_task_sha256'][:12]}"
        try:
            simulation = simulation_task_from_dict(row["simulation"])
            child_envelope = execute_task(
                "simulate",
                simulation,
                child_dir,
                settings=config.child_execution,
                detail=detail,
            )
            result_path = child_dir / "SIMULATION_RESULT.json"
            metrics: dict[str, float] = {}
            if child_envelope.get("ok") and result_path.is_file():
                result_payload = json.loads(result_path.read_text(encoding="utf-8"))
                metrics = {
                    metric.name: evaluate_metric(result_payload, metric)
                    for metric in sweep.metrics
                }
            child = {
                "index": index,
                "parameters": row["parameters"],
                "child_task_sha256": row["child_task_sha256"],
                "child_run_id": child_envelope.get("run_id"),
                "status": "completed" if child_envelope.get("ok") else "failed",
                "ok": bool(child_envelope.get("ok")),
                "metrics": metrics,
                "failures": child_envelope.get("failures", []),
                "artifact_root": child_dir.relative_to(output).as_posix(),
            }
        except Exception as exc:
            failure = failure_from_exception(exc).to_dict()
            child = {
                "index": index,
                "parameters": row["parameters"],
                "child_task_sha256": row["child_task_sha256"],
                "child_run_id": None,
                "status": "failed",
                "ok": False,
                "metrics": {},
                "failures": [failure],
                "artifact_root": child_dir.relative_to(output).as_posix(),
            }
        children.append(child)
        executed_now += 1
        # Checkpoint after every child so an external interruption loses no study state.
        checkpoint = {
            "schema_version": SWEEP_RESULT_SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "run_id": parent_run_id,
            "task_sha256": task_sha256,
            "archive_schema_version": ARCHIVE_SCHEMA_VERSION,
            "status": "running",
            "children": sorted(children, key=lambda item: int(item["index"])),
        }
        write_json(output / "SWEEP_RESULT.json", checkpoint)

    children.sort(key=lambda item: int(item["index"]))
    successful = sum(1 for child in children if child.get("ok"))
    failed = sum(1 for child in children if child.get("status") == "failed")
    pending = sum(1 for child in children if child.get("status") == "pending")
    status = "interrupted" if interrupted or pending else ("completed" if successful else "failed")
    sweep_result = {
        "schema_version": SWEEP_RESULT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "run_id": parent_run_id,
        "task_sha256": task_sha256,
        "archive_schema_version": ARCHIVE_SCHEMA_VERSION,
        "status": status,
        "ok": status == "completed" and failed == 0,
        "partial_success": successful > 0 and (failed > 0 or pending > 0),
        "child_count": len(children),
        "successful_child_count": successful,
        "failed_child_count": failed,
        "pending_child_count": pending,
        "resumed_child_count": len(previous),
        "executed_child_count": executed_now,
        "metrics": [metric.model_dump(mode="json") for metric in sweep.metrics],
        "children": children,
    }
    write_json(output / "SWEEP_RESULT.json", sweep_result)
    _write_table(output / "SWEEP_TABLE.csv", children, [metric.name for metric in sweep.metrics])
    summary = build_result_summary(
        mode="sweep",
        forward=None,
        certificate=None,
        run_id=parent_run_id,
        task_sha256=task_sha256,
        run_status=status,
    )
    summary["sweep"] = {
        key: sweep_result[key]
        for key in (
            "child_count",
            "successful_child_count",
            "failed_child_count",
            "pending_child_count",
            "partial_success",
        )
    }
    write_json(output / "RESULT_SUMMARY.json", summary)
    write_json(
        output / "RUN_MANIFEST.json",
        {
            "schema_version": "veritmm-run-manifest-v1",
            "mode": "sweep",
            "status": status,
            "run_id": parent_run_id,
            "task_sha256": task_sha256,
            "archive_schema_version": ARCHIVE_SCHEMA_VERSION,
        },
    )
    failures = [
        failure
        for child in children
        for failure in child.get("failures", [])
    ]
    return write_run_result(
        output,
        operation="sweep",
        task_sha256=task_sha256,
        status=status,
        ok=bool(sweep_result["ok"]),
        summary=summary,
        failures=failures,
        run_id=parent_run_id,
        detail=detail,
    )


__all__ = [
    "SWEEP_RESULT_SCHEMA_VERSION",
    "SweepExecutionSettings",
    "execute_sweep",
    "expand_sweep",
]
