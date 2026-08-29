"""Framework-neutral A/B harness for agent-use trajectories.

No model SDK is imported here.  A caller may provide any callable adapter or
score trajectories produced by an external coding agent.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .agent_bench import BenchmarkCase
from .agent_bench.interactive_env import (
    InteractiveAgentPolicy,
    InteractiveCase,
    InteractiveEnv,
    InteractiveEnvEpisode,
)
from .managed_execution import normalized_operation
from .preflight import preflight_path
from .protocol.capabilities import describe_capabilities
from .protocol.models import RunResultEnvelope
from .protocol.responses import DEFAULT_RESPONSE_DETAIL, project_response
from .protocol.schema_export import export_schema
from .run_artifacts import file_sha256, stable_payload_sha256, write_json
from .task_io import load_task

TRAJECTORY_SCHEMA_VERSION = "veritmm-agent-trajectory-v1"
AGENT_AB_RESULT_SCHEMA_VERSION = "veritmm-agent-ab-result-v1"


class TrajectoryStep(BaseModel):
    model_config = ConfigDict(extra="allow")

    index: int = Field(ge=0)
    action: str
    observation: Any = None


class AgentTrajectory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = TRAJECTORY_SCHEMA_VERSION
    benchmark_case: str
    model: str | None = None
    agent_version: str | None = None
    exposure: Literal["traditional", "agent_native"]
    prompt: str
    steps: list[TrajectoryStep] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    correction_turns: int = Field(default=0, ge=0)
    final_run_id: str | None = None
    certificate_id: str | None = None
    physics_certificate_expected: bool | None = None
    success: bool
    first_task_valid: bool | None = None
    preflight_failures: int | None = None
    physics_failures: int | None = None
    unsupported_false_accept: bool | None = None
    reproducible: bool | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    wall_seconds: float | None = None
    notes: list[str] = Field(default_factory=list)
    task_attempts: list[dict[str, Any]] = Field(default_factory=list)
    final_run_result: dict[str, Any] | None = None


class AgentRunner(Protocol):
    """Minimal adapter contract; proprietary SDKs stay in the caller."""

    def __call__(
        self, case: BenchmarkCase, exposure: Mapping[str, Any]
    ) -> AgentTrajectory | Mapping[str, Any]: ...


def build_exposure(
    case: BenchmarkCase, kind: Literal["traditional", "agent_native"]
) -> dict[str, Any]:
    """Build the controlled information package for one A/B arm."""

    common = {
        "exposure": kind,
        "natural_language_task": case.natural_language_task,
        "expected_deliverable": "A valid VeriTMM task and an auditable result.",
    }
    if kind == "traditional":
        return {
            **common,
            "available_resources": ["README.md", "Python API", "basic examples"],
            "protocol_tools": [],
        }
    schema_kind = {
        "simulate": "simulation",
        "optimize": "optimization",
    }.get(case.expected_mode, case.expected_mode)
    return {
        **common,
        "available_resources": [
            "describe",
            "schema",
            "preflight",
            "typed failure actions",
            "RUN_RESULT.json",
            "ExperimentStore",
            "study commands",
        ],
        "capabilities": describe_capabilities().model_dump(mode="json"),
        "task_schema": export_schema(schema_kind),
        "protocol_tools": [
            "describe",
            "schema",
            "preflight",
            "run",
            "history",
            "inspect",
            "compare",
        ],
    }


def run_agent_ab(
    cases: Iterable[BenchmarkCase],
    runner: AgentRunner,
    *,
    output_path: str | Path | None = None,
    detail: str = DEFAULT_RESPONSE_DETAIL,
) -> dict[str, Any]:
    """Run the same cases with traditional and agent-native exposure."""

    trajectories: list[AgentTrajectory] = []
    ordered_cases = sorted(list(cases), key=lambda item: item.case_id)
    for case in ordered_cases:
        for exposure in ("traditional", "agent_native"):
            started = time.perf_counter()
            raw = runner(case, build_exposure(case, exposure))
            trajectory = (
                raw if isinstance(raw, AgentTrajectory) else AgentTrajectory.model_validate(raw)
            )
            if trajectory.benchmark_case != case.case_id or trajectory.exposure != exposure:
                raise ValueError("runner returned a trajectory for the wrong case or exposure")
            if trajectory.wall_seconds is None:
                trajectory.wall_seconds = float(time.perf_counter() - started)
            trajectories.append(trajectory)
    return score_trajectories(
        trajectories,
        cases={case.case_id: case for case in ordered_cases},
        output_path=output_path,
        detail=detail,
    )


def run_interactive_episode(
    case: InteractiveCase | Mapping[str, Any],
    agent_policy: InteractiveAgentPolicy,
    *,
    max_steps: int = 20,
) -> InteractiveEnvEpisode:
    """Run one agent-agnostic policy through a stepwise environment."""

    environment = InteractiveEnv(case, max_steps=max_steps)
    while not environment.done and len(environment.steps) < max_steps:
        request = environment.request()
        try:
            action = agent_policy(request)
            environment.step(action)
        except Exception as exc:
            environment.terminate("fail", f"{type(exc).__name__}: {exc}")
            break
    if not environment.done:
        environment.terminate("timeout", "max_steps reached")
    return environment.episode()


def _rate(values: list[bool]) -> float | None:
    return None if not values else float(sum(int(item) for item in values) / len(values))


def _mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2.0)


def _preflight_attempts(trajectory: AgentTrajectory) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    if not trajectory.task_attempts:
        return reports
    with tempfile.TemporaryDirectory(prefix="veritmm_trajectory_audit_") as temporary:
        root = Path(temporary)
        for index, task in enumerate(trajectory.task_attempts):
            task_path = root / f"attempt_{index:03d}.json"
            write_json(task_path, task)
            reports.append(preflight_path(task_path))
    return reports


def _audit_run_evidence(trajectory: AgentTrajectory) -> dict[str, Any]:
    """Validate an embedded run envelope against the final submitted task."""

    evidence: dict[str, Any] = {
        "present": trajectory.final_run_result is not None,
        "schema_valid": False,
        "task_hash_match": False,
        "operation_match": False,
        "certificate_consistent": False,
        "valid": False,
        "diagnostic": None,
        "envelope": None,
    }
    if trajectory.final_run_result is None:
        return evidence
    try:
        envelope = RunResultEnvelope.model_validate(trajectory.final_run_result)
        evidence["schema_valid"] = True
        evidence["envelope"] = envelope.model_dump(mode="json")
        if not trajectory.task_attempts:
            evidence["diagnostic"] = "final run evidence has no corresponding task attempt"
            return evidence
        with tempfile.TemporaryDirectory(prefix="veritmm_run_evidence_audit_") as temporary:
            task_path = Path(temporary) / "final_task.json"
            write_json(task_path, trajectory.task_attempts[-1])
            mode, task = load_task(task_path)
        expected_hash = stable_payload_sha256(normalized_operation(mode, task))
        evidence["expected_task_sha256"] = expected_hash
        evidence["task_hash_match"] = envelope.task_sha256 == expected_hash
        evidence["operation_match"] = envelope.operation == mode
        physics = envelope.summary.get("physics") or {}
        if envelope.certificate_id is None:
            evidence["certificate_consistent"] = not bool(physics.get("accepted"))
        else:
            evidence["certificate_consistent"] = bool(
                physics.get("accepted") and physics.get("certificate_id") == envelope.certificate_id
            )
        evidence["valid"] = bool(
            evidence["schema_valid"]
            and evidence["task_hash_match"]
            and evidence["operation_match"]
            and evidence["certificate_consistent"]
        )
    except Exception as exc:
        evidence["diagnostic"] = f"{type(exc).__name__}: {exc}"
    return evidence


def _audit_trajectory(
    trajectory: AgentTrajectory,
    case: BenchmarkCase | None,
) -> tuple[AgentTrajectory, dict[str, Any]]:
    reports = _preflight_attempts(trajectory)
    run_evidence = _audit_run_evidence(trajectory)
    update: dict[str, Any] = {}
    provenance: dict[str, str] = {}
    if reports:
        if case is None or case.expected_capability == "supported":
            update["first_task_valid"] = bool(reports[0].get("ok"))
            provenance["first_task_valid"] = "recomputed_from_task_attempts"
        update["preflight_failures"] = sum(int(not item.get("ok")) for item in reports)
        update["correction_turns"] = max(0, len(reports) - 1)
        provenance["preflight_failures"] = "recomputed_from_task_attempts"
        provenance["correction_turns"] = "recomputed_from_task_attempts"

    if run_evidence["valid"]:
        result = run_evidence["envelope"]
        update["final_run_id"] = result.get("run_id")
        update["certificate_id"] = result.get("certificate_id")
        update["physics_failures"] = len(result.get("failures") or [])
        provenance["final_run_id"] = "recomputed_from_final_run_result"
        provenance["certificate_id"] = "recomputed_from_final_run_result"
        provenance["physics_failures"] = "recomputed_from_final_run_result"

    if case is not None:
        update["physics_certificate_expected"] = "physics_certificate" in case.expected_artifacts
        provenance["physics_certificate_expected"] = "benchmark_case_contract"
        if reports:
            final = reports[-1]
            if case.expected_capability == "supported":
                ready = bool(final.get("ok")) and final.get("mode") == case.expected_mode
                if case.execution == "run":
                    ready = ready and bool(
                        run_evidence["valid"]
                        and run_evidence["envelope"]
                        and run_evidence["envelope"].get("ok")
                    )
                update["success"] = ready
                update["unsupported_false_accept"] = False
            else:
                observed = {str(item.get("code")) for item in final.get("failures", [])}
                rejected_as_expected = not bool(final.get("ok")) and observed == set(
                    case.expected_failure_codes
                )
                update["success"] = rejected_as_expected
                update["unsupported_false_accept"] = bool(final.get("ok"))
            provenance["success"] = "recomputed_from_case_and_evidence"
            provenance["unsupported_false_accept"] = "recomputed_from_case_and_evidence"

    audited = trajectory.model_copy(update=update)
    return audited, {
        "benchmark_case": trajectory.benchmark_case,
        "exposure": trajectory.exposure,
        "task_attempt_count": len(trajectory.task_attempts),
        "preflight_reports": reports,
        "run_evidence": {key: value for key, value in run_evidence.items() if key != "envelope"},
        "metric_provenance": provenance,
    }


def score_trajectories(
    trajectories: Iterable[AgentTrajectory | Mapping[str, Any]],
    *,
    cases: Mapping[str, BenchmarkCase] | None = None,
    output_path: str | Path | None = None,
    detail: str = DEFAULT_RESPONSE_DETAIL,
) -> dict[str, Any]:
    """Score trajectories and independently audit supplied task/result evidence."""

    parsed = [
        item if isinstance(item, AgentTrajectory) else AgentTrajectory.model_validate(item)
        for item in trajectories
    ]
    if not parsed:
        raise ValueError("at least one agent trajectory is required")
    audited_pairs = [
        _audit_trajectory(item, None if cases is None else cases.get(item.benchmark_case))
        for item in parsed
    ]
    parsed = [item for item, _ in audited_pairs]
    audit_records = [audit for _, audit in audited_pairs]
    arms: dict[str, Any] = {}
    for exposure in ("traditional", "agent_native"):
        rows = [item for item in parsed if item.exposure == exposure]
        first = [item.first_task_valid for item in rows if item.first_task_valid is not None]
        corrections = [float(item.correction_turns) for item in rows]
        unsupported = [
            bool(item.unsupported_false_accept)
            for item in rows
            if item.unsupported_false_accept is not None
        ]
        reproducible = [item.reproducible for item in rows if item.reproducible is not None]
        arms[exposure] = {
            "trajectory_count": len(rows),
            "first_valid_task_rate": _rate([bool(item) for item in first]),
            "final_success_rate": _rate([item.success for item in rows]),
            "mean_correction_turns": _mean(corrections),
            "median_correction_turns": _median(corrections),
            "mean_tool_calls": _mean([float(len(item.tool_calls)) for item in rows]),
            "mean_input_tokens": _mean(
                [float(item.input_tokens) for item in rows if item.input_tokens is not None]
            ),
            "mean_output_tokens": _mean(
                [float(item.output_tokens) for item in rows if item.output_tokens is not None]
            ),
            "mean_wall_seconds": _mean(
                [float(item.wall_seconds) for item in rows if item.wall_seconds is not None]
            ),
            "mean_preflight_failures": _mean(
                [
                    float(item.preflight_failures)
                    for item in rows
                    if item.preflight_failures is not None
                ]
            ),
            "mean_physics_failures": _mean(
                [float(item.physics_failures) for item in rows if item.physics_failures is not None]
            ),
            "unsupported_false_accept_rate": _rate(unsupported),
            "certified_success_rate": _rate(
                [
                    item.success and item.certificate_id is not None
                    for item in rows
                    if item.physics_certificate_expected is True
                ]
            ),
            "reproducibility_rate": _rate([bool(item) for item in reproducible]),
        }
    result = {
        "schema_version": AGENT_AB_RESULT_SCHEMA_VERSION,
        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
        "status": "completed",
        "arms": arms,
        "independent_audit": audit_records,
        "trajectories": [item.model_dump(mode="json") for item in parsed],
    }
    if output_path is not None:
        # The optional file is the complete audit artifact.  The returned
        # machine-facing mapping is projected separately below.
        target = Path(output_path).resolve()
        write_json(target, result)
        result["artifact_root"] = str(target.parent)
        result["artifacts"] = [
            {
                "kind": "agent_benchmark_result",
                "path": target.name,
                "schema_version": AGENT_AB_RESULT_SCHEMA_VERSION,
                "sha256": file_sha256(target),
                "size_bytes": int(target.stat().st_size),
            }
        ]
    return project_response(result, detail=detail)


def load_trajectories(path: str | Path) -> list[AgentTrajectory]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, Mapping) and "trajectories" in payload:
        rows = payload["trajectories"]
    elif (
        isinstance(payload, Mapping) and payload.get("schema_version") == TRAJECTORY_SCHEMA_VERSION
    ):
        rows = [payload]
    else:
        raise ValueError(
            "trajectory input must be one AgentTrajectory object, a non-empty list, "
            "or an object with a non-empty trajectories list"
        )
    if not isinstance(rows, list) or not rows:
        raise ValueError("trajectory input contains no trajectories")
    return [AgentTrajectory.model_validate(item) for item in rows]


__all__ = [
    "AGENT_AB_RESULT_SCHEMA_VERSION",
    "TRAJECTORY_SCHEMA_VERSION",
    "AgentRunner",
    "AgentTrajectory",
    "TrajectoryStep",
    "build_exposure",
    "load_trajectories",
    "run_interactive_episode",
    "run_agent_ab",
    "score_trajectories",
]
