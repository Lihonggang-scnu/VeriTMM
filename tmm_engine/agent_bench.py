"""Deterministic offline benchmark for VeriTMM's agent-facing protocol.

The benchmark is deliberately outside the physics acceptance path.  It may
observe preflight and execution artifacts, but it cannot relax a capability
boundary, alter a task, or manufacture a physics certificate.
"""

from __future__ import annotations

import copy
import json
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .execution import ExecutionSettings
from .experiment_store import ExperimentStore
from .managed_execution import execute_managed_task
from .preflight import preflight_path
from .protocol.responses import DEFAULT_RESPONSE_DETAIL, project_response
from .run_artifacts import file_sha256, stable_payload_sha256, write_json
from .task_io import load_task

BENCHMARK_SCHEMA_VERSION = "veritmm-agentbench-v1"
BENCHMARK_CASE_SCHEMA_VERSION = "veritmm-agentbench-case-v1"

# Keep the historical single-module import stable while allowing optional
# interactive benchmark submodules under ``tmm_engine.agent_bench``.
__path__ = [str(Path(__file__).with_suffix(""))]


class BenchmarkAssertion(BaseModel):
    """One small, deterministic assertion over a JSON benchmark artifact."""

    model_config = ConfigDict(extra="forbid")

    source: Literal[
        "preflight",
        "run_result",
        "summary",
        "certificate",
        "simulation_result",
        "optimization_result",
        "sweep_result",
        "sensitivity_result",
        "tolerance_result",
        "robustness_report",
    ] = "summary"
    path: str
    operator: Literal["exists", "not_empty", "eq", "ne", "ge", "gt", "le", "lt", "between"] = (
        "exists"
    )
    expected: Any = None
    minimum: float | None = None
    maximum: float | None = None
    tolerance: float = Field(default=0.0, ge=0.0)


class BenchmarkCase(BaseModel):
    """Portable benchmark case independent of any agent or model SDK."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = BENCHMARK_CASE_SCHEMA_VERSION
    case_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    natural_language_task: str = Field(min_length=1)
    task: dict[str, Any]
    expected_mode: Literal["simulate", "optimize", "sweep", "sensitivity", "tolerance"]
    expected_capability: Literal["supported", "unsupported", "invalid"] = "supported"
    expected_failure_codes: list[str] = Field(default_factory=list)
    expected_artifacts: list[str] = Field(default_factory=list)
    physics_assertions: list[BenchmarkAssertion] = Field(default_factory=list)
    difficulty: Literal["basic", "intermediate", "advanced", "adversarial"] = "basic"
    tags: list[str] = Field(default_factory=list)
    execution: Literal["preflight_only", "run"] = "preflight_only"
    scenario: Literal["standard", "cache_replay", "sweep_resume"] = "standard"
    reproducibility_runs: int = Field(default=2, ge=2, le=3)

    @model_validator(mode="after")
    def _expectation_is_coherent(self) -> "BenchmarkCase":
        if self.expected_capability == "supported" and self.expected_failure_codes:
            raise ValueError("supported cases cannot declare expected failure codes")
        if self.expected_capability != "supported" and not self.expected_failure_codes:
            raise ValueError("rejected cases must declare at least one expected failure code")
        if self.expected_capability != "supported" and self.execution == "run":
            raise ValueError("rejected cases must stop at preflight")
        if self.scenario == "sweep_resume" and self.expected_mode != "sweep":
            raise ValueError("sweep_resume scenario requires expected_mode=sweep")
        if self.scenario != "standard" and self.execution != "run":
            raise ValueError("cache/resume scenarios require execution=run")
        return self


@dataclass
class _Metric:
    passed: int = 0
    total: int = 0

    def add(self, passed: bool) -> None:
        self.total += 1
        self.passed += int(bool(passed))

    def rate(self) -> float | None:
        return None if self.total == 0 else float(self.passed / self.total)

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "total": self.total, "rate": self.rate()}


@dataclass
class _BenchmarkMetrics:
    valid: _Metric = field(default_factory=_Metric)
    invalid: _Metric = field(default_factory=_Metric)
    failure_codes: _Metric = field(default_factory=_Metric)
    artifacts: _Metric = field(default_factory=_Metric)
    certificates: _Metric = field(default_factory=_Metric)
    reproducibility: _Metric = field(default_factory=_Metric)
    unsupported_false_accepts: int = 0
    unsupported_total: int = 0

    def to_dict(self) -> dict[str, Any]:
        false_rate = (
            None
            if self.unsupported_total == 0
            else float(self.unsupported_false_accepts / self.unsupported_total)
        )
        return {
            "valid_case_pass_rate": self.valid.to_dict(),
            "invalid_case_rejection_rate": self.invalid.to_dict(),
            "expected_failure_code_accuracy": self.failure_codes.to_dict(),
            "artifact_completeness_rate": self.artifacts.to_dict(),
            "certificate_success_rate": self.certificates.to_dict(),
            "reproducibility_rate": self.reproducibility.to_dict(),
            "unsupported_false_accept_count": self.unsupported_false_accepts,
            "unsupported_case_count": self.unsupported_total,
            "unsupported_false_accept_rate": false_rate,
        }


def load_benchmark_cases(root: str | Path) -> list[BenchmarkCase]:
    """Load and validate a directory tree of one-case-per-file JSON records."""

    base = Path(root)
    if not base.is_dir():
        raise FileNotFoundError(f"benchmark case directory does not exist: {base}")
    cases: list[BenchmarkCase] = []
    seen: set[str] = set()
    for path in sorted(base.rglob("*.json")):
        # Active challenge fixtures have their own CLI contract and are run by
        # challengebench; they are not final-answer AgentBench cases.
        if {"adaptive", "challenge", "fitting", "interactive"} & set(path.relative_to(base).parts):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        case = BenchmarkCase.model_validate(payload)
        if case.case_id in seen:
            raise ValueError(f"duplicate benchmark case_id: {case.case_id}")
        seen.add(case.case_id)
        cases.append(case)
    if not cases:
        raise ValueError(f"no benchmark cases found under {base}")
    return cases


def default_benchmark_cases_dir() -> Path:
    """Locate cases in a source checkout or an installed benchmark package."""

    source = Path(__file__).resolve().parents[1] / "benchmarks" / "cases"
    if source.is_dir():
        return source
    try:
        from importlib.resources import files

        resource = files("benchmarks").joinpath("cases")
        path = Path(str(resource))
        if path.is_dir():
            return path
    except (ImportError, ModuleNotFoundError, TypeError):
        pass
    raise FileNotFoundError(
        "packaged AgentBench cases are unavailable; pass --cases-dir explicitly"
    )


def _failure_codes(preflight: Mapping[str, Any]) -> set[str]:
    return {str(item.get("code")) for item in preflight.get("failures", [])}


def _artifact_kinds(envelope: Mapping[str, Any]) -> set[str]:
    return {str(item.get("kind")) for item in envelope.get("artifacts", [])}


def _resolve_path(payload: Any, path: str) -> tuple[bool, Any]:
    current = payload
    if not path:
        return True, current
    for token in path.split("."):
        if isinstance(current, Mapping) and token in current:
            current = current[token]
        elif isinstance(current, list) and token.isdigit() and int(token) < len(current):
            current = current[int(token)]
        else:
            return False, None
    return True, current


def _evaluate_assertion(
    assertion: BenchmarkAssertion, documents: Mapping[str, Any]
) -> dict[str, Any]:
    exists, actual = _resolve_path(documents.get(assertion.source), assertion.path)
    passed = False
    diagnostic: str | None = None
    try:
        if assertion.operator == "exists":
            passed = exists
        elif assertion.operator == "not_empty":
            passed = exists and actual not in (None, "", [], {})
        elif not exists:
            passed = False
        elif assertion.operator in {"eq", "ne"}:
            if isinstance(actual, (int, float)) and isinstance(assertion.expected, (int, float)):
                equal = abs(float(actual) - float(assertion.expected)) <= assertion.tolerance
            else:
                equal = actual == assertion.expected
            passed = equal if assertion.operator == "eq" else not equal
        elif assertion.operator == "between":
            passed = (
                assertion.minimum is not None
                and assertion.maximum is not None
                and float(assertion.minimum) - assertion.tolerance
                <= float(actual)
                <= float(assertion.maximum) + assertion.tolerance
            )
        else:
            expected = float(assertion.expected)
            value = float(actual)
            if assertion.operator == "ge":
                passed = value >= expected - assertion.tolerance
            elif assertion.operator == "gt":
                passed = value > expected - assertion.tolerance
            elif assertion.operator == "le":
                passed = value <= expected + assertion.tolerance
            elif assertion.operator == "lt":
                passed = value < expected + assertion.tolerance
    except (TypeError, ValueError) as exc:
        diagnostic = str(exc)
        passed = False
    return {
        "source": assertion.source,
        "path": assertion.path,
        "operator": assertion.operator,
        "expected": assertion.expected,
        "actual": actual if exists else None,
        "passed": bool(passed),
        "diagnostic": diagnostic,
    }


def _reproducibility_view(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Remove invocation identity while retaining numerical/scientific content."""

    volatile = {
        "run_id",
        "certificate_id",
        "wall_seconds",
        "wall_time_seconds",
        "artifact_root",
        "sha256",
        "size_bytes",
        "source_run_id",
    }

    def scrub(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): scrub(item) for key, item in value.items() if str(key) not in volatile
            }
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return copy.deepcopy(value)

    return scrub(payload)


def _documents(
    output: Path, preflight: Mapping[str, Any], envelope: Mapping[str, Any]
) -> dict[str, Any]:
    documents: dict[str, Any] = {
        "preflight": dict(preflight),
        "run_result": dict(envelope),
        "summary": envelope.get("summary") or {},
        "certificate": {},
    }
    artifact_documents = {
        "certificate": "PHYSICS_ACCEPTANCE_CERTIFICATE.json",
        "simulation_result": "SIMULATION_RESULT.json",
        "optimization_result": "OPTIMIZATION_RESULT.json",
        "sweep_result": "SWEEP_RESULT.json",
        "sensitivity_result": "SENSITIVITY_RESULT.json",
        "tolerance_result": "TOLERANCE_RESULT.json",
        "robustness_report": "ROBUSTNESS_REPORT.json",
    }
    for name, filename in artifact_documents.items():
        path = output / filename
        documents[name] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return documents


def _run_case_once(
    case: BenchmarkCase, root: Path, run_index: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    case_root = root / case.case_id / f"attempt_{run_index:02d}"
    case_root.mkdir(parents=True, exist_ok=True)
    task_path = case_root / "TASK.json"
    write_json(task_path, case.task)
    preflight = preflight_path(task_path)
    envelope: dict[str, Any] = {}
    if case.expected_capability == "supported" and case.execution == "run" and preflight.get("ok"):
        mode, task = load_task(task_path)
        settings = ExecutionSettings(write_plot=False)
        if case.scenario == "cache_replay":
            store = ExperimentStore(case_root / "store")
            execute_managed_task(
                mode,
                task,
                case_root / "cache_source",
                input_path=task_path,
                execution_settings=settings,
                store=store,
                cache=True,
            )
            envelope = execute_managed_task(
                mode,
                task,
                case_root / "run",
                input_path=task_path,
                execution_settings=settings,
                store=store,
                cache=True,
            )
        elif case.scenario == "sweep_resume":
            execute_managed_task(
                mode,
                task,
                case_root / "run",
                input_path=task_path,
                execution_settings=settings,
                store=None,
                cache=False,
                resume=False,
            )
            envelope = execute_managed_task(
                mode,
                task,
                case_root / "run",
                input_path=task_path,
                execution_settings=settings,
                store=None,
                cache=False,
                resume=True,
            )
        else:
            envelope = execute_managed_task(
                mode,
                task,
                case_root / "run",
                input_path=task_path,
                execution_settings=settings,
                store=None,
                cache=False,
            )
    return dict(preflight), envelope


def run_offline_benchmark(
    cases: Iterable[BenchmarkCase],
    *,
    output_path: str | Path | None = None,
    work_dir: str | Path | None = None,
    minimum_case_count: int = 80,
    detail: str = DEFAULT_RESPONSE_DETAIL,
) -> dict[str, Any]:
    """Run deterministic protocol/physics checks without using an LLM or network."""

    ordered = sorted(list(cases), key=lambda item: item.case_id)
    if not ordered:
        raise ValueError("at least one benchmark case is required")
    if minimum_case_count < 1:
        raise ValueError("minimum_case_count must be positive")
    started = time.perf_counter()
    metrics = _BenchmarkMetrics()
    records: list[dict[str, Any]] = []
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if work_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="veritmm_agentbench_")
        root = Path(temporary.name)
    else:
        root = Path(work_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
    try:
        for case in ordered:
            attempts: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for run_index in range(case.reproducibility_runs):
                attempts.append(_run_case_once(case, root, run_index))
            preflight, envelope = attempts[0]
            observed_codes = _failure_codes(preflight)
            expected_codes = set(case.expected_failure_codes)
            accepted = bool(preflight.get("ok"))
            mode_match = preflight.get("mode") == case.expected_mode
            unsupported = case.expected_capability == "unsupported"
            if unsupported:
                metrics.unsupported_total += 1
                metrics.unsupported_false_accepts += int(accepted)

            if case.expected_capability == "supported":
                execution_ok = case.execution != "run" or bool(envelope.get("ok"))
                failure_codes_ok = not observed_codes
                expectation_ok = accepted and execution_ok and mode_match
                metrics.valid.add(expectation_ok)
            else:
                failure_codes_ok = expected_codes == observed_codes
                expectation_ok = not accepted
                metrics.invalid.add(expectation_ok)
                metrics.failure_codes.add(failure_codes_ok)

            actual_artifacts = _artifact_kinds(envelope)
            artifact_ok = set(case.expected_artifacts) <= actual_artifacts
            if case.expected_artifacts:
                metrics.artifacts.add(artifact_ok)
            certificate_expected = "physics_certificate" in case.expected_artifacts
            certificate_ok = (
                bool((envelope.get("summary") or {}).get("physics", {}).get("accepted"))
                and envelope.get("certificate_id") is not None
            )
            if certificate_expected:
                metrics.certificates.add(certificate_ok)

            documents = _documents(root / case.case_id / "attempt_00" / "run", preflight, envelope)
            assertions = [
                _evaluate_assertion(assertion, documents) for assertion in case.physics_assertions
            ]
            assertions_ok = all(item["passed"] for item in assertions)

            fingerprints = []
            for attempt_preflight, attempt_envelope in attempts:
                value = attempt_envelope or attempt_preflight
                fingerprints.append(stable_payload_sha256(_reproducibility_view(value)))
            reproducible = len(set(fingerprints)) == 1
            metrics.reproducibility.add(reproducible)

            case_passed = bool(
                expectation_ok
                and failure_codes_ok
                and artifact_ok
                and assertions_ok
                and reproducible
                and (not certificate_expected or certificate_ok)
            )
            records.append(
                {
                    "case_id": case.case_id,
                    "category": case.category,
                    "difficulty": case.difficulty,
                    "expected_capability": case.expected_capability,
                    "expected_mode": case.expected_mode,
                    "observed_mode": preflight.get("mode"),
                    "mode_match": mode_match,
                    "preflight_accepted": accepted,
                    "expected_failure_codes": sorted(expected_codes),
                    "observed_failure_codes": sorted(observed_codes),
                    "execution_requested": case.execution == "run",
                    "scenario": case.scenario,
                    "execution_ok": None if not envelope else bool(envelope.get("ok")),
                    "expected_artifacts": sorted(case.expected_artifacts),
                    "observed_artifacts": sorted(actual_artifacts),
                    "assertions": assertions,
                    "reproducibility_fingerprints": fingerprints,
                    "reproducible": reproducible,
                    "passed": case_passed,
                }
            )
    finally:
        if temporary is not None:
            temporary.cleanup()

    metric_payload = metrics.to_dict()
    result = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "offline": True,
        "llm_calls": 0,
        "network_calls": 0,
        "case_count": len(ordered),
        "minimum_required_case_count": int(minimum_case_count),
        "case_count_requirement_passed": len(ordered) >= minimum_case_count,
        "case_contract_validation_rate": {
            "passed": len(ordered),
            "total": len(ordered),
            "rate": 1.0,
        },
        "passed_case_count": sum(int(item["passed"]) for item in records),
        "failed_case_count": sum(int(not item["passed"]) for item in records),
        "status": (
            "passed"
            if len(ordered) >= minimum_case_count and all(item["passed"] for item in records)
            else "failed"
        ),
        "release_gate_passed": bool(
            len(ordered) >= minimum_case_count
            and all(item["passed"] for item in records)
            and metric_payload["unsupported_false_accept_rate"] == 0.0
        ),
        "metrics": metric_payload,
        "cases": records,
        "wall_seconds": float(time.perf_counter() - started),
    }
    result["case_catalog_sha256"] = stable_payload_sha256(
        [item.model_dump(mode="json") for item in ordered]
    )
    result["content_hash_scope"] = "benchmark_result_without_wall_seconds_or_result_content_sha256"
    content_view = {
        key: value
        for key, value in result.items()
        if key not in {"wall_seconds", "result_content_sha256"}
    }
    result["result_content_sha256"] = stable_payload_sha256(content_view)
    if output_path is not None:
        # The benchmark result artifact remains the complete audit record.  A
        # caller receives the shared compact projection unless it asks for a
        # richer profile.
        target = Path(output_path).resolve()
        write_json(target, result)
        result["artifact_root"] = str(target.parent)
        result["artifacts"] = [
            {
                "kind": "benchmark_result",
                "path": target.name,
                "schema_version": BENCHMARK_SCHEMA_VERSION,
                "sha256": file_sha256(target),
                "size_bytes": int(target.stat().st_size),
            }
        ]
    return project_response(result, detail=detail)


__all__ = [
    "BENCHMARK_CASE_SCHEMA_VERSION",
    "BENCHMARK_SCHEMA_VERSION",
    "BenchmarkAssertion",
    "BenchmarkCase",
    "default_benchmark_cases_dir",
    "load_benchmark_cases",
    "run_offline_benchmark",
]
