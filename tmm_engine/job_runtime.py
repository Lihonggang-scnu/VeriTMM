"""SimulationJob runtime: job state machine, batch entry, fingerprint cache.

Turns "one call" into "one job": explicit state machine, a batch entry with
failure isolation, and a tool-fingerprint cache layer on top of the existing
content-addressed cache.  The fingerprint combines the capability declaration,
the source hashes of the implementing modules, and the parameter schema — any
code change changes the key, so a stale cache entry can never be presented as
current (ToolUniverse-style: unchanged retries may reuse, any substantive
change invalidates).

Asynchronous semantics are intentionally narrow: submission is synchronous, so
`cancel` honestly reports that it is not supported instead of pretending, and
`status` is only meaningfully visible once a job has reached a terminal state.
Real process parallelism belongs to V-12; `run_batch(workers=...)` accepts the
parameter and requires 1 until that lands.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import __version__
from .execution import ExecutionSettings
from .hashing import IDENTITY_SCHEME, stable_sha256
from .protocol.models import PROTOCOL_VERSION

JOB_SCHEMA_VERSION = "veritmm-simulation-job-v1"
JOB_BATCH_SCHEMA_VERSION = "veritmm-job-batch-result-v1"
FINGERPRINT_LEDGER_FILENAME = "veritmm-job-fingerprints.json"

JOB_STATES = ("queued", "running", "verifying", "completed", "failed", "cancelled")
JOB_TRANSITIONS: Dict[str, set] = {
    "queued": {"running", "cancelled"},
    "running": {"verifying", "completed", "failed", "cancelled"},
    "verifying": {"completed", "failed"},
    "completed": set(),
    "failed": set(),
    "cancelled": set(),
}

# Modules whose implementation defines the simulate capability's numerics and
# verdicts; their source hashes are part of the tool fingerprint.
_FINGERPRINT_MODULES = (
    "tmm_engine.workbench",
    "tmm_engine.tmm_solver",
    "tmm_engine.acceptance",
    "tmm_engine.convergence",
    "tmm_engine.material_registry",
    "tmm_engine.hashing",
)


class JobRuntimeError(ValueError):
    """Typed job-runtime failure with a machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _module_source_hashes() -> Dict[str, str]:
    import tmm_engine

    from .hashing import file_sha256

    hashes: Dict[str, str] = {}
    for module_name in _FINGERPRINT_MODULES:
        module = __import__(module_name, fromlist=["__file__"])
        hashes[module_name] = file_sha256(inspect.getfile(module))
    hashes["tmm_engine"] = file_sha256(Path(inspect.getfile(tmm_engine)))
    return hashes


def engine_fingerprint(capability: str = "simulate") -> str:
    """Tool fingerprint: capability declaration + implementation source hashes
    + parameter schema.  Any substantive change produces a new key."""

    from .protocol.schema_export import export_schema

    declaration = {
        "capability": capability,
        "identity_scheme": IDENTITY_SCHEME,
        "package_version": __version__,
        "protocol_version": PROTOCOL_VERSION,
    }
    return stable_sha256(
        {
            "declaration": declaration,
            "source_hashes": _module_source_hashes(),
            "parameter_schema_sha256": stable_sha256(export_schema("simulation")),
        }
    )


@dataclass
class SimulationJob:
    """One submitted unit of work with an explicit, closed state machine."""

    job_id: str
    config_hash: str
    capability_id: str
    status: str = "queued"
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    numerical_flags: Dict[str, Any] = field(default_factory=dict)
    runtime: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    progress: int = 0
    engine_fingerprint: str = ""

    def transition(self, target: str) -> None:
        if target not in JOB_STATES:
            raise JobRuntimeError("invalid_job_state", f"unknown job state {target!r}")
        allowed = JOB_TRANSITIONS[self.status]
        if target not in allowed:
            raise JobRuntimeError(
                "invalid_job_transition",
                f"transition {self.status!r} -> {target!r} is not allowed",
            )
        self.status = target

    def to_dict(self) -> dict:
        return {
            "schema_version": JOB_SCHEMA_VERSION,
            "job_id": self.job_id,
            "config_hash": self.config_hash,
            "capability_id": self.capability_id,
            "status": self.status,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "diagnostics": self.diagnostics,
            "numerical_flags": self.numerical_flags,
            "runtime": self.runtime,
            "provenance": self.provenance,
            "progress": self.progress,
            "engine_fingerprint": self.engine_fingerprint,
        }


@dataclass(frozen=True)
class JobRequest:
    """A submission request: one managed task invocation."""

    mode: str
    task: Any
    output_dir: Path
    device: str = "cpu"


class JobRuntime:
    """Synchronous job runtime with a fingerprint-guarded cache layer.

    The fingerprint ledger keys the existing execution identities; a ledger
    entry matches only when the engine fingerprint is unchanged.  First-time
    identities and fingerprint mismatches run with the content cache disabled
    (fail-closed), then record their fingerprint for future retries.
    """

    def __init__(self, store: Any = None) -> None:
        self.store = store
        self._jobs: Dict[str, SimulationJob] = {}
        self._results: Dict[str, dict] = {}
        self._counter = 0

    # -- ledger -----------------------------------------------------------

    @property
    def ledger_path(self) -> Optional[Path]:
        if self.store is None:
            return None
        return Path(self.store.root) / FINGERPRINT_LEDGER_FILENAME

    def _load_ledger(self) -> Dict[str, str]:
        path = self.ledger_path
        if path is None or not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _save_ledger(self, ledger: Dict[str, str]) -> None:
        path = self.ledger_path
        if path is None:
            return
        path.write_text(json.dumps(ledger, indent=1, sort_keys=True), encoding="utf-8")

    # -- submission -------------------------------------------------------

    def submit_job(self, request: JobRequest) -> dict:
        import time as time_module

        from .experiment_store import ExperimentStore
        from .hashing import stable_sha256 as _payload_hash
        from .managed_execution import (
            execute_managed_task,
            material_catalog_identity,
            normalized_operation,
        )

        if request.mode not in ("simulate", "optimize", "sweep", "sensitivity", "tolerance"):
            raise JobRuntimeError("unsupported_mode", f"unsupported job mode {request.mode!r}")
        self._counter += 1
        fingerprint = engine_fingerprint()
        job_id = f"job_{self._counter:06d}_{fingerprint[:8]}"

        settings = ExecutionSettings(device=request.device)
        normalized = normalized_operation(request.mode, request.task)
        identity = ExperimentStore.execution_identity(
            normalized,
            package_version=__version__,
            protocol_version=PROTOCOL_VERSION,
            material_catalog_sha256=material_catalog_identity(),
            execution_settings={"mode": request.mode, "device": request.device},
        )
        job = SimulationJob(
            job_id=job_id,
            config_hash=_payload_hash(normalized),
            capability_id=fingerprint,
            inputs={
                "mode": request.mode,
                "task_sha256": stable_sha256(normalized),
                "output_dir": str(Path(request.output_dir)),
            },
            diagnostics={"attempts": 0},
            runtime={"submitted_at": time_module.strftime("%Y-%m-%dT%H:%M:%S%z")},
            provenance={
                "engine_fingerprint": fingerprint,
                "parent_experiment": None,
                "identity_scheme": IDENTITY_SCHEME,
            },
            progress=5,
        )
        self._jobs[job_id] = job
        job.transition("running")
        job.runtime["started_at"] = job.runtime["submitted_at"]
        job.progress = 40

        ledger = self._load_ledger()
        cache_allowed = ledger.get(identity) == fingerprint
        try:
            envelope = execute_managed_task(
                request.mode,
                request.task,
                request.output_dir,
                execution_settings=settings,
                store=self.store,
                cache=cache_allowed,
            )
        except Exception as exc:
            job.transition("failed")
            job.runtime["finished_at"] = job.runtime["started_at"]
            job.progress = 100
            job.diagnostics["failure_code"] = "execution_failed"
            job.diagnostics["error"] = str(exc)[:500]
            self._results[job_id] = {
                "job": job.to_dict(),
                "envelope": None,
            }
            return job.to_dict()

        job.transition("verifying")
        job.progress = 90
        accepted = envelope.get("status") == "completed" and bool(
            envelope.get("certificate_id")
        )
        if accepted:
            job.transition("completed")
        else:
            job.transition("failed")
            job.diagnostics["failure_code"] = "not_certified"
        job.runtime["finished_at"] = job.runtime["started_at"]
        job.progress = 100
        job.outputs = {
            "run_id": envelope.get("run_id"),
            "certificate_id": envelope.get("certificate_id"),
            "status": envelope.get("status"),
            "cache_hit": envelope.get("cache_hit"),
        }
        ledger[identity] = fingerprint
        self._save_ledger(ledger)
        self._results[job_id] = {"job": job.to_dict(), "envelope": envelope}
        return job.to_dict()

    # -- introspection ----------------------------------------------------

    def status(self, job_id: str) -> dict:
        job = self._jobs.get(job_id)
        if job is None:
            raise JobRuntimeError("unknown_job", f"unknown job_id {job_id!r}")
        return job.to_dict()

    def cancel(self, job_id: str) -> dict:
        job = self._jobs.get(job_id)
        if job is None:
            raise JobRuntimeError("unknown_job", f"unknown job_id {job_id!r}")
        if job.status in ("queued", "running"):
            job.transition("cancelled")
            return {"job_id": job_id, "cancelled": True}
        return {
            "job_id": job_id,
            "cancelled": False,
            "supported": False,
            "reason": "synchronous execution; the job already reached a terminal state",
            "status": job.status,
        }

    def result(self, job_id: str) -> dict:
        if job_id not in self._jobs:
            raise JobRuntimeError("unknown_job", f"unknown job_id {job_id!r}")
        return self._results.get(
            job_id, {"job": self._jobs[job_id].to_dict(), "envelope": None}
        )

    # -- batch ------------------------------------------------------------

    def run_batch(self, requests: Sequence[JobRequest], workers: int = 1) -> dict:
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise JobRuntimeError("invalid_workers", "workers must be a positive integer")
        if workers > 1:
            raise JobRuntimeError(
                "workers_unsupported",
                "workers > 1 requires process parallel execution (V-12); "
                "the synchronous runtime supports workers=1 only",
            )
        entries: List[dict] = []
        completed = 0
        failed = 0
        for index, request in enumerate(requests):
            try:
                job_dict = self.submit_job(request)
            except Exception as exc:
                if isinstance(exc, JobRuntimeError):
                    code, message = exc.code, exc.message
                else:
                    code, message = "execution_failed", str(exc)[:500]
                job_dict = {
                    "job_id": f"batch_entry_{index}",
                    "status": "failed",
                    "diagnostics": {"failure_code": code, "error": message},
                    "outputs": {},
                }
            status = job_dict["status"]
            if status == "completed":
                completed += 1
            else:
                failed += 1
            entries.append(
                {
                    "index": index,
                    "job_id": job_dict.get("job_id"),
                    "status": status,
                    "physics_accepted": bool(
                        job_dict.get("outputs", {}).get("certificate_id")
                    )
                    and status == "completed",
                    "certificate_id": job_dict.get("outputs", {}).get("certificate_id"),
                    "failure_code": job_dict.get("diagnostics", {}).get("failure_code"),
                }
            )
        return {
            "schema_version": JOB_BATCH_SCHEMA_VERSION,
            "batch_size": len(entries),
            "completed_count": completed,
            "failed_count": failed,
            "workers": workers,
            "entries": entries,
        }


__all__ = [
    "FINGERPRINT_LEDGER_FILENAME",
    "JOB_BATCH_SCHEMA_VERSION",
    "JOB_SCHEMA_VERSION",
    "JOB_STATES",
    "JOB_TRANSITIONS",
    "JobRequest",
    "JobRuntime",
    "JobRuntimeError",
    "SimulationJob",
    "engine_fingerprint",
]
