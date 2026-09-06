"""Agent-safe MCP stdio surface for VeriTMM (1.1 phase).

This module is a **transport projection** of the existing protocol, not a
second execution path.  Every tool delegates to the same managed execution,
experiment store, and verification entry points the CLI uses; the server adds
only one thing — an explicit, agent-safe allowlist:

```text
exposed:   describe, schema, examples, preflight, run, history, inspect,
           lineage, compare, verify_run
rejected:  skip_certificate, physics_python, convergence_*, device,
           output_dir, store_dir, user_metadata_json, and every other
           operator/debug flag — they do not exist on this surface
```

`run` executes simulate-mode tasks through the existing managed path with the
default acceptance settings; the server owns the artifact root, the task root,
and the experiment store, so an agent can neither choose an output location
nor weaken a gate.  Tasks are accepted inline (`{"simulation": {...}}`), as a
`veritmm://task/...` resource reference, or as a path inside the server-owned
task root; every file reference is containment- and symlink-checked.  Large
artifacts stay out of tool responses: envelopes carry artifact references and
the full files are served as MCP resources
(`veritmm://run/{run_id}/{artifact}`).

The transport is stdio only in this release (Streamable HTTP, auth, and DSSE
signing follow in 1.2).  The pure tool layer below imports nothing from the
MCP SDK so the whole surface is testable without the optional dependency;
`create_server()` performs the guarded import.

Rejection semantics: parameters that are absent from a tool signature are
rejected at the **schema level** by the SDK (error format comes from the MCP
layer), while requests that reach a tool but violate this surface's rules
raise the typed `McpToolError`.  Clients must therefore expect either kind of
rejection envelope — the guarantee is that the parameter is unavailable, not
that every rejection carries the same shape.
"""

from __future__ import annotations

import argparse
import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .execution import ExecutionSettings
from .experiment_store import ExperimentStore, compare_runs
from .hashing import IDENTITY_SCHEME
from .managed_execution import execute_managed_task
from .preflight import preflight_path
from .protocol.capabilities import describe_capabilities
from .protocol.responses import DEFAULT_RESPONSE_DETAIL
from .protocol.schema_export import export_schema
from .schemas import (
    IlluminationSpec,
    LayerSpec,
    MediumSpec,
    SimulationTask,
    SpectralGrid,
    StackSpec,
)
from .task_io import simulation_task_from_dict
from .verify_run import verify_run_dir

SERVER_NAME = "veritmm"
SERVER_VERSION = "2.0.0"

VALID_RESPONSE_DETAILS = ("compact", "standard", "full")
VALID_SCHEMA_KINDS = ("simulation", "optimization", "sweep", "sensitivity", "tolerance")
RUN_ID_PATTERN = re.compile(r"^run_[0-9a-f]{8,64}$")

_TASK_RESOURCE_PREFIX = "veritmm://task/"
_RUN_RESOURCE_PREFIX = "veritmm://run/"


class McpToolError(ValueError):
    """Raised when an agent request violates the agent-safe allowlist.

    The message is safe to surface to the agent: it names the rejected
    parameter or path without leaking server-side details.
    """


def _reject(parameter: str, reason: str) -> "McpToolError":
    return McpToolError(f"rejected parameter {parameter!r}: {reason}")


def _contained(candidate: Path, root: Path) -> Path:
    """Resolve ``candidate`` and require it to stay inside ``root``.

    Symlinks are resolved as part of ``Path.resolve``, so a symlink that
    points outside the root fails the containment check; explicit symlinks on
    the path are rejected as well.
    """

    resolved_root = root.resolve()
    if candidate.is_symlink():
        raise _reject(candidate.name, "symbolic links are not accepted")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise _reject(
            str(candidate), "path escapes the server-owned root after resolution"
        )
    current = candidate
    while current != resolved_root:
        if current.is_symlink():
            raise _reject(str(candidate), "path contains a symbolic link")
        current = current.parent
    return resolved


class McpSurface:
    """Pure tool implementations behind the MCP projection.

    Holds the server-owned roots and exposes one method per allowed tool.
    No MCP SDK types appear here, so the full surface is testable without
    the optional dependency.
    """

    def __init__(
        self,
        *,
        artifact_root: str | Path,
        task_root: str | Path,
    ) -> None:
        self.artifact_root = Path(artifact_root).resolve()
        self.task_root = Path(task_root).resolve()
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.task_root.mkdir(parents=True, exist_ok=True)
        self._store: Optional[ExperimentStore] = None
        self._job_runtime_instance = None

    # ---- shared plumbing -------------------------------------------------

    @property
    def store(self) -> ExperimentStore:
        if self._store is None:
            self._store = ExperimentStore(self.artifact_root / "store")
        return self._store

    def _runs_root(self) -> Path:
        """The store's canonical per-run archive tree (keyed by run_id)."""

        runs_root = self.store.root / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        return runs_root

    def _invocations_root(self) -> Path:
        invocations = self.artifact_root / "invocations"
        invocations.mkdir(parents=True, exist_ok=True)
        return invocations

    def resolve_task(
        self, task: Union[str, Dict[str, Any]]
    ) -> Tuple[SimulationTask, Optional[Path]]:
        """Resolve one task specification into a validated task.

        Accepts an inline payload (``{"simulation": {...}}``), a
        ``veritmm://task/<path>`` reference, or a path relative to the
        server-owned task root.  Returns the task and, for file references,
        the resolved source path.
        """

        if isinstance(task, dict):
            payload = task.get("simulation")
            if payload is None or not isinstance(payload, dict):
                raise _reject(
                    "task",
                    "inline tasks must be {\"simulation\": {...}} objects",
                )
            return simulation_task_from_dict(payload), None
        if not isinstance(task, str) or not task.strip():
            raise _reject("task", "task must be an inline object or a reference")
        if task.startswith(_TASK_RESOURCE_PREFIX):
            relative = task[len(_TASK_RESOURCE_PREFIX) :]
        else:
            relative = task
        if Path(relative).is_absolute() or relative.startswith(".."):
            raise _reject("task", "task references must be relative to the task root")
        candidate = _contained(self.task_root / relative, self.task_root)
        if not candidate.is_file():
            raise _reject("task", f"task file not found in the task root: {relative}")
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise _reject("task", f"task file is unreadable: {exc}") from exc
        simulation = payload.get("simulation")
        if not isinstance(simulation, dict):
            raise _reject("task", "task files must contain a \"simulation\" object")
        return simulation_task_from_dict(simulation), candidate

    def _detail(self, detail: str) -> str:
        if detail not in VALID_RESPONSE_DETAILS:
            raise _reject("detail", f"must be one of {list(VALID_RESPONSE_DETAILS)}")
        return detail

    def _run_dir(self, run_id: str) -> Path:
        if not RUN_ID_PATTERN.match(run_id):
            raise _reject("run_id", "expected a run_<hex> identity")
        return _contained(self._runs_root() / run_id, self._runs_root())

    # ---- tools ------------------------------------------------------------

    def describe(self) -> Dict[str, Any]:
        manifest = describe_capabilities()
        payload = (
            manifest.model_dump(mode="json")
            if hasattr(manifest, "model_dump")
            else dict(manifest)
        )
        return {"identity_scheme": IDENTITY_SCHEME, **payload}

    def schema(self, kind: str) -> Dict[str, Any]:
        if kind not in VALID_SCHEMA_KINDS:
            raise _reject("kind", f"must be one of {list(VALID_SCHEMA_KINDS)}")
        return export_schema(kind)

    def examples(self) -> Dict[str, Any]:
        packaged = Path(__file__).resolve().parent / "examples"
        entries = []
        for path in sorted(packaged.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            entries.append(
                {
                    "name": path.stem,
                    "mode": (
                        "simulation"
                        if "simulation" in payload
                        else next(iter(payload), None)
                    ),
                    "description": payload.get("name") or path.stem,
                }
            )
        return {"ok": True, "examples": entries}

    def preflight(self, task: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        task_object, source = self.resolve_task(task)
        if source is None:
            staging = self.artifact_root / "staging"
            staging.mkdir(parents=True, exist_ok=True)
            source = staging / f"preflight_{uuid.uuid4().hex}.json"
            source.write_text(
                json.dumps(
                    {"mode": "simulate",
                     "simulation": task_object_to_payload(task_object)["simulation"]},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        payload = preflight_path(source)
        self._cleanup_staging(source)
        return payload

    def run(
        self,
        task: Union[str, Dict[str, Any]],
        *,
        detail: str = DEFAULT_RESPONSE_DETAIL,
        cache: bool = True,
        experiment_id: Optional[str] = None,
        parent_run_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
        hypothesis: Optional[str] = None,
        change_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        detail = self._detail(detail)
        task_object, _ = self.resolve_task(task)
        output = self._invocations_root() / f"invocation_{uuid.uuid4().hex}"
        output.mkdir(parents=True)
        envelope = execute_managed_task(
            "simulate",
            task_object,
            output,
            execution_settings=ExecutionSettings(),
            store=self.store,
            experiment_id=experiment_id,
            parent_run_id=parent_run_id,
            tags=tags or (),
            hypothesis=hypothesis,
            change_reason=change_reason,
            cache=cache,
            detail=detail,
        )
        return envelope

    def history(
        self,
        *,
        limit: int = 100,
        experiment_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 1000:
            raise _reject("limit", "must be an integer between 1 and 1000")
        runs = self.store.list_runs(experiment_id=experiment_id, limit=limit)
        return {
            "schema_version": "veritmm-history-v1",
            "ok": True,
            "store_root": str(self.store.root),
            "runs": [record.to_dict() for record in runs],
        }

    def inspect(self, run_id: str, *, detail: str = DEFAULT_RESPONSE_DETAIL) -> Dict[str, Any]:
        detail = self._detail(detail)
        return self.store.inspect(run_id, detail=detail)

    def lineage(self, run_id: str, *, detail: str = DEFAULT_RESPONSE_DETAIL) -> Dict[str, Any]:
        detail = self._detail(detail)
        return {"ok": True, **self.store.get_lineage(run_id, detail=detail)}

    def compare(self, run_a: str, run_b: str, *, detail: str = DEFAULT_RESPONSE_DETAIL) -> Dict[str, Any]:
        detail = self._detail(detail)
        return {"ok": True, **compare_runs(self.store, run_a, run_b)}

    def verify_run(self, run_id: str) -> Dict[str, Any]:
        run_dir = self._run_dir(run_id)
        if not (run_dir / "RUN_RESULT.json").is_file():
            raise _reject(
                "run_id", "no RUN_RESULT.json under the server-owned runs root"
            )
        return verify_run_dir(run_dir)

    def read_run_resource(self, run_id: str, artifact_path: str) -> str:
        run_dir = self._run_dir(run_id)
        if Path(artifact_path).is_absolute() or artifact_path.startswith(".."):
            raise _reject("path", "artifact references must be relative")
        candidate = _contained(run_dir / artifact_path, run_dir)
        if not candidate.is_file():
            raise _reject("path", "artifact not found in the run directory")
        return candidate.read_text(encoding="utf-8")

    def read_task_resource(self, relative_path: str) -> str:
        if Path(relative_path).is_absolute() or relative_path.startswith(".."):
            raise _reject("path", "task references must be relative to the task root")
        candidate = _contained(self.task_root / relative_path, self.task_root)
        if not candidate.is_file():
            raise _reject("path", "task file not found in the task root")
        return candidate.read_text(encoding="utf-8")

    def _cleanup_staging(self, source: Path) -> None:
        try:
            if source.parent == (self.artifact_root / "staging"):
                source.unlink(missing_ok=True)
        except OSError:
            pass

    # ---- V-11 namespaced tools -------------------------------------------

    def _job_runtime(self):
        if self._job_runtime_instance is None:
            from .job_runtime import JobRuntime

            self._job_runtime_instance = JobRuntime(store=self.store)
        return self._job_runtime_instance

    def discover(self, query: Optional[str] = None) -> Dict[str, Any]:
        text = (query or "").strip().lower()
        entries = []
        for tool in TOOL_CATALOG:
            haystack = " ".join(
                [tool["name"], tool["category"], tool["one_line"].lower()]
            )
            if text and text not in haystack:
                continue
            entries.append(
                {
                    "name": tool["name"],
                    "category": tool["category"],
                    "one_line": tool["one_line"],
                    "annotation": tool["annotation"],
                    "bound_in": tool["mode"],
                }
            )
        return {
            "ok": True,
            "count": len(entries),
            "tools": entries,
            "usage": (
                "compact servers bind only veritmm_discover and "
                "veritmm_catalog_get; request tools mode 'full' for the "
                "complete grid (legacy names stay bound there)."
            ),
        }

    def catalog_get(self, query: Optional[str] = None) -> Dict[str, Any]:
        from .capability_catalog import build_catalog

        catalog = build_catalog()
        text = (query or "").strip().lower()
        records = []
        for record in catalog["capabilities"]:
            haystack = " ".join(
                [record["capability_id"], record["provider"], *record["operations"]]
            ).lower()
            if text and text not in haystack:
                continue
            records.append(record)
        return {
            "ok": True,
            "count": len(records),
            "generated_at": catalog["generated_at"],
            "capabilities": records,
        }

    def material_search(self, query: str, limit: int = 10) -> Dict[str, Any]:
        from .material_registry import MaterialRegistry

        if not query or not query.strip():
            raise _reject("query", "material query must be non-empty")
        limit = int(limit)
        if not 1 <= limit <= 50:
            raise _reject("limit", "limit must be within 1..50")
        registry = MaterialRegistry()
        candidates = registry.search(query)[:limit]
        return {
            "ok": True,
            "query": query,
            "count": len(candidates),
            "candidates": [
                {
                    "provider": item.provider,
                    "dataset_id": str(item.dataset_id),
                    "book": item.book,
                    "page": item.page,
                    "has_n": item.has_n,
                    "has_k": item.has_k,
                    "points": item.points,
                    "range_nm": [item.range_min, item.range_max],
                }
                for item in candidates
            ],
        }

    def material_explain(self, name: str) -> Dict[str, Any]:
        return self.material_search(name, limit=5)

    def stack_validate(self, stack: Dict[str, Any]) -> Dict[str, Any]:
        task = SimulationTask(
            stack=StackSpec(
                layers=tuple(
                    LayerSpec(
                        item.get("material"),
                        float(item["thickness_nm"]),
                        constant_n=item.get("constant_n"),
                        constant_k=float(item.get("constant_k", 0.0)),
                        coherence=item.get("coherence", "coherent"),
                        label=item.get("label"),
                    )
                    for item in stack.get("layers", [])
                ),
                incident=_medium_from_payload(stack.get("incident", {})),
                exit=_medium_from_payload(stack.get("exit", {})),
            ),
            spectrum=SpectralGrid(500.0, 600.0, 11),
            illumination=IlluminationSpec((0.0,), ("unpolarized",)),
        )
        return {
            "ok": True,
            "layer_count": len(task.stack.layers),
            "coherent": not task.stack.has_incoherent_layers,
        }

    def simulate_spectrum(
        self,
        task: Union[str, Dict[str, Any]],
        detail: str = DEFAULT_RESPONSE_DETAIL,
        cache: bool = True,
    ) -> Dict[str, Any]:
        return self.run(task, detail=detail, cache=cache)

    def simulate_batch(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        from .job_runtime import JobRequest

        runtime = self._job_runtime()
        requests = []
        for task in tasks:
            output = self._invocations_root() / f"batch_{uuid.uuid4().hex}"
            requests.append(JobRequest("simulate", self.resolve_task(task)[0], output))
        report = runtime.run_batch(requests, workers=1)
        return {"ok": report["failed_count"] == 0, **report}

    def gradient_compute(
        self,
        task: Union[str, Dict[str, Any]],
        objective: str = "mean_R",
    ) -> Dict[str, Any]:
        from .gradient_api import compute_gradient

        task_object, _ = self.resolve_task(task)
        result = compute_gradient(task_object, objective)
        return {"ok": True, **result.to_dict()}

    def sensitivity_analyze(
        self,
        task: Union[str, Dict[str, Any]],
        objective: str = "mean_R",
        characteristic_scale_nm: float = 10.0,
    ) -> Dict[str, Any]:
        from .gradient_api import compute_sensitivity

        task_object, _ = self.resolve_task(task)
        result = compute_sensitivity(
            task_object,
            objective=objective,
            characteristic_scale_nm=float(characteristic_scale_nm),
        )
        return {"ok": True, **result.to_dict()}

    def optimize_problem(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        from .design_problem import (
            OptimizationProblemModel,
            compile_design_problem,
            problem_summary,
        )

        problem_model = OptimizationProblemModel.model_validate(problem)
        task = compile_design_problem(problem_model)
        output = self._invocations_root() / f"problem_{uuid.uuid4().hex}"
        envelope = execute_managed_task(
            "optimize",
            task,
            output,
            execution_settings=ExecutionSettings(),
            store=self.store,
        )
        return {
            "ok": envelope.get("status") == "completed",
            "problem": problem_summary(problem_model, task),
            "envelope": envelope,
        }

    def verify_explain(self, run_a: str, run_b: str) -> Dict[str, Any]:
        from .provenance_graph import build_graph

        graph = build_graph([self._run_dir(run_a), self._run_dir(run_b)])
        explanation = graph.explain_superiority(
            graph.run_id_for_source(self._run_dir(run_a)),
            graph.run_id_for_source(self._run_dir(run_b)),
        )
        return {"ok": True, "explanation": explanation}

    def job_status(self, job_id: str) -> Dict[str, Any]:
        return self._job_runtime().status(job_id)

    def job_result(self, job_id: str) -> Dict[str, Any]:
        return self._job_runtime().result(job_id)

    def intent_compile(
        self, spec: Dict[str, Any], stack: Dict[str, Any]
    ) -> Dict[str, Any]:
        from .intent import IntentSpec, compile_intent

        stack_task = self.resolve_task({"simulation": {"stack": stack}})[0]
        intent = IntentSpec.model_validate(spec)
        compiled = compile_intent(intent, stack_task.stack)
        payload: Dict[str, Any] = {
            "ok": compiled.certificate.semantic_status != "rejected",
            "task_kind": compiled.task_kind,
            "certificate": compiled.certificate.model_dump(mode="json"),
        }
        if compiled.task_kind == "optimization":
            payload["targets"] = [
                {
                    "name": target.name,
                    "observable": target.observable,
                    "constraint": target.constraint,
                    "target": target.target,
                }
                for target in compiled.optimization.targets
            ]
        elif compiled.simulation is not None:
            payload["spectrum_nm"] = [
                compiled.simulation.spectrum.wavelengths_nm()[0],
                compiled.simulation.spectrum.wavelengths_nm()[-1],
            ]
        return payload

    # ---- end V-11 namespaced tools ---------------------------------------



def _medium_from_payload(payload: dict) -> MediumSpec:
    return MediumSpec(
        material=payload.get("material"),
        constant_n=payload.get("constant_n"),
        constant_k=float(payload.get("constant_k", 0.0)),
    )



def task_object_to_payload(task: SimulationTask) -> Dict[str, Any]:
    """Serialize a task back into the public task-file payload shape."""

    from .schemas import dataclass_to_dict

    return {"simulation": dataclass_to_dict(task)}




# --------------------------------------------------------------- V-11 -----
# Namespaced tool grid (underscore style: MCP tool names must match
# [A-Za-z0-9_-]{1,64}, so dots are not wire-safe).  15 namespaced tools +
# 10 legacy aliases = 25 <= 30.

TOOL_ANNOTATIONS = {
    "discover": "READ_ONLY",
    "catalog_get": "READ_ONLY",
    "material_search": "READ_ONLY",
    "material_explain": "READ_ONLY",
    "stack_validate": "READ_ONLY",
    "simulate_spectrum": "COMPUTATIONAL",
    "simulate_batch": "EXPENSIVE_COMPUTE",
    "gradient_compute": "COMPUTATIONAL",
    "sensitivity_analyze": "COMPUTATIONAL",
    "optimize_problem": "EXPENSIVE_COMPUTE",
    "verify_run": "COMPUTATIONAL",
    "verify_explain": "COMPUTATIONAL",
    "job_status": "READ_ONLY",
    "job_result": "READ_ONLY",
    "intent_compile": "COMPUTATIONAL",
}

TOOL_CATALOG: List[Dict[str, Any]] = [
    {"name": "veritmm_discover", "category": "discovery",
     "one_line": "List the namespaced VeriTMM tools (name, category, one_line) matching an optional query.",
     "annotation": "READ_ONLY", "mode": "compact"},
    {"name": "veritmm_catalog_get", "category": "discovery",
     "one_line": "Return runtime-generated capability catalog records, optionally filtered.",
     "annotation": "READ_ONLY", "mode": "compact"},
    {"name": "veritmm_material_search", "category": "material",
     "one_line": "Search the bundled optical-constant catalog for a material name.",
     "annotation": "READ_ONLY", "mode": "full"},
    {"name": "veritmm_material_explain", "category": "material",
     "one_line": "Explain one material dataset: provider, dataset id, coverage, provenance.",
     "annotation": "READ_ONLY", "mode": "full"},
    {"name": "veritmm_stack_validate", "category": "stack",
     "one_line": "Validate a stack skeleton (layers, incident, exit) without running it.",
     "annotation": "READ_ONLY", "mode": "full"},
    {"name": "veritmm_simulate_spectrum", "category": "simulate",
     "one_line": "Execute one simulate task through managed execution (legacy: run).",
     "annotation": "COMPUTATIONAL", "mode": "full"},
    {"name": "veritmm_simulate_batch", "category": "simulate",
     "one_line": "Run several simulate tasks as isolated jobs with an aggregate report.",
     "annotation": "EXPENSIVE_COMPUTE", "mode": "full"},
    {"name": "veritmm_gradient_compute", "category": "gradient",
     "one_line": "Per-layer thickness gradients of a band objective (proposal basis).",
     "annotation": "COMPUTATIONAL", "mode": "full"},
    {"name": "veritmm_sensitivity_analyze", "category": "gradient",
     "one_line": "Normalized per-layer thickness sensitivities (objective change per scale).",
     "annotation": "COMPUTATIONAL", "mode": "full"},
    {"name": "veritmm_optimize_problem", "category": "optimize",
     "one_line": "Compile a declarative OptimizationProblem and run the optimize chain.",
     "annotation": "EXPENSIVE_COMPUTE", "mode": "full"},
    {"name": "veritmm_verify_run", "category": "verify",
     "one_line": "Independent four-state verification of a stored run (legacy: verify_run).",
     "annotation": "COMPUTATIONAL", "mode": "full"},
    {"name": "veritmm_verify_explain", "category": "verify",
     "one_line": "Explain why one run is superior to another via the provenance graph.",
     "annotation": "COMPUTATIONAL", "mode": "full"},
    {"name": "veritmm_job_status", "category": "job",
     "one_line": "Read the state machine status of one submitted job.",
     "annotation": "READ_ONLY", "mode": "full"},
    {"name": "veritmm_job_result", "category": "job",
     "one_line": "Read the envelope and outputs of one finished job.",
     "annotation": "READ_ONLY", "mode": "full"},
    {"name": "veritmm_intent_compile", "category": "intent",
     "one_line": "Compile a ScientificIntentSpec plus stack into a task and equivalence certificate.",
     "annotation": "COMPUTATIONAL", "mode": "full"},
]

assert len(TOOL_CATALOG) + 10 <= 30

def create_server(
    surface: Optional[McpSurface] = None,
    *,
    artifact_root: Optional[str | Path] = None,
    task_root: Optional[str | Path] = None,
    tools: str = "compact",
) -> Any:
    """Build the MCP stdio server (requires the optional ``mcp`` package).

    ``tools`` selects the exposure: "compact" (default) binds only the
    discovery pair; "full" binds the legacy grid plus the complete namespaced
    namespace (25 tools <= 30).
    """
    if tools not in ("compact", "full"):
        raise ValueError(f"unknown tools mode {tools!r}; expected compact or full")

    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError as exc:  # pragma: no cover - guarded optional import
        raise RuntimeError(
            "the MCP surface requires the optional 'mcp' extra: "
            "pip install 'veritmm[mcp]'"
        ) from exc

    surface = surface or McpSurface(
        artifact_root=artifact_root or Path.cwd() / "veritmm-mcp-artifacts",
        task_root=task_root or Path.cwd() / "veritmm-mcp-tasks",
    )
    server = MCPServer(
        name=SERVER_NAME,
        version=SERVER_VERSION,
        instructions=(
            "Agent-safe projection of the VeriTMM verifier-first TMM protocol. "
            "Discover capabilities with describe, inspect a task contract with "
            "schema, gate tasks with preflight, execute with run, and verify "
            "any run directory with verify_run. Large artifacts are served as "
            f"resources under veritmm://run/{{run_id}}/{{path}}; identity "
            f"scheme is {IDENTITY_SCHEME}."
        ),
    )

    @server.tool(
        name="describe",
        description="[READ_ONLY] Return the VeriTMM capability manifest and protocol identity.",
    )
    def describe_tool() -> Dict[str, Any]:
        return surface.describe()

    @server.tool(
        name="schema",
        description="[READ_ONLY] Export one public task JSON Schema (simulation/optimization/sweep/sensitivity/tolerance).",
    )
    def schema_tool(kind: str) -> Dict[str, Any]:
        return surface.schema(kind)

    @server.tool(name="examples", description="[READ_ONLY] List bundled example tasks.")
    def examples_tool() -> Dict[str, Any]:
        return surface.examples()

    @server.tool(
        name="preflight",
        description="[READ_ONLY] Validate one simulate task (contract, capability, material coverage) without running it.",
    )
    def preflight_tool(task: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        return surface.preflight(task)

    @server.tool(
        name="run",
        description=(
            "[EXPENSIVE_COMPUTE] Execute one simulate task through managed "
            "execution with default acceptance settings and return the compact "
            "run envelope with artifact references. Deprecated alias of "
            "veritmm_simulate_spectrum."
        ),
    )
    def run_tool(
        task: Union[str, Dict[str, Any]],
        detail: str = "compact",
        cache: bool = True,
        experiment_id: Optional[str] = None,
        parent_run_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
        hypothesis: Optional[str] = None,
        change_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        return surface.run(
            task,
            detail=detail,
            cache=cache,
            experiment_id=experiment_id,
            parent_run_id=parent_run_id,
            tags=tags,
            hypothesis=hypothesis,
            change_reason=change_reason,
        )

    @server.tool(
        name="history",
        description="[READ_ONLY] List persisted experiment runs in the server-owned store.",
    )
    def history_tool(limit: int = 100, experiment_id: Optional[str] = None) -> Dict[str, Any]:
        return surface.history(limit=limit, experiment_id=experiment_id)

    @server.tool(name="inspect", description="[READ_ONLY] Inspect one persisted run.")
    def inspect_tool(run_id: str, detail: str = "compact") -> Dict[str, Any]:
        return surface.inspect(run_id, detail=detail)

    @server.tool(name="lineage", description="[READ_ONLY] Show ancestors and children of one run.")
    def lineage_tool(run_id: str, detail: str = "compact") -> Dict[str, Any]:
        return surface.lineage(run_id, detail=detail)

    @server.tool(name="compare", description="[COMPUTATIONAL] Compare two persisted runs.")
    def compare_tool(run_a: str, run_b: str, detail: str = "compact") -> Dict[str, Any]:
        return surface.compare(run_a, run_b, detail=detail)

    @server.tool(
        name="verify_run",
        description=(
            "[COMPUTATIONAL] Independently verify a run directory: integrity, "
            "certification, replay, and authenticity. Read-only; no policy "
            "parameters. Deprecated alias of veritmm_verify_run."
        ),
    )
    def verify_run_tool(run_id: str) -> Dict[str, Any]:
        return surface.verify_run(run_id)

    @server.resource(
        "veritmm://task/{relative_path}",
        description="A task file from the server-owned task root.",
        mime_type="application/json",
    )
    def task_resource(relative_path: str) -> str:
        return surface.read_task_resource(relative_path)

    @server.resource(
        "veritmm://run/{run_id}/{artifact_path}",
        description="An artifact file from a server-executed run directory.",
        mime_type="application/json",
    )
    def run_resource(run_id: str, artifact_path: str) -> str:
        return surface.read_run_resource(run_id, artifact_path)

    # ---- V-11 namespaced tools -------------------------------------------

    @server.tool(
        name="veritmm_discover",
        description="[READ_ONLY] Compact discovery: list the namespaced VeriTMM tools.",
    )
    def veritmm_discover_tool(query: str = "") -> Dict[str, Any]:
        return surface.discover(query)

    @server.tool(
        name="veritmm_catalog_get",
        description="[READ_ONLY] Query the runtime-generated capability catalog.",
    )
    def veritmm_catalog_get_tool(query: str = "") -> Dict[str, Any]:
        return surface.catalog_get(query)

    @server.tool(
        name="veritmm_material_search",
        description="[READ_ONLY] Search the bundled optical-constant catalog for a material name.",
    )
    def veritmm_material_search_tool(query: str, limit: int = 10) -> Dict[str, Any]:
        return surface.material_search(query, limit=limit)

    @server.tool(
        name="veritmm_material_explain",
        description="[READ_ONLY] Explain one material dataset (provider, coverage, provenance).",
    )
    def veritmm_material_explain_tool(name: str) -> Dict[str, Any]:
        return surface.material_explain(name)

    @server.tool(
        name="veritmm_stack_validate",
        description="[READ_ONLY] Validate a stack skeleton without running it.",
    )
    def veritmm_stack_validate_tool(stack: Dict[str, Any]) -> Dict[str, Any]:
        return surface.stack_validate(stack)

    @server.tool(
        name="veritmm_simulate_spectrum",
        description="[COMPUTATIONAL] Execute one simulate task (managed execution, default acceptance settings).",
    )
    def veritmm_simulate_spectrum_tool(
        task: Union[str, Dict[str, Any]], detail: str = "compact"
    ) -> Dict[str, Any]:
        return surface.simulate_spectrum(task, detail=detail)

    @server.tool(
        name="veritmm_simulate_batch",
        description="[EXPENSIVE_COMPUTE] Run several simulate tasks as isolated jobs with an aggregate report.",
    )
    def veritmm_simulate_batch_tool(tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        return surface.simulate_batch(tasks)

    @server.tool(
        name="veritmm_gradient_compute",
        description="[COMPUTATIONAL] Per-layer thickness gradients of a band objective (proposal basis, not a certificate).",
    )
    def veritmm_gradient_compute_tool(
        task: Union[str, Dict[str, Any]], objective: str = "mean_R"
    ) -> Dict[str, Any]:
        return surface.gradient_compute(task, objective)

    @server.tool(
        name="veritmm_sensitivity_analyze",
        description="[COMPUTATIONAL] Normalized per-layer thickness sensitivities (proposal basis).",
    )
    def veritmm_sensitivity_analyze_tool(
        task: Union[str, Dict[str, Any]],
        objective: str = "mean_R",
        characteristic_scale_nm: float = 10.0,
    ) -> Dict[str, Any]:
        return surface.sensitivity_analyze(
            task, objective=objective, characteristic_scale_nm=characteristic_scale_nm
        )

    @server.tool(
        name="veritmm_optimize_problem",
        description="[EXPENSIVE_COMPUTE] Compile a declarative OptimizationProblem and run the optimize chain with certification.",
    )
    def veritmm_optimize_problem_tool(problem: Dict[str, Any]) -> Dict[str, Any]:
        return surface.optimize_problem(problem)

    @server.tool(
        name="veritmm_verify_run",
        description="[COMPUTATIONAL] Independent four-state verification of a stored run.",
    )
    def veritmm_verify_run_tool(run_id: str) -> Dict[str, Any]:
        return surface.verify_run(run_id)

    @server.tool(
        name="veritmm_verify_explain",
        description="[COMPUTATIONAL] Explain why one run is superior to another via the provenance graph.",
    )
    def veritmm_verify_explain_tool(run_a: str, run_b: str) -> Dict[str, Any]:
        return surface.verify_explain(run_a, run_b)

    @server.tool(
        name="veritmm_job_status",
        description="[READ_ONLY] Read the state machine status of one submitted job.",
    )
    def veritmm_job_status_tool(job_id: str) -> Dict[str, Any]:
        return surface.job_status(job_id)

    @server.tool(
        name="veritmm_job_result",
        description="[READ_ONLY] Read the envelope and outputs of one finished job.",
    )
    def veritmm_job_result_tool(job_id: str) -> Dict[str, Any]:
        return surface.job_result(job_id)

    @server.tool(
        name="veritmm_intent_compile",
        description="[COMPUTATIONAL] Compile a ScientificIntentSpec plus stack into a task and equivalence certificate.",
    )
    def veritmm_intent_compile_tool(spec: Dict[str, Any], stack: Dict[str, Any]) -> Dict[str, Any]:
        return surface.intent_compile(spec, stack)

    if tools == "compact":
        keep = {"veritmm_discover", "veritmm_catalog_get"}
        legacy = {
            "describe",
            "schema",
            "examples",
            "preflight",
            "run",
            "history",
            "inspect",
            "lineage",
            "compare",
            "verify_run",
        }
        namespaced_names = {tool["name"] for tool in TOOL_CATALOG}
        for name in sorted((legacy | namespaced_names) - keep):
            server.remove_tool(name)

    return server


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="veritmm-mcp",
        description="Run the VeriTMM agent-safe MCP server over stdio.",
    )
    parser.add_argument(
        "--artifact-root",
        default=None,
        help="Server-owned directory for run artifacts and the experiment store.",
    )
    parser.add_argument(
        "--task-root",
        default=None,
        help="Server-owned directory agents may reference tasks from.",
    )
    args = parser.parse_args(argv)
    surface = McpSurface(
        artifact_root=args.artifact_root or Path.cwd() / "veritmm-mcp-artifacts",
        task_root=args.task_root or Path.cwd() / "veritmm-mcp-tasks",
    )
    server = create_server(surface)
    server.run(transport="stdio")
    return 0


__all__ = [
    "McpSurface",
    "McpToolError",
    "SERVER_NAME",
    "SERVER_VERSION",
    "create_server",
    "main",
]

if __name__ == "__main__":  # pragma: no cover - manual/stdio entry
    raise SystemExit(main())
