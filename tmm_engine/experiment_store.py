"""Persistent experiment/run lineage for agent-driven VeriTMM studies.

The store deliberately keeps research metadata outside the numerical core.  A
row can describe *why* a caller ran a task, but none of these fields are read by
the capability gate or physics certificate.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .run_artifacts import stable_payload_sha256, write_json

EXPERIMENT_STORE_SCHEMA_VERSION = "veritmm-experiment-store-v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def default_store_root() -> Path:
    """Return the local-first store root for the current working directory."""

    return Path.cwd() / ".veritmm"


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    experiment_id: str
    parent_run_id: str | None
    task_sha256: str | None
    execution_identity_sha256: str | None
    operation: str
    status: str
    created_at: str
    completed_at: str | None
    protocol_version: str
    package_version: str
    certificate_id: str | None
    artifact_root: str
    cache_hit: bool
    source_run_id: str | None
    tags: tuple[str, ...]
    hypothesis: str | None
    change_reason: str | None
    user_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExperimentStore:
    """SQLite index plus filesystem artifacts for reproducible experiments."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else default_store_root()
        self.root = self.root.resolve()
        self.runs_root = self.root / "runs"
        self.db_path = self.root / "experiments.db"
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    parent_run_id TEXT,
                    task_sha256 TEXT,
                    execution_identity_sha256 TEXT,
                    operation TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    protocol_version TEXT NOT NULL,
                    package_version TEXT NOT NULL,
                    certificate_id TEXT,
                    artifact_root TEXT NOT NULL,
                    cache_hit INTEGER NOT NULL DEFAULT 0,
                    source_run_id TEXT,
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    hypothesis TEXT,
                    change_reason TEXT,
                    user_metadata_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(parent_run_id) REFERENCES runs(run_id),
                    FOREIGN KEY(source_run_id) REFERENCES runs(run_id)
                );
                CREATE INDEX IF NOT EXISTS idx_runs_experiment
                    ON runs(experiment_id, created_at, run_id);
                CREATE INDEX IF NOT EXISTS idx_runs_parent
                    ON runs(parent_run_id, created_at, run_id);
                CREATE INDEX IF NOT EXISTS idx_runs_execution_identity
                    ON runs(execution_identity_sha256, status, completed_at);
                """
            )
            conn.execute(
                "INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version',?)",
                (EXPERIMENT_STORE_SCHEMA_VERSION,),
            )

    @staticmethod
    def new_run_id() -> str:
        return f"run_{uuid.uuid4().hex}"

    @staticmethod
    def new_experiment_id() -> str:
        return f"exp_{uuid.uuid4().hex}"

    def artifact_dir(self, run_id: str) -> Path:
        if not run_id.startswith("run_") or any(part in run_id for part in ("/", "\\", "..")):
            raise ValueError("invalid run_id")
        return self.runs_root / run_id

    def archive_artifacts(self, source: str | Path, run_id: str) -> Path:
        """Copy one completed invocation into the store's canonical run tree."""

        source_root = Path(source).resolve()
        target = self.artifact_dir(run_id).resolve()
        if source_root == target:
            return target
        target.mkdir(parents=True, exist_ok=True)
        for path in sorted(source_root.iterdir()):
            destination = target / path.name
            if path.is_dir():
                shutil.copytree(path, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(path, destination)
        return target

    def record_run(
        self,
        *,
        run_id: str,
        experiment_id: str,
        task_sha256: str | None,
        execution_identity_sha256: str | None,
        operation: str,
        status: str,
        protocol_version: str,
        package_version: str,
        artifact_root: str | Path,
        parent_run_id: str | None = None,
        certificate_id: str | None = None,
        cache_hit: bool = False,
        source_run_id: str | None = None,
        tags: Iterable[str] = (),
        hypothesis: str | None = None,
        change_reason: str | None = None,
        user_metadata: Mapping[str, Any] | None = None,
        created_at: str | None = None,
        completed_at: str | None = None,
    ) -> RunRecord:
        """Insert or replace one auditable run record.

        Research metadata is stored verbatim but never inspected by the solver.
        """

        root = Path(artifact_root).resolve()
        normalized_tags = tuple(sorted({str(tag) for tag in tags if str(tag)}))
        metadata = dict(user_metadata or {})
        created = created_at or _utc_now()
        completed = completed_at or _utc_now()
        with self._connect() as conn:
            if parent_run_id is not None and conn.execute(
                "SELECT 1 FROM runs WHERE run_id=?", (parent_run_id,)
            ).fetchone() is None:
                raise KeyError(f"unknown parent_run_id: {parent_run_id}")
            if source_run_id is not None and conn.execute(
                "SELECT 1 FROM runs WHERE run_id=?", (source_run_id,)
            ).fetchone() is None:
                raise KeyError(f"unknown source_run_id: {source_run_id}")
            conn.execute(
                """
                INSERT OR REPLACE INTO runs(
                    run_id, experiment_id, parent_run_id, task_sha256,
                    execution_identity_sha256, operation, status, created_at,
                    completed_at, protocol_version, package_version, certificate_id,
                    artifact_root, cache_hit, source_run_id, tags_json, hypothesis,
                    change_reason, user_metadata_json
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    run_id,
                    experiment_id,
                    parent_run_id,
                    task_sha256,
                    execution_identity_sha256,
                    operation,
                    status,
                    created,
                    completed,
                    protocol_version,
                    package_version,
                    certificate_id,
                    str(root),
                    int(bool(cache_hit)),
                    source_run_id,
                    _json(normalized_tags),
                    hypothesis,
                    change_reason,
                    _json(metadata),
                ),
            )
        record = self.get_run(run_id)
        if record is None:  # pragma: no cover - SQLite contract guard
            raise RuntimeError("run record disappeared after insertion")
        return record

    def record_envelope(
        self,
        envelope: Mapping[str, Any],
        *,
        artifact_root: str | Path,
        experiment_id: str | None = None,
        parent_run_id: str | None = None,
        execution_identity_sha256: str | None = None,
        tags: Iterable[str] = (),
        hypothesis: str | None = None,
        change_reason: str | None = None,
        user_metadata: Mapping[str, Any] | None = None,
    ) -> RunRecord:
        from . import __version__

        run_id = str(envelope["run_id"])
        return self.record_run(
            run_id=run_id,
            experiment_id=experiment_id or self.new_experiment_id(),
            parent_run_id=parent_run_id,
            task_sha256=envelope.get("task_sha256"),
            execution_identity_sha256=execution_identity_sha256,
            operation=str(envelope.get("operation", "unknown")),
            status=str(envelope.get("status", "unknown")),
            protocol_version=str(envelope.get("protocol_version", "unknown")),
            package_version=__version__,
            certificate_id=envelope.get("certificate_id"),
            artifact_root=artifact_root,
            cache_hit=bool(envelope.get("cache_hit", False)),
            source_run_id=envelope.get("source_run_id"),
            tags=tags,
            hypothesis=hypothesis,
            change_reason=change_reason,
            user_metadata=user_metadata,
        )

    @staticmethod
    def _from_row(row: sqlite3.Row) -> RunRecord:
        return RunRecord(
            run_id=row["run_id"],
            experiment_id=row["experiment_id"],
            parent_run_id=row["parent_run_id"],
            task_sha256=row["task_sha256"],
            execution_identity_sha256=row["execution_identity_sha256"],
            operation=row["operation"],
            status=row["status"],
            created_at=row["created_at"],
            completed_at=row["completed_at"],
            protocol_version=row["protocol_version"],
            package_version=row["package_version"],
            certificate_id=row["certificate_id"],
            artifact_root=row["artifact_root"],
            cache_hit=bool(row["cache_hit"]),
            source_run_id=row["source_run_id"],
            tags=tuple(json.loads(row["tags_json"])),
            hypothesis=row["hypothesis"],
            change_reason=row["change_reason"],
            user_metadata=dict(json.loads(row["user_metadata_json"])),
        )

    def get_run(self, run_id: str) -> RunRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return None if row is None else self._from_row(row)

    def list_runs(
        self, *, experiment_id: str | None = None, limit: int = 100
    ) -> list[RunRecord]:
        count = max(1, min(int(limit), 10_000))
        query = "SELECT * FROM runs"
        params: tuple[Any, ...] = ()
        if experiment_id is not None:
            query += " WHERE experiment_id=?"
            params = (experiment_id,)
        query += " ORDER BY created_at DESC, run_id DESC LIMIT ?"
        params += (count,)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._from_row(row) for row in rows]

    def list_children(self, run_id: str) -> list[RunRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs WHERE parent_run_id=? ORDER BY created_at,run_id",
                (run_id,),
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def get_lineage(self, run_id: str) -> dict[str, Any]:
        current = self.get_run(run_id)
        if current is None:
            raise KeyError(f"unknown run_id: {run_id}")
        ancestors: list[RunRecord] = []
        seen = {run_id}
        parent_id = current.parent_run_id
        while parent_id is not None:
            if parent_id in seen:
                raise RuntimeError("cycle detected in experiment lineage")
            seen.add(parent_id)
            parent = self.get_run(parent_id)
            if parent is None:
                break
            ancestors.append(parent)
            parent_id = parent.parent_run_id
        ancestors.reverse()
        return {
            "schema_version": "veritmm-lineage-v1",
            "run": current.to_dict(),
            "ancestors": [record.to_dict() for record in ancestors],
            "children": [record.to_dict() for record in self.list_children(run_id)],
        }

    def find_cache_source(self, execution_identity_sha256: str) -> RunRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM runs
                WHERE execution_identity_sha256=? AND status IN
                    ('completed','physically_valid','physically_valid_with_limits')
                ORDER BY cache_hit ASC, completed_at DESC, run_id DESC LIMIT 1
                """,
                (execution_identity_sha256,),
            ).fetchone()
        if row is None:
            return None
        record = self._from_row(row)
        return record if Path(record.artifact_root, "RUN_RESULT.json").is_file() else None

    @staticmethod
    def execution_identity(
        normalized_task: Mapping[str, Any],
        *,
        package_version: str,
        protocol_version: str,
        material_catalog_sha256: str | None,
        execution_settings: Mapping[str, Any],
    ) -> str:
        return stable_payload_sha256(
            {
                "normalized_task": normalized_task,
                "package_version": package_version,
                "protocol_version": protocol_version,
                "material_catalog_sha256": material_catalog_sha256,
                "execution_settings": dict(execution_settings),
            }
        )

    def materialize_cache_hit(
        self,
        source: RunRecord,
        destination: str | Path,
        *,
        new_run_id: str,
    ) -> dict[str, Any]:
        """Copy immutable artifacts while preserving source provenance."""

        source_root = Path(source.artifact_root)
        target = Path(destination).resolve()
        if source_root.resolve() == target:
            raise ValueError("cache destination must differ from source artifact root")
        target.mkdir(parents=True, exist_ok=True)
        for path in sorted(source_root.iterdir()):
            if path.name == "RUN_RESULT.json":
                continue
            destination_path = target / path.name
            if path.is_dir():
                shutil.copytree(path, destination_path, dirs_exist_ok=True)
            else:
                shutil.copy2(path, destination_path)
        result = json.loads((source_root / "RUN_RESULT.json").read_text(encoding="utf-8"))
        result["run_id"] = new_run_id
        result["cache_hit"] = True
        result["source_run_id"] = source.run_id
        result["artifact_provenance"] = {
            "mode": "cache_copy",
            "source_run_id": source.run_id,
        }
        summary_path = target / "RESULT_SUMMARY.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["run_id"] = new_run_id
            summary["cache_hit"] = True
            summary["source_run_id"] = source.run_id
            write_json(summary_path, summary)
        manifest_path = target / "RUN_MANIFEST.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["run_id"] = new_run_id
            manifest["cache_hit"] = True
            manifest["source_run_id"] = source.run_id
            write_json(manifest_path, manifest)
        # Refresh hashes because RESULT_SUMMARY changed.
        from .run_artifacts import index_artifacts

        result["artifacts"] = index_artifacts(target)
        write_json(target / "RUN_RESULT.json", result)
        return result

    def inspect(self, run_id: str) -> dict[str, Any]:
        record = self.get_run(run_id)
        if record is None:
            raise KeyError(f"unknown run_id: {run_id}")
        root = Path(record.artifact_root)
        result_path = root / "RUN_RESULT.json"
        result = (
            json.loads(result_path.read_text(encoding="utf-8"))
            if result_path.is_file()
            else None
        )
        return {
            "schema_version": "veritmm-inspect-v1",
            "record": record.to_dict(),
            "run_result": result,
            "artifact_root_exists": root.is_dir(),
        }


def _stable_diff(left: Any, right: Any, path: str = "") -> list[dict[str, Any]]:
    """Return deterministic RFC-6901-style value deltas."""

    if isinstance(left, Mapping) and isinstance(right, Mapping):
        deltas: list[dict[str, Any]] = []
        for key in sorted(set(left) | set(right), key=str):
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            child = f"{path}/{escaped}"
            if key not in left:
                deltas.append({"path": child, "from": None, "to": right[key]})
            elif key not in right:
                deltas.append({"path": child, "from": left[key], "to": None})
            else:
                deltas.extend(_stable_diff(left[key], right[key], child))
        return deltas
    if isinstance(left, list) and isinstance(right, list):
        deltas = []
        for index in range(max(len(left), len(right))):
            child = f"{path}/{index}"
            if index >= len(left):
                deltas.append({"path": child, "from": None, "to": right[index]})
            elif index >= len(right):
                deltas.append({"path": child, "from": left[index], "to": None})
            else:
                deltas.extend(_stable_diff(left[index], right[index], child))
        return deltas
    if left != right:
        return [{"path": path or "/", "from": left, "to": right}]
    return []


def _load_json_if_exists(root: Path, name: str) -> dict[str, Any]:
    path = root / name
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {"value": payload}


def _declared_solver(task: Mapping[str, Any]) -> Any:
    if isinstance(task.get("simulation"), Mapping):
        return task["simulation"].get("solver")
    optimization = task.get("optimization")
    if isinstance(optimization, Mapping) and isinstance(optimization.get("simulation"), Mapping):
        return optimization["simulation"].get("solver")
    sweep = task.get("sweep")
    if isinstance(sweep, Mapping) and isinstance(sweep.get("base_simulation"), Mapping):
        return sweep["base_simulation"].get("solver")
    return None


def compare_runs(store: ExperimentStore, run_a: str, run_b: str) -> dict[str, Any]:
    """Compare two runs without making a scientific preference judgment."""

    a = store.get_run(run_a)
    b = store.get_run(run_b)
    if a is None or b is None:
        missing = run_a if a is None else run_b
        raise KeyError(f"unknown run_id: {missing}")
    root_a, root_b = Path(a.artifact_root), Path(b.artifact_root)
    task_a = _load_json_if_exists(root_a, "NORMALIZED_TASK.json")
    task_b = _load_json_if_exists(root_b, "NORMALIZED_TASK.json")
    summary_a = _load_json_if_exists(root_a, "RESULT_SUMMARY.json")
    summary_b = _load_json_if_exists(root_b, "RESULT_SUMMARY.json")
    cert_a = _load_json_if_exists(root_a, "PHYSICS_ACCEPTANCE_CERTIFICATE.json")
    cert_b = _load_json_if_exists(root_b, "PHYSICS_ACCEPTANCE_CERTIFICATE.json")

    material_paths = ("/simulation/stack", "/optimization/simulation/stack")
    all_task_diff = _stable_diff(task_a, task_b)
    material_diff = [
        delta for delta in all_task_diff if any(token in delta["path"] for token in material_paths)
        and any(name in delta["path"] for name in ("material", "provider", "dataset_id", "constant_n", "constant_k"))
    ]
    solver_diff = {
        "from": _declared_solver(task_a),
        "to": _declared_solver(task_b),
    }
    artifacts_a = {
        item["path"]: item["sha256"]
        for item in (_load_json_if_exists(root_a, "RUN_RESULT.json").get("artifacts") or [])
    }
    artifacts_b = {
        item["path"]: item["sha256"]
        for item in (_load_json_if_exists(root_b, "RUN_RESULT.json").get("artifacts") or [])
    }
    return {
        "schema_version": "veritmm-run-compare-v1",
        "run_a": run_a,
        "run_b": run_b,
        "task_diff": all_task_diff,
        "material_diff": material_diff,
        "solver_diff": solver_diff,
        "summary_diff": _stable_diff(summary_a, summary_b),
        "certificate_diff": _stable_diff(cert_a, cert_b),
        "artifact_diff": _stable_diff(artifacts_a, artifacts_b),
    }


__all__ = [
    "EXPERIMENT_STORE_SCHEMA_VERSION",
    "ExperimentStore",
    "RunRecord",
    "compare_runs",
    "default_store_root",
]
