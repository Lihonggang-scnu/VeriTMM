"""Persistent experiment/run lineage for agent-driven VeriTMM studies.

The store deliberately keeps research metadata outside the numerical core.  A
row can describe *why* a caller ran a task, but none of these fields are read by
the capability gate or physics certificate.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

from ._version import __version__
from .archive.schema_registry import ARCHIVE_SCHEMA_VERSION
from .protocol.responses import (
    CANONICAL_MAX_ARTIFACT_REFS,
    LINEAGE_MAX_RECORDS,
    RESPONSE_CONTEXT_FILENAME,
    normalize_response_detail,
    project_response,
    rebase_response_context,
    response_profile,
)
from .run_artifacts import stable_payload_sha256, write_json

EXPERIMENT_STORE_SCHEMA_VERSION = "veritmm-experiment-store-v2"
_RUN_SCOPED_CACHE_ARTIFACTS: tuple[str, ...] = (
    "RESULT_SUMMARY.json",
    "RUN_MANIFEST.json",
    "SWEEP_RESULT.json",
    "SENSITIVITY_RESULT.json",
    "TOLERANCE_RESULT.json",
    "ROBUSTNESS_REPORT.json",
)


class RunLedgerConflictError(RuntimeError):
    """Raised when an operation would overwrite immutable run provenance."""

    def __init__(self, run_id: str, message: str) -> None:
        super().__init__(message)
        self.run_id = run_id


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
    archive_schema_version: int = ARCHIVE_SCHEMA_VERSION

    @property
    def version_identity_status(self) -> Literal["verified", "legacy_inconsistent"]:
        """Report whether the stored package identity matches this runtime."""

        return (
            "verified"
            if self.package_version == __version__
            else "legacy_inconsistent"
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["version_identity_status"] = self.version_identity_status
        return payload


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
                    archive_schema_version INTEGER NOT NULL DEFAULT 1,
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
            columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(runs)").fetchall()
            }
            if "archive_schema_version" not in columns:
                conn.execute(
                    "ALTER TABLE runs ADD COLUMN archive_schema_version INTEGER NOT NULL DEFAULT 1"
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

    @staticmethod
    def assert_no_path_redirection(path: str | Path) -> None:
        """Reject symlink/junction components before resolving an output path."""

        absolute = Path(path).absolute()
        for candidate in (absolute, *absolute.parents):
            if not candidate.exists() and not candidate.is_symlink():
                continue
            candidate_stat = candidate.lstat()
            if candidate.is_symlink() or getattr(candidate_stat, "st_reparse_tag", 0):
                raise ValueError(
                    f"artifact paths must not traverse links or reparse points: {candidate}"
                )

    def archive_artifacts(self, source: str | Path, run_id: str) -> Path:
        """Copy one completed invocation into the store's canonical run tree."""

        source_path = Path(source)
        if source_path.is_symlink():
            raise ValueError("artifact source root must not be a symbolic link")
        source_root = source_path.resolve()
        target_path = self.artifact_dir(run_id)
        if target_path.is_symlink():
            raise ValueError("canonical artifact directory must not be a symbolic link")
        target = target_path.resolve()
        if target.parent != self.runs_root.resolve():
            raise ValueError("canonical artifact directory escapes the runs root")
        if source_root == target:
            return target
        if target.exists() and (
            not target.is_dir() or any(target.iterdir())
        ):
            raise RunLedgerConflictError(
                run_id,
                f"canonical artifact directory is already non-empty for run_id: {run_id}",
            )
        self.assert_artifact_tree_no_links(source_root)
        staging = self.runs_root / f".stage-{uuid.uuid4().hex[:8]}"
        staging.mkdir()
        try:
            for path in sorted(source_root.iterdir()):
                destination = staging / path.name
                if path.is_dir():
                    shutil.copytree(path, destination, symlinks=True)
                else:
                    shutil.copy2(path, destination)
            self.assert_artifact_tree_no_links(staging)
            if target.exists():
                target.rmdir()
            staging.replace(target)
        except Exception:
            if staging.is_dir():
                shutil.rmtree(staging)
            raise
        return target

    def discard_unindexed_artifacts(self, run_ids: Iterable[str]) -> None:
        """Rollback staged canonical directories that have no committed ledger row."""

        for raw_run_id in run_ids:
            run_id = str(raw_run_id)
            if self.get_run(run_id) is not None:
                continue
            target = self.artifact_dir(run_id).resolve()
            if target.parent != self.runs_root.resolve():  # pragma: no cover - guard
                raise ValueError("refusing to clean artifacts outside canonical runs root")
            if target.is_dir():
                shutil.rmtree(target)

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
        archive_schema_version: int = ARCHIVE_SCHEMA_VERSION,
        created_at: str | None = None,
        completed_at: str | None = None,
    ) -> RunRecord:
        """Append one immutable run identity to the ledger.

        Research metadata is stored verbatim but never inspected by the solver.
        An existing ``run_id`` is a provenance conflict and is never replaced.
        """
        return self.record_run_batch(
            (
                {
                    "run_id": run_id,
                    "experiment_id": experiment_id,
                    "task_sha256": task_sha256,
                    "execution_identity_sha256": execution_identity_sha256,
                    "operation": operation,
                    "status": status,
                    "protocol_version": protocol_version,
                    "package_version": package_version,
                    "archive_schema_version": archive_schema_version,
                    "artifact_root": artifact_root,
                    "parent_run_id": parent_run_id,
                    "certificate_id": certificate_id,
                    "cache_hit": cache_hit,
                    "source_run_id": source_run_id,
                    "tags": tuple(tags),
                    "hypothesis": hypothesis,
                    "change_reason": change_reason,
                    "user_metadata": dict(user_metadata or {}),
                    "created_at": created_at,
                    "completed_at": completed_at,
                },
            )
        )[0]

    def record_run_batch(
        self,
        records: Iterable[Mapping[str, Any]],
    ) -> list[RunRecord]:
        """Append a parent/child run group in one SQLite transaction."""

        prepared: list[dict[str, Any]] = []
        for record in records:
            run_id = str(record["run_id"])
            tags = record.get("tags") or ()
            prepared.append(
                {
                    **dict(record),
                    "run_id": run_id,
                    "artifact_root": str(Path(record["artifact_root"]).resolve()),
                    "cache_hit": int(bool(record.get("cache_hit", False))),
                    "archive_schema_version": int(
                        record.get("archive_schema_version", ARCHIVE_SCHEMA_VERSION)
                    ),
                    "tags_json": _json(
                        tuple(sorted({str(tag) for tag in tags if str(tag)}))
                    ),
                    "user_metadata_json": _json(
                        dict(record.get("user_metadata") or {})
                    ),
                    "created_at": record.get("created_at") or _utc_now(),
                    "completed_at": record.get("completed_at") or _utc_now(),
                }
            )
        if not prepared:
            return []
        ids = [record["run_id"] for record in prepared]
        if len(ids) != len(set(ids)):
            raise RunLedgerConflictError(
                next(run_id for run_id in ids if ids.count(run_id) > 1),
                "duplicate run_id within append-only batch",
            )

        with self._connect() as conn:
            for record in prepared:
                run_id = record["run_id"]
                if conn.execute(
                    "SELECT 1 FROM runs WHERE run_id=?", (run_id,)
                ).fetchone() is not None:
                    raise RunLedgerConflictError(
                        run_id,
                        f"run_id already exists in append-only ledger: {run_id}",
                    )
                parent_run_id = record.get("parent_run_id")
                if parent_run_id is not None and conn.execute(
                    "SELECT 1 FROM runs WHERE run_id=?", (parent_run_id,)
                ).fetchone() is None:
                    raise KeyError(f"unknown parent_run_id: {parent_run_id}")
                source_run_id = record.get("source_run_id")
                if source_run_id is not None and conn.execute(
                    "SELECT 1 FROM runs WHERE run_id=?", (source_run_id,)
                ).fetchone() is None:
                    raise KeyError(f"unknown source_run_id: {source_run_id}")
                try:
                    conn.execute(
                        """
                        INSERT INTO runs(
                            run_id, experiment_id, parent_run_id, task_sha256,
                            execution_identity_sha256, operation, status, created_at,
                            completed_at, protocol_version, package_version,
                            archive_schema_version, certificate_id, artifact_root, cache_hit,
                            source_run_id,
                            tags_json, hypothesis, change_reason, user_metadata_json
                        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            run_id,
                            record["experiment_id"],
                            parent_run_id,
                            record.get("task_sha256"),
                            record.get("execution_identity_sha256"),
                            record["operation"],
                            record["status"],
                            record["created_at"],
                            record["completed_at"],
                            record["protocol_version"],
                            record["package_version"],
                            record["archive_schema_version"],
                            record.get("certificate_id"),
                            record["artifact_root"],
                            record["cache_hit"],
                            source_run_id,
                            record["tags_json"],
                            record.get("hypothesis"),
                            record.get("change_reason"),
                            record["user_metadata_json"],
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    if conn.execute(
                        "SELECT 1 FROM runs WHERE run_id=?", (run_id,)
                    ).fetchone() is not None:
                        raise RunLedgerConflictError(
                            run_id,
                            f"run_id already exists in append-only ledger: {run_id}",
                        ) from exc
                    raise
        inserted = [self.get_run(run_id) for run_id in ids]
        if any(record is None for record in inserted):  # pragma: no cover
            raise RuntimeError("run record disappeared after batch insertion")
        return [record for record in inserted if record is not None]

    def update_run_status(
        self,
        run_id: str,
        *,
        status: str,
        completed_at: str | None = None,
        certificate_id: str | None = None,
    ) -> RunRecord:
        """Update only lifecycle fields; identity and research provenance stay immutable."""

        normalized_status = str(status).strip()
        if not normalized_status:
            raise ValueError("status must be a non-empty string")
        assignments = ["status=?"]
        values: list[Any] = [normalized_status]
        if completed_at is not None:
            assignments.append("completed_at=?")
            values.append(str(completed_at))
        if certificate_id is not None:
            assignments.append("certificate_id=?")
            values.append(str(certificate_id))
        values.append(run_id)
        with self._connect() as conn:
            cursor = conn.execute(
                f"UPDATE runs SET {', '.join(assignments)} WHERE run_id=?",
                tuple(values),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown run_id: {run_id}")
        record = self.get_run(run_id)
        if record is None:  # pragma: no cover - SQLite contract guard
            raise RuntimeError("run record disappeared after lifecycle update")
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
        fields = self.envelope_record_fields(
            envelope,
            artifact_root=artifact_root,
            experiment_id=experiment_id,
            parent_run_id=parent_run_id,
            execution_identity_sha256=execution_identity_sha256,
            tags=tags,
            hypothesis=hypothesis,
            change_reason=change_reason,
            user_metadata=user_metadata,
        )
        return self.record_run(**fields)

    def envelope_record_fields(
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
    ) -> dict[str, Any]:
        """Normalize an envelope into one append-only ledger row request."""

        run_id = str(envelope["run_id"])
        return {
            "run_id": run_id,
            "experiment_id": experiment_id or self.new_experiment_id(),
            "parent_run_id": parent_run_id,
            "task_sha256": envelope.get("task_sha256"),
            "execution_identity_sha256": execution_identity_sha256,
            "operation": str(envelope.get("operation", "unknown")),
            "status": str(envelope.get("status", "unknown")),
            "protocol_version": str(envelope.get("protocol_version", "unknown")),
            "package_version": __version__,
            "archive_schema_version": ARCHIVE_SCHEMA_VERSION,
            "certificate_id": envelope.get("certificate_id"),
            "artifact_root": artifact_root,
            "cache_hit": bool(envelope.get("cache_hit", False)),
            "source_run_id": envelope.get("source_run_id"),
            "tags": tuple(tags),
            "hypothesis": hypothesis,
            "change_reason": change_reason,
            "user_metadata": dict(user_metadata or {}),
        }

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
            archive_schema_version=int(row["archive_schema_version"]),
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

    def get_lineage(
        self, run_id: str, *, detail: str | None = None
    ) -> dict[str, Any]:
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
        profile = None if detail is None else normalize_response_detail(detail)
        if profile == "compact":
            with self._connect() as conn:
                child_count = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM runs WHERE parent_run_id=?",
                        (run_id,),
                    ).fetchone()[0]
                )
                child_rows = conn.execute(
                    """
                    SELECT * FROM runs WHERE parent_run_id=?
                    ORDER BY created_at,run_id LIMIT ?
                    """,
                    (run_id, LINEAGE_MAX_RECORDS),
                ).fetchall()
            children = [self._from_row(row) for row in child_rows]
        else:
            children = self.list_children(run_id)
            child_count = len(children)
        payload = {
            "schema_version": "veritmm-lineage-v1",
            "run": current.to_dict(),
            "ancestors": [record.to_dict() for record in ancestors],
            "children": [record.to_dict() for record in children],
        }
        if profile == "compact":
            payload["children_count"] = child_count
        return payload

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
        result_path = Path(record.artifact_root, "RUN_RESULT.json")
        if not result_path.is_file():
            return None
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return record if isinstance(result, dict) and result.get("ok") is True else None

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
        detail: str = "compact",
    ) -> dict[str, Any]:
        """Copy immutable artifacts while preserving source provenance."""

        source_root = Path(source.artifact_root)
        self.assert_no_path_redirection(destination)
        target = Path(destination).resolve()
        if source_root.resolve() == target:
            raise ValueError("cache destination must differ from source artifact root")
        if target.exists() and (
            not target.is_dir() or any(target.iterdir())
        ):
            raise RunLedgerConflictError(
                new_run_id,
                "cache destination must be empty; existing artifacts are immutable",
            )
        self.assert_artifact_tree_no_links(source_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_name(f".stage-{uuid.uuid4().hex[:8]}")
        staging.mkdir()
        try:
            result = self._materialize_cache_hit_into(
                source,
                staging,
                new_run_id=new_run_id,
                detail=detail,
            )
            if target.exists():
                target.rmdir()
            staging.replace(target)
            return result
        except Exception:
            if staging.is_dir():
                shutil.rmtree(staging)
            raise

    @staticmethod
    def assert_artifact_tree_no_links(root: Path) -> None:
        root_stat = root.lstat()
        if root.is_symlink() or getattr(root_stat, "st_reparse_tag", 0):
            raise ValueError("artifact root must not be a symbolic link")
        pending = [root]
        while pending:
            current = pending.pop()
            with os.scandir(current) as entries:
                for entry in entries:
                    entry_stat = entry.stat(follow_symlinks=False)
                    if entry.is_symlink() or getattr(entry_stat, "st_reparse_tag", 0):
                        raise ValueError(
                            f"artifact trees must not contain links: {entry.name}"
                        )
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))

    @staticmethod
    def _rebase_cached_payload_identity(
        payload: dict[str, Any],
        *,
        new_run_id: str,
        source_run_id: str,
        provenance: Mapping[str, Any],
    ) -> None:
        payload["run_id"] = new_run_id
        payload.setdefault("archive_schema_version", ARCHIVE_SCHEMA_VERSION)
        payload["cache_hit"] = True
        payload["source_run_id"] = source_run_id
        payload["artifact_provenance"] = dict(provenance)
        summary = payload.get("summary")
        if isinstance(summary, dict):
            summary.setdefault("archive_schema_version", ARCHIVE_SCHEMA_VERSION)
            summary["run_id"] = new_run_id
            summary["cache_hit"] = True
            summary["source_run_id"] = source_run_id
            summary["artifact_provenance"] = dict(provenance)
            summary.pop("response", None)

    def _materialize_cache_hit_into(
        self,
        source: RunRecord,
        target: Path,
        *,
        new_run_id: str,
        detail: str,
    ) -> dict[str, Any]:
        source_root = Path(source.artifact_root)
        from .run_artifacts import validate_run_artifact_integrity
        for path in sorted(source_root.iterdir()):
            if path.name == "RUN_RESULT.json":
                continue
            destination_path = target / path.name
            if path.is_dir():
                shutil.copytree(path, destination_path, symlinks=True)
            else:
                shutil.copy2(path, destination_path)
        self.assert_artifact_tree_no_links(target)
        result = json.loads((source_root / "RUN_RESULT.json").read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError("cached RUN_RESULT.json must contain a JSON object")
        if str(result.get("run_id") or "") != source.run_id:
            raise ValueError("cached RUN_RESULT.run_id does not match source ledger")
        if result.get("ok") is not True:
            raise ValueError("only a successful source run can be materialized from cache")
        # Modern runs carry a hashed artifact index and must pass it before any
        # copied file is rewritten. Minimal pre-index legacy fixtures have no
        # references to validate and continue through the legacy normalization
        # below.
        if "artifacts" in result:
            validate_run_artifact_integrity(source_root)
        artifact_payloads: dict[str, dict[str, Any]] = {}
        for name in _RUN_SCOPED_CACHE_ARTIFACTS:
            artifact_path = target / name
            if not artifact_path.is_file():
                continue
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            if not isinstance(artifact, dict):
                raise ValueError(f"cache artifact must contain a JSON object: {name}")
            artifact_run_id = artifact.get("run_id")
            if artifact_run_id is not None and str(artifact_run_id) != source.run_id:
                raise ValueError(
                    f"cache artifact run_id does not match source ledger: {name}"
                )
            artifact_payloads[name] = artifact
        if source.operation == "sweep" and "SWEEP_RESULT.json" not in artifact_payloads:
            raise ValueError("a cached sweep source must contain SWEEP_RESULT.json")
        if source.operation != "sweep" and "SWEEP_RESULT.json" in artifact_payloads:
            raise ValueError("SWEEP_RESULT.json is inconsistent with the source operation")
        self._validate_cached_sweep_children(
            target,
            artifact_payloads.get("SWEEP_RESULT.json"),
            source_parent_run_id=source.run_id,
        )
        if not isinstance(result.get("summary"), Mapping):
            result["summary"] = dict(
                artifact_payloads.get("RESULT_SUMMARY.json")
                or {
                    "run_id": source.run_id,
                    "task_sha256": source.task_sha256,
                    "status": source.status,
                }
            )
        result.setdefault("schema_version", "veritmm-run-result-v1")
        result.setdefault("protocol_version", source.protocol_version)
        result.setdefault("task_sha256", source.task_sha256)
        result.setdefault("task_hash_scope", "normalized_operation_wrapper")
        result.setdefault("input_sha256", None)
        result.setdefault("operation", source.operation)
        result.setdefault("status", source.status)
        result.setdefault("warnings", [])
        result.setdefault("failures", [])
        result.setdefault("certificate_id", source.certificate_id)
        result.setdefault("next_machine_actions", [])
        result_summary = result["summary"]
        if isinstance(result_summary, dict):
            result_summary["run_id"] = source.run_id
            result_summary["task_sha256"] = source.task_sha256
        provenance = {
            "mode": "cache_copy",
            "source_run_id": source.run_id,
        }
        self._rebase_cached_payload_identity(
            result,
            new_run_id=new_run_id,
            source_run_id=source.run_id,
            provenance=provenance,
        )
        for name in _RUN_SCOPED_CACHE_ARTIFACTS:
            artifact_path = target / name
            artifact = artifact_payloads.get(name)
            if artifact is None:
                continue
            if name == "RESULT_SUMMARY.json":
                artifact.setdefault("task_sha256", source.task_sha256)
            self._rebase_cached_payload_identity(
                artifact,
                new_run_id=new_run_id,
                source_run_id=source.run_id,
                provenance=provenance,
            )
            write_json(artifact_path, artifact)
        self._rebase_cached_sweep_children(
            target,
            source_parent_run_id=source.run_id,
        )
        context_path = target / RESPONSE_CONTEXT_FILENAME
        self._refresh_cached_response_context(target, result)
        # Refresh hashes only after all copied run-scoped artifacts and children
        # have been integrity-checked and rebound.
        from .run_artifacts import index_artifacts, load_run_result

        result["artifacts"] = index_artifacts(target)
        write_json(target / "RUN_RESULT.json", result)
        requested_profile = normalize_response_detail(detail)
        if context_path.is_file():
            final_result = load_run_result(
                target,
                detail=requested_profile,
                force_context=True,
            )
        else:
            legacy_source = dict(result)
            legacy_source.pop("response", None)
            legacy_summary = legacy_source.get("summary")
            if isinstance(legacy_summary, dict):
                legacy_summary.pop("response", None)
            if requested_profile != "compact":
                from .run_artifacts import ResponseDetailUnavailableError

                raise ResponseDetailUnavailableError(
                    requested_profile,
                    "the cached legacy source has no RESPONSE_CONTEXT.json",
                )
            final_result = project_response(legacy_source, detail="compact")
        write_json(target / "RUN_RESULT.json", final_result)
        validate_run_artifact_integrity(target)
        return final_result

    @staticmethod
    def _root_artifact_refs(root: Path) -> list[dict[str, Any]]:
        from .run_artifacts import index_artifacts

        records = [
            item
            for item in index_artifacts(root)
            if "/" not in str(item.get("path", ""))
            and str(item.get("path")) != RESPONSE_CONTEXT_FILENAME
        ]
        return records[:CANONICAL_MAX_ARTIFACT_REFS]

    def _refresh_cached_response_context(
        self, target: Path, result: Mapping[str, Any]
    ) -> None:
        """Rebuild run-scoped context after cache identities and hashes change."""

        context_path = target / RESPONSE_CONTEXT_FILENAME
        if not context_path.is_file():
            return
        context = json.loads(context_path.read_text(encoding="utf-8"))
        if not isinstance(context, Mapping):
            raise ValueError("cached RESPONSE_CONTEXT.json must contain an object")
        rebased = rebase_response_context(context, result)
        source = rebased.get("source")
        if not isinstance(source, dict):
            raise ValueError("cached RESPONSE_CONTEXT.json source must contain an object")
        source["artifacts"] = self._root_artifact_refs(target)
        rebased["source"] = source
        write_json(context_path, rebased)

    def _validate_cached_sweep_children(
        self,
        target: Path,
        sweep: Mapping[str, Any] | None,
        *,
        source_parent_run_id: str,
    ) -> None:
        if sweep is None:
            return
        from .run_artifacts import validate_run_artifact_integrity

        children = sweep.get("children")
        if not isinstance(children, list):
            raise ValueError("cached SWEEP_RESULT.children must be a list")
        if not children:
            raise ValueError("a successful cached sweep must contain at least one child")
        children_root = (target / "children").resolve()
        seen_roots: set[Path] = set()
        seen_ids: set[str] = set()
        for child in children:
            if not isinstance(child, Mapping):
                raise ValueError("cached sweep child must be a JSON object")
            source_child_run_id = child.get("child_run_id")
            relative_root = child.get("artifact_root")
            if not source_child_run_id or not relative_root:
                raise ValueError(
                    "cached sweep child requires child_run_id and artifact_root"
                )
            source_id = str(source_child_run_id)
            if source_id in seen_ids:
                raise ValueError("cached sweep child_run_id values must be unique")
            seen_ids.add(source_id)
            source_record = self.get_run(source_id)
            if source_record is None or source_record.parent_run_id != source_parent_run_id:
                raise ValueError(
                    "cached sweep child identity is not linked to the source parent ledger"
                )
            if source_record.status not in {
                "completed",
                "physically_valid",
                "physically_valid_with_limits",
            }:
                raise ValueError("cached sweep child ledger status is not successful")
            child_root = (target / str(relative_root)).resolve()
            try:
                child_root.relative_to(children_root)
            except ValueError as exc:
                raise ValueError(
                    "cached sweep child artifact_root must stay under children/"
                ) from exc
            if child_root == children_root or child_root.parent != children_root:
                raise ValueError(
                    "cached sweep child artifact_root must name one direct child directory"
                )
            if child_root in seen_roots:
                raise ValueError("cached sweep child artifact roots must be unique")
            seen_roots.add(child_root)
            if not child_root.is_dir():
                raise ValueError(
                    f"cached sweep child artifact directory is missing: {relative_root}"
                )
            for name in ("RUN_RESULT.json", "RESULT_SUMMARY.json", "RUN_MANIFEST.json"):
                artifact_path = child_root / name
                if name == "RUN_RESULT.json" and not artifact_path.is_file():
                    raise ValueError(
                        f"cached sweep child RUN_RESULT.json is missing: {relative_root}"
                    )
                if not artifact_path.is_file():
                    continue
                payload = json.loads(artifact_path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError(
                        f"cached sweep child artifact must be a JSON object: {name}"
                    )
                artifact_run_id = payload.get("run_id")
                if name == "RUN_RESULT.json" and artifact_run_id is None:
                    raise ValueError("cached sweep child RUN_RESULT.json requires run_id")
                if artifact_run_id is not None and str(artifact_run_id) != source_id:
                    raise ValueError(
                        f"cached sweep child artifact has inconsistent run_id: {name}"
                    )
                if name == "RUN_RESULT.json" and payload.get("ok") is not True:
                    raise ValueError(
                        "cached sweep child RUN_RESULT.json must describe a successful run"
                    )
                if name == "RUN_RESULT.json" and "artifacts" in payload:
                    validate_run_artifact_integrity(child_root)
        table_path = target / "SWEEP_TABLE.csv"
        if not table_path.is_file():
            raise ValueError("cached sweep is missing SWEEP_TABLE.csv")
        if seen_ids:
            with table_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if "child_run_id" not in list(reader.fieldnames or []):
                    raise ValueError("cached SWEEP_TABLE.csv lacks child_run_id")
                table_id_list = [
                    str(row.get("child_run_id") or "").strip() for row in reader
                ]
            if any(not run_id for run_id in table_id_list):
                raise ValueError("cached SWEEP_TABLE.csv contains a blank child ID")
            if len(table_id_list) != len(set(table_id_list)):
                raise ValueError("cached SWEEP_TABLE.csv contains duplicate child IDs")
            if set(table_id_list) != seen_ids or len(table_id_list) != len(seen_ids):
                raise ValueError(
                    "cached SWEEP_TABLE.csv child IDs must exactly match SWEEP_RESULT"
                )

    def _rebase_cached_sweep_children(
        self,
        target: Path,
        *,
        source_parent_run_id: str,
    ) -> None:
        """Give every copied sweep child a fresh invocation identity.

        A cached sweep is a new parent invocation, not an alias of the source
        sweep.  Its child invocations therefore need fresh ``run_id`` values as
        well.  The original child identities remain explicit provenance links.
        """

        sweep_path = target / "SWEEP_RESULT.json"
        if not sweep_path.is_file():
            return
        sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
        if not isinstance(sweep, dict):
            raise ValueError("SWEEP_RESULT.json must contain a JSON object")
        children = sweep.get("children")
        if not isinstance(children, list):
            return

        from .run_artifacts import index_artifacts, validate_run_artifact_integrity

        child_id_map: dict[str, str] = {}
        for child in children:
            if not isinstance(child, dict):
                continue
            source_child_run_id = child.get("child_run_id")
            relative_root = child.get("artifact_root")
            if not source_child_run_id or not relative_root:
                continue
            child_root = (target / str(relative_root)).resolve()

            new_child_run_id = self.new_run_id()
            child_id_map[str(source_child_run_id)] = new_child_run_id
            provenance = {
                "mode": "cache_copy",
                "source_run_id": str(source_child_run_id),
                "source_parent_run_id": source_parent_run_id,
            }
            for name in ("RESULT_SUMMARY.json", "RUN_MANIFEST.json"):
                artifact_path = child_root / name
                if not artifact_path.is_file():
                    continue
                artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
                if not isinstance(artifact, dict):
                    raise ValueError(
                        f"cached sweep child artifact must be a JSON object: {name}"
                    )
                self._rebase_cached_payload_identity(
                    artifact,
                    new_run_id=new_child_run_id,
                    source_run_id=str(source_child_run_id),
                    provenance=provenance,
                )
                write_json(artifact_path, artifact)

            result_path = child_root / "RUN_RESULT.json"
            if not result_path.is_file():
                raise ValueError(
                    f"cached sweep child RUN_RESULT.json is missing: {relative_root}"
                )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if not isinstance(result, dict):
                raise ValueError("cached sweep child RUN_RESULT.json must be an object")
            self._rebase_cached_payload_identity(
                result,
                new_run_id=new_child_run_id,
                source_run_id=str(source_child_run_id),
                provenance=provenance,
            )
            context_path = child_root / RESPONSE_CONTEXT_FILENAME
            if context_path.is_file():
                context = json.loads(context_path.read_text(encoding="utf-8"))
                if not isinstance(context, Mapping):
                    raise ValueError(
                        "cached sweep child RESPONSE_CONTEXT.json must be an object"
                    )
                rebased = rebase_response_context(context, result)
                source = rebased.get("source")
                if not isinstance(source, dict):
                    raise ValueError(
                        "cached sweep child RESPONSE_CONTEXT.json source must be an object"
                    )
                source["artifacts"] = self._root_artifact_refs(child_root)
                rebased["source"] = source
                write_json(context_path, rebased)
            result["artifacts"] = index_artifacts(child_root)
            write_json(result_path, result)
            existing_profile = response_profile(result) or "compact"
            if context_path.is_file():
                from .run_artifacts import load_run_result

                result = load_run_result(
                    child_root,
                    detail=existing_profile,
                    force_context=True,
                )
            else:
                legacy_source = dict(result)
                legacy_source.pop("response", None)
                legacy_summary = legacy_source.get("summary")
                if isinstance(legacy_summary, dict):
                    legacy_summary.pop("response", None)
                result = project_response(legacy_source, detail=existing_profile)
            write_json(result_path, result)
            validate_run_artifact_integrity(child_root)

            child["source_child_run_id"] = str(source_child_run_id)
            child["child_run_id"] = new_child_run_id
            child["cache_hit"] = True
            child["artifact_provenance"] = provenance

        write_json(sweep_path, sweep)
        table_path = target / "SWEEP_TABLE.csv"
        if table_path.is_file() and child_id_map:
            with table_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fieldnames = list(reader.fieldnames or [])
                if "child_run_id" not in fieldnames:
                    raise ValueError("cached SWEEP_TABLE.csv lacks child_run_id")
                rows = [dict(row) for row in reader]
            for row in rows:
                source_child_id = str(row.get("child_run_id") or "")
                if source_child_id not in child_id_map:
                    raise ValueError("cached SWEEP_TABLE.csv contains an unknown child ID")
                row["child_run_id"] = child_id_map[source_child_id]
            temporary = table_path.with_suffix(".csv.tmp")
            with temporary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            temporary.replace(table_path)

    def inspect(self, run_id: str, *, detail: str = "compact") -> dict[str, Any]:
        record = self.get_run(run_id)
        if record is None:
            raise KeyError(f"unknown run_id: {run_id}")
        profile = normalize_response_detail(detail)
        root = Path(record.artifact_root)
        result_path = root / "RUN_RESULT.json"
        if result_path.is_file():
            from .run_artifacts import load_run_result_source

            result, retention = load_run_result_source(root, detail=profile)
        else:
            result = None
            retention = {
                "schema_version": "veritmm-response-retention-v1",
                "semantics": "no_run_result_available",
                "bounded": True,
                "full_profile_scope": "unavailable",
            }
        payload = {
            "schema_version": "veritmm-inspect-v2",
            "ok": True,
            "record": record.to_dict(),
            "run_result": result,
            "detail_source": retention,
            "artifacts": [] if result is None else result.get("artifacts", []),
            "artifact_root_exists": root.is_dir(),
        }
        return project_response(payload, detail=profile)


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
    "RunLedgerConflictError",
    "RunRecord",
    "compare_runs",
    "default_store_root",
]
