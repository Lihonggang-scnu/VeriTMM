from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from tmm_engine import ExecutionSettings
from tmm_engine.experiment_store import ExperimentStore
from tmm_engine.managed_execution import execute_managed_task
from tmm_engine.protocol import COMPACT_MAX_BYTES, SweepTaskContract, project_response
from tmm_engine.run_artifacts import write_run_result

ROOT = Path(__file__).resolve().parents[1]


def _record(
    store: ExperimentStore,
    root: Path,
    run_id: str,
    *,
    parent_run_id: str | None = None,
    created_at: str,
    user_metadata: dict[str, object] | None = None,
) -> None:
    artifact_root = root / run_id
    artifact_root.mkdir(parents=True)
    store.record_run(
        run_id=run_id,
        experiment_id="exp_lineage",
        parent_run_id=parent_run_id,
        task_sha256=run_id.encode().hex().ljust(64, "0")[:64],
        execution_identity_sha256=(run_id + "identity").encode().hex().ljust(64, "0")[:64],
        operation="simulate",
        status="physically_valid",
        protocol_version="veritmm-agent-v1",
        package_version="0.3.0",
        artifact_root=artifact_root,
        created_at=created_at,
        completed_at=created_at,
        user_metadata=user_metadata,
    )


def _sweep() -> object:
    document = {
        "schema_version": "sweep-task-v1",
        "mode": "sweep",
        "sweep": {
            "base_simulation": {
                "stack": {
                    "layers": [{"constant_n": 2.0, "thickness_nm": 100.0}],
                    "incident": {"constant_n": 1.0},
                    "exit": {"constant_n": 1.5},
                },
                "spectrum": {"start_nm": 500.0, "stop_nm": 600.0, "points": 11},
                "illumination": {
                    "angles_deg": [0.0],
                    "polarizations": ["unpolarized"],
                },
            },
            "parameters": [
                {"path": "/stack/layers/0/thickness_nm", "values": [90.0, 100.0]}
            ],
            "metrics": [{"name": "mean_R", "observable": "R", "aggregation": "mean"}],
        },
    }
    return SweepTaskContract.model_validate(document).sweep


def test_parent_child_and_grandchild_lineage_is_queryable(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / ".veritmm")
    artifacts = tmp_path / "artifacts"
    _record(store, artifacts, "run_root", created_at="2026-01-01T00:00:00+00:00")
    _record(
        store,
        artifacts,
        "run_child_a",
        parent_run_id="run_root",
        created_at="2026-01-01T00:00:01+00:00",
    )
    _record(
        store,
        artifacts,
        "run_child_b",
        parent_run_id="run_root",
        created_at="2026-01-01T00:00:02+00:00",
    )
    _record(
        store,
        artifacts,
        "run_grandchild",
        parent_run_id="run_child_a",
        created_at="2026-01-01T00:00:03+00:00",
    )

    assert [item.run_id for item in store.list_children("run_root")] == [
        "run_child_a",
        "run_child_b",
    ]
    lineage = store.get_lineage("run_grandchild")
    assert lineage["schema_version"] == "veritmm-lineage-v1"
    assert lineage["run"]["run_id"] == "run_grandchild"
    assert [item["run_id"] for item in lineage["ancestors"]] == [
        "run_root",
        "run_child_a",
    ]
    assert lineage["children"] == []


def test_compact_lineage_inlines_bounded_experiment_child_identities(
    tmp_path: Path,
) -> None:
    store = ExperimentStore(tmp_path / ".veritmm")
    artifacts = tmp_path / "artifacts"
    _record(store, artifacts, "run_lineage_root", created_at="2026-01-01T00:00:00+00:00")
    for index in range(40):
        _record(
            store,
            artifacts,
            f"run_lineage_child_{index:02d}",
            parent_run_id="run_lineage_root",
            created_at=f"2026-01-01T00:00:{index + 1:02d}+00:00",
        )

    bounded_source = store.get_lineage("run_lineage_root", detail="compact")
    assert len(bounded_source["children"]) == 16
    assert bounded_source["children_count"] == 40
    compact = project_response({"ok": True, **bounded_source}, detail="compact")
    children = compact["children"]

    assert len(children) == 16
    assert compact["children_count"] == 40
    assert compact["children_truncated_count"] == 24
    assert compact["children_truncated"] is True
    assert set(children[0]) == {
        "run_id",
        "operation",
        "status",
        "parent_run_id",
        "created_at",
    }
    assert all("child_run_id" not in item for item in children)


def test_managed_sweep_records_child_runs_under_parent(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / ".veritmm")
    parent = execute_managed_task(
        "sweep",
        _sweep(),
        tmp_path / "study",
        store=store,
        experiment_id="exp_sweep_lineage",
        execution_settings=ExecutionSettings(
            write_plot=False, convergence_max_refinements=1
        ),
        cache=False,
    )

    children = store.list_children(parent["run_id"])
    assert parent["operation"] == "sweep"
    assert len(children) == 2
    assert all(item.parent_run_id == parent["run_id"] for item in children)
    assert all(item.experiment_id == "exp_sweep_lineage" for item in children)
    assert all(Path(item.artifact_root, "RUN_RESULT.json").is_file() for item in children)
    assert all("sweep_child" in item.tags for item in children)

    lineage = store.get_lineage(parent["run_id"])
    assert len(lineage["children"]) == 2
    assert {item["run_id"] for item in lineage["children"]} == {
        item.run_id for item in children
    }


def test_history_and_lineage_cli_emit_one_parseable_json_object(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / ".veritmm")
    _record(
        store,
        tmp_path / "artifacts",
        "run_cli",
        created_at="2026-01-01T00:00:00+00:00",
    )
    store_dir = str(tmp_path / ".veritmm")
    for command in (
        ["history", "--store-dir", store_dir, "--json"],
        ["inspect", "run_cli", "--store-dir", store_dir, "--json"],
        ["lineage", "run_cli", "--store-dir", store_dir, "--json"],
    ):
        result = subprocess.run(
            [sys.executable, "-m", "tmm_engine.cli", *command],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert isinstance(payload, dict)
        assert result.stdout.count("\n") <= 1


def test_inspect_cli_projects_whole_document_and_bounds_unicode_metadata(
    tmp_path: Path,
) -> None:
    store = ExperimentStore(tmp_path / ".veritmm")
    artifacts = tmp_path / "artifacts"
    metadata = {"notes": ["測試-данные-Δ" * 8 for _ in range(4000)]}
    _record(
        store,
        artifacts,
        "run_inspect_budget",
        created_at="2026-01-01T00:00:00+00:00",
        user_metadata=metadata,
    )
    run_root = artifacts / "run_inspect_budget"
    write_run_result(
        run_root,
        operation="simulate",
        task_sha256="d" * 64,
        status="completed",
        ok=True,
        summary={"diagnostics": {f"field_{index}": index for index in range(400)}},
        run_id="run_inspect_budget",
    )
    env = {
        **os.environ,
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    payloads: dict[str, dict[str, object]] = {}
    sizes: dict[str, int] = {}
    for detail in ("compact", "standard", "full"):
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "tmm_engine.cli",
                "inspect",
                "run_inspect_budget",
                "--store-dir",
                str(tmp_path / ".veritmm"),
                "--detail",
                detail,
                "--json",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            timeout=60,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert len(completed.stdout.splitlines()) == 1
        payload = json.loads(completed.stdout)
        payloads[detail] = payload
        sizes[detail] = len(completed.stdout.encode("utf-8"))
        assert payload["response"]["profile"] == detail
        assert "response" not in payload["run_result"]["summary"]

    assert sizes["compact"] <= COMPACT_MAX_BYTES
    assert "user_metadata" not in payloads["compact"]["record"]
    assert len(payloads["standard"]["record"]["user_metadata"]["notes"]) == 256
    assert len(payloads["full"]["record"]["user_metadata"]["notes"]) == 4000
    assert sizes["full"] > sizes["standard"] > sizes["compact"]
