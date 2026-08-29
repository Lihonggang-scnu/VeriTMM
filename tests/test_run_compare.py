from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tmm_engine.experiment_store import ExperimentStore, compare_runs
from tmm_engine.run_artifacts import stable_payload_sha256, write_json, write_run_result

ROOT = Path(__file__).resolve().parents[1]


def _write_run(
    store: ExperimentStore,
    root: Path,
    run_id: str,
    *,
    thickness_nm: float,
    dataset_id: int,
    solver: str,
) -> None:
    artifact_root = root / run_id
    artifact_root.mkdir(parents=True)
    task = {
        "mode": "simulate",
        "simulation": {
            "stack": {
                "layers": [
                    {
                        "material": "TiO2",
                        "provider": "rii",
                        "dataset_id": dataset_id,
                        "thickness_nm": thickness_nm,
                    }
                ]
            },
            "solver": solver,
        },
    }
    summary = {
        "schema_version": "veritmm-result-summary-v1",
        "status": "physically_valid",
        "solver": solver,
        "physics": {"accepted": True},
    }
    certificate = {
        "schema_version": "physics-acceptance-certificate-v1",
        "accepted": True,
        "status": "physically_valid",
        "certificate_id": f"{run_id}_certificate",
    }
    write_json(artifact_root / "NORMALIZED_TASK.json", task)
    write_json(artifact_root / "RESULT_SUMMARY.json", summary)
    write_json(artifact_root / "PHYSICS_ACCEPTANCE_CERTIFICATE.json", certificate)
    task_sha256 = stable_payload_sha256(task)
    envelope = write_run_result(
        artifact_root,
        operation="simulate",
        task_sha256=task_sha256,
        status="physically_valid",
        ok=True,
        summary=summary,
        certificate_id=certificate["certificate_id"],
        run_id=run_id,
    )
    store.record_envelope(envelope, artifact_root=artifact_root, experiment_id="exp_compare")


def test_compare_reports_stable_machine_deltas_and_material_dataset_changes(
    tmp_path: Path,
) -> None:
    store = ExperimentStore(tmp_path / ".veritmm")
    _write_run(
        store,
        tmp_path / "artifacts",
        "run_a",
        thickness_nm=90.0,
        dataset_id=101,
        solver="smatrix",
    )
    _write_run(
        store,
        tmp_path / "artifacts",
        "run_b",
        thickness_nm=96.0,
        dataset_id=202,
        solver="byrnes",
    )

    first = compare_runs(store, "run_a", "run_b")
    second = compare_runs(store, "run_a", "run_b")
    assert first == second
    assert set(first) == {
        "schema_version",
        "run_a",
        "run_b",
        "task_diff",
        "material_diff",
        "solver_diff",
        "summary_diff",
        "certificate_diff",
        "artifact_diff",
    }
    assert first["schema_version"] == "veritmm-run-compare-v1"
    assert {
        item["path"] for item in first["task_diff"]
    } == {
        "/simulation/stack/layers/0/dataset_id",
        "/simulation/stack/layers/0/thickness_nm",
        "/simulation/solver",
    }
    assert [item["path"] for item in first["material_diff"]] == [
        "/simulation/stack/layers/0/dataset_id"
    ]
    assert first["solver_diff"] == {"from": "smatrix", "to": "byrnes"}
    assert "better" not in json.dumps(first, sort_keys=True).lower()
    assert "preference" not in json.dumps(first, sort_keys=True).lower()


def test_compare_cli_stdout_is_one_json_object(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / ".veritmm")
    _write_run(
        store,
        tmp_path / "artifacts",
        "run_a",
        thickness_nm=90.0,
        dataset_id=101,
        solver="smatrix",
    )
    _write_run(
        store,
        tmp_path / "artifacts",
        "run_b",
        thickness_nm=90.0,
        dataset_id=101,
        solver="smatrix",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tmm_engine.cli",
            "compare",
            "run_a",
            "run_b",
            "--store-dir",
            str(tmp_path / ".veritmm"),
            "--json",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["task_diff"] == []
    assert result.stdout.count("\n") <= 1
