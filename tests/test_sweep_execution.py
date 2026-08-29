from __future__ import annotations

import csv
import json
from pathlib import Path

from tmm_engine import ExecutionSettings
from tmm_engine.protocol import RunResultEnvelope, SweepTaskContract
from tmm_engine.run_artifacts import stable_payload_sha256
from tmm_engine.sweep import SweepExecutionSettings, execute_sweep, expand_sweep


def _sweep(
    parameters: list[dict[str, object]],
    metrics: list[dict[str, object]] | None = None,
):
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
            "parameters": parameters,
            "metrics": metrics
            or [
                {
                    "name": "mean_R",
                    "observable": "R",
                    "wavelength_min_nm": 500.0,
                    "wavelength_max_nm": 600.0,
                    "aggregation": "mean",
                }
            ],
        },
    }
    return SweepTaskContract.model_validate(document).sweep


def _settings() -> SweepExecutionSettings:
    return SweepExecutionSettings(
        child_execution=ExecutionSettings(
            write_plot=False,
            convergence_max_refinements=1,
        )
    )


def test_one_dimensional_sweep_writes_parent_table_and_child_artifacts(
    tmp_path: Path,
) -> None:
    sweep = _sweep(
        [{"path": "/stack/layers/0/thickness_nm", "values": [90.0, 100.0, 110.0]}],
        metrics=[
            {"name": "mean_R", "observable": "R", "aggregation": "mean"},
            {"name": "min_R", "observable": "R", "aggregation": "min"},
            {"name": "max_R", "observable": "R", "aggregation": "max"},
            {
                "name": "worst_R",
                "observable": "R",
                "aggregation": "worst_case",
                "threshold_direction": "at_least",
            },
            {
                "name": "value_R",
                "observable": "R",
                "aggregation": "value_at_wavelength",
                "wavelength_nm": 550.0,
            },
            {
                "name": "band_R",
                "observable": "R",
                "aggregation": "threshold_band_width",
                "threshold": 0.0,
            },
        ],
    )
    output = tmp_path / "sweep"
    envelope = execute_sweep(sweep, output, settings=_settings())
    result = json.loads((output / "SWEEP_RESULT.json").read_text(encoding="utf-8"))

    assert envelope["ok"] is True
    assert envelope["status"] == "completed"
    RunResultEnvelope.model_validate(envelope)
    assert result["child_count"] == 3
    assert result["successful_child_count"] == 3
    assert result["failed_child_count"] == 0
    assert result["pending_child_count"] == 0
    assert [child["index"] for child in result["children"]] == [0, 1, 2]
    assert len({child["child_run_id"] for child in result["children"]}) == 3

    expanded = expand_sweep(sweep)
    for child, expected in zip(result["children"], expanded, strict=True):
        assert child["child_task_sha256"] == expected["child_task_sha256"]
        child_root = output / child["artifact_root"]
        assert child_root.is_dir()
        assert {
            "RUN_RESULT.json",
            "RESULT_SUMMARY.json",
            "NORMALIZED_TASK.json",
            "PHYSICS_ACCEPTANCE_CERTIFICATE.json",
            "SPECTRA.csv",
        } <= {path.name for path in child_root.iterdir()}
        assert set(child["metrics"]) == {
            "mean_R",
            "min_R",
            "max_R",
            "worst_R",
            "value_R",
            "band_R",
        }
        assert all(value >= 0.0 for value in child["metrics"].values())

    with (output / "SWEEP_TABLE.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0][:4] == ["index", "child_run_id", "status", "ok"]
    assert len(rows) == 4
    assert json.loads((output / "RUN_RESULT.json").read_text(encoding="utf-8"))["ok"] is True


def test_cartesian_sweep_has_stable_declaration_and_value_order() -> None:
    sweep = _sweep(
        [
            {"path": "/stack/layers/0/thickness_nm", "values": [90.0, 100.0]},
            {"path": "/illumination/angles_deg/0", "values": [0.0, 30.0]},
        ]
    )
    rows = expand_sweep(sweep)
    assert [
        [(item["path"], item["value"]) for item in row["parameters"]]
        for row in rows
    ] == [
        [
            ("/stack/layers/0/thickness_nm", 90.0),
            ("/illumination/angles_deg/0", 0.0),
        ],
        [
            ("/stack/layers/0/thickness_nm", 90.0),
            ("/illumination/angles_deg/0", 30.0),
        ],
        [
            ("/stack/layers/0/thickness_nm", 100.0),
            ("/illumination/angles_deg/0", 0.0),
        ],
        [
            ("/stack/layers/0/thickness_nm", 100.0),
            ("/illumination/angles_deg/0", 30.0),
        ],
    ]
    assert [row["index"] for row in rows] == list(range(4))
    assert len({row["child_task_sha256"] for row in rows}) == 4
    for row in rows:
        assert row["child_task_sha256"] == stable_payload_sha256(
            {"mode": "simulate", "simulation": row["simulation"]}
        )


def test_partial_child_failure_preserves_typed_child_provenance(tmp_path: Path) -> None:
    sweep = _sweep(
        [{"path": "/spectrum/start_nm", "values": [500.0, 700.0]}]
    )
    output = tmp_path / "partial"
    envelope = execute_sweep(sweep, output, settings=_settings())
    result = json.loads((output / "SWEEP_RESULT.json").read_text(encoding="utf-8"))

    assert envelope["ok"] is False
    assert result["status"] == "completed"
    assert result["partial_success"] is True
    assert result["child_count"] == 2
    assert result["successful_child_count"] == 1
    assert result["failed_child_count"] == 1
    failed = next(child for child in result["children"] if not child["ok"])
    successful = next(child for child in result["children"] if child["ok"])
    assert failed["status"] == "failed"
    assert failed["child_task_sha256"]
    assert failed["failures"]
    assert failed["failures"][0]["code"] == "invalid_task"
    assert successful["status"] == "completed"
    assert successful["artifact_root"]
    assert (output / "SWEEP_TABLE.csv").is_file()

