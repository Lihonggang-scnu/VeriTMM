from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tmm_engine import OptimizationTask, SpectralTarget
from tmm_engine.task_io import load_task, simulation_task_from_dict, write_normalized_task


ROOT = Path(__file__).resolve().parents[1]


def _simulation_payload() -> dict:
    return {
        "stack": {
            "name": "constant_demo",
            "incident": {"constant_n": 1.0},
            "layers": [
                {
                    "constant_n": 2.1,
                    "thickness_nm": 100.0,
                    "optimizable": True,
                    "min_thickness_nm": 20.0,
                    "max_thickness_nm": 250.0,
                }
            ],
            "exit": {"constant_n": 1.5},
        },
        "spectrum": {"start_nm": 450.0, "stop_nm": 750.0, "points": 21},
        "illumination": {"angles_deg": [0.0, 45.0], "polarizations": ["s", "p"]},
        "solver": "smatrix",
    }


def test_json_round_trip_preserves_standard_contract(tmp_path: Path) -> None:
    task = simulation_task_from_dict(_simulation_payload())
    path = tmp_path / "task.json"
    write_normalized_task(path, "simulate", task)
    mode, restored = load_task(path)
    assert mode == "simulate"
    assert restored == task


def test_task_contract_rejects_missing_layer_material_model() -> None:
    payload = _simulation_payload()
    del payload["stack"]["layers"][0]["constant_n"]
    with pytest.raises(ValueError, match="exactly one"):
        simulation_task_from_dict(payload)


def test_forward_cli_writes_auditable_artifacts(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps({"mode": "simulate", "simulation": _simulation_payload()}), encoding="utf-8"
    )
    output_dir = tmp_path / "run"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_tmm_task.py"),
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads((output_dir / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["physics_audit"]["passivity_check_passed"]
    assert manifest["acceptance_certificate_status"] == "physically_valid"
    for name in (
        "NORMALIZED_TASK.json",
        "SIMULATION_RESULT.json",
        "SPECTRA.csv",
        "PHYSICS_ACCEPTANCE_CERTIFICATE.json",
    ):
        assert (output_dir / name).exists()
    assert (output_dir / "SPECTRA.png").exists() or (
        output_dir / "SPECTRA_PLOT_SKIPPED.txt"
    ).exists()


def test_optimization_contract_rejects_undeclared_target_channel() -> None:
    simulation = simulation_task_from_dict(_simulation_payload())
    task = OptimizationTask(
        simulation=simulation,
        targets=(SpectralTarget("R", 0.9, 500.0, 600.0, angle_deg=30.0),),
    )
    with pytest.raises(ValueError, match="declared"):
        task.validate()


def test_material_search_cli_exports_selected_rii_dataset(tmp_path: Path) -> None:
    output = tmp_path / "search.json"
    export = tmp_path / "nk.csv"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "search_optical_material.py"),
            "TiO2",
            "--provider",
            "rii",
            "--start-nm",
            "500",
            "--stop-nm",
            "800",
            "--dataset-id",
            "418",
            "--sample-points",
            "5",
            "--output",
            str(output),
            "--export-csv",
            str(export),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["selected_dataset"]["dataset_id"] == 418
    assert payload["catalog_status"]["rii_sqlite"]["page_count"] > 2_000
    assert export.exists()
