from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from tmm_engine import MaterialRegistry
from tmm_engine.preflight import preflight_path, preflight_task
from tmm_engine.task_io import load_task

ROOT = Path(__file__).resolve().parents[1]
INVALID = ROOT / "tests" / "fixtures" / "agent_invalid"
EXAMPLES = ROOT / "examples" / "tmm_tasks"


def _failure_codes(report: dict) -> set[str]:
    return {item["code"] for item in report["failures"]}


def test_preflight_accepts_valid_example_without_executing_solver(monkeypatch) -> None:
    from tmm_engine import workbench as workbench_module

    def forbidden(*args, **kwargs):  # pragma: no cover - called only on regression
        raise AssertionError("preflight must not execute a TMM spectrum")

    monkeypatch.setattr(workbench_module.TMMWorkbench, "simulate", forbidden)
    report = preflight_path(EXAMPLES / "periodic_dbr_simulation.json")
    assert report["ok"] is True
    assert report["status"] == "ready"
    assert report["backend_resolution"]["resolved_solver"] == "smatrix"
    assert report["estimated_work"]["wavelength_points"] == 551
    assert all(
        item.get("sampled_wavelength_count") == 551
        for item in report["materials"]
        if item["material_model"] == "tabulated_nk"
    )


def test_preflight_routes_mixed_coherence_to_byrnes() -> None:
    mode, task = load_task(EXAMPLES / "mixed_coherence_finite_substrate.json")
    task = replace(task, solver="smatrix")
    report = preflight_task(mode, task, MaterialRegistry())
    assert report["ok"] is True
    assert report["backend_resolution"] == {
        "requested_solver": "smatrix",
        "resolved_solver": "byrnes",
        "reason": "requested_outputs_or_mixed_coherence_require_reference_backend",
    }


def test_invalid_contract_is_typed_and_not_recoverable() -> None:
    report = preflight_path(INVALID / "negative_thickness.json")
    assert report["ok"] is False
    assert report["contract_valid"] is False
    assert _failure_codes(report) == {"invalid_task"}
    assert report["failures"][0]["recoverable"] is False


def test_unsupported_geometry_is_typed_and_navigable() -> None:
    report = preflight_path(INVALID / "unsupported_geometry.json")
    assert report["ok"] is False
    assert _failure_codes(report) == {"unsupported_geometry"}
    failure = report["failures"][0]
    assert failure["actions"]
    assert failure["actions"][0]["safety"] == "requires_scientific_judgment"
    assert failure["suggested_solver_family"] == "rcwa"


def test_material_not_found_is_typed_and_requires_selection() -> None:
    report = preflight_path(INVALID / "material_not_found.json")
    assert report["ok"] is False
    assert "material_not_found" in _failure_codes(report)
    failure = next(item for item in report["failures"] if item["code"] == "material_not_found")
    assert failure["requires_user_choice"] is True
    assert failure["actions"][0]["action_id"] == "select_available_material_dataset"


def test_material_ambiguity_is_typed_and_lists_candidates() -> None:
    report = preflight_path(INVALID / "material_ambiguity.json")
    assert report["ok"] is False
    assert "material_ambiguity" in _failure_codes(report)
    failure = next(
        item for item in report["failures"] if item["code"] == "material_ambiguity"
    )
    assert len(failure["context"]["candidates"]) >= 2
    assert failure["actions"][0]["action_id"] == "select_explicit_dataset_id"
    assert failure["actions"][0]["safety"] == "requires_scientific_judgment"
    assert failure["actions"][0]["context"]["candidates"] == failure["context"]["candidates"]


def test_material_range_failure_never_enables_extrapolation() -> None:
    report = preflight_path(INVALID / "material_out_of_range.json")
    assert report["ok"] is False
    assert "material_range_error" in _failure_codes(report)
    action = next(
        item for item in report["failures"] if item["code"] == "material_range_error"
    )["actions"][0]
    assert action["safety"] == "requires_scientific_judgment"
    assert action["patch"] == []


def test_numerical_risk_is_warning_not_hard_gate() -> None:
    mode, task = load_task(EXAMPLES / "periodic_dbr_simulation.json")
    sparse = replace(task, spectrum=replace(task.spectrum, points=5))
    report = preflight_task(mode, sparse)
    assert report["ok"] is True
    assert "sparse_spectral_grid" in {item["code"] for item in report["warnings"]}


def test_preflight_rejects_invalid_mode_without_raising() -> None:
    _, task = load_task(EXAMPLES / "periodic_dbr_simulation.json")
    report = preflight_task("bogus", task)
    assert report["ok"] is False
    assert report["mode"] == "unknown"
    assert report["estimated_work"] == {}
    assert _failure_codes(report) == {"invalid_task"}


def test_preflight_invalid_grid_returns_report_instead_of_secondary_exception() -> None:
    mode, task = load_task(EXAMPLES / "periodic_dbr_simulation.json")
    invalid = replace(task, spectrum=replace(task.spectrum, points=1))
    report = preflight_task(mode, invalid)
    assert report["ok"] is False
    assert report["estimated_work"] == {}
    assert _failure_codes(report) == {"invalid_task"}


def test_phase_dispersion_requires_three_wavelengths_at_preflight() -> None:
    mode, task = load_task(EXAMPLES / "periodic_dbr_simulation.json")
    task = replace(
        task,
        spectrum=replace(task.spectrum, values_nm=(500.0, 600.0), start_nm=None, stop_nm=None, points=None),
        requested_outputs=("R", "phase_dispersion"),
    )
    report = preflight_task(mode, task)
    assert report["ok"] is False
    assert _failure_codes(report) == {"unsupported_output_combination"}
    assert report["failures"][0]["context"]["minimum_wavelength_points"] == 3
