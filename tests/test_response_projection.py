from __future__ import annotations

import json
from pathlib import Path

import pytest

from tmm_engine.protocol import (
    COMPACT_MAX_BYTES,
    ContextBudgetError,
    RunResultEnvelope,
    guard_context_budget,
    project_response,
    validate_artifact_references,
)
from tmm_engine.run_artifacts import write_json, write_run_result

_BULKY_KEY_TOKENS = (
    "spectra",
    "spectrum",
    "wavelength",
    "sample",
    "history",
    "children",
    "provenance",
    "trajectory",
    "case",
    "loss",
    "objective",
    "perturbation",
    "offset",
    "draw",
    "independentaudit",
)
_CHANNEL_KEYS = {"r", "t", "a", "reflectance", "transmittance", "absorptance"}


def _is_bulky_key(key: str, value: object, parent: dict[str, object]) -> bool:
    if not isinstance(value, list):
        return False
    normalized = "".join(character for character in key.lower() if character.isalnum())
    if normalized in _CHANNEL_KEYS:
        has_grid = any(
            "wavelength"
            in "".join(character for character in name.lower() if character.isalnum())
            for name in parent
        )
        return len(value) > 8 or has_grid
    return any(token in normalized for token in _BULKY_KEY_TOKENS)


def _walk_bulky_arrays(
    value: object,
    path: str = "",
    parent: dict[str, object] | None = None,
) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            label = f"{path}.{key}" if path else key
            if _is_bulky_key(key, child, value):
                found.append(label)
            found.extend(_walk_bulky_arrays(child, label, value))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_walk_bulky_arrays(child, f"{path}[{index}]", parent))
    return found


def _large_response(size: int) -> dict[str, object]:
    return {
        "schema_version": "synthetic-v1",
        "operation": "sweep",
        "status": "completed",
        "run_id": "run_test",
        "artifacts": [],
        "children": [{"index": index, "status": "completed"} for index in range(size)],
        "samples": [{"sample_index": index, "metric": 0.5} for index in range(size)],
        "history": [{"run_id": f"run_{index}"} for index in range(size)],
        "provenance": [{"source": f"source_{index}"} for index in range(size)],
        "trajectory": [{"step": index, "observation": "x"} for index in range(size)],
        "spectra": [[float(index), 0.5] for index in range(size)],
        "wavelengths_nm": [400.0 + index for index in range(size)],
        "wavelength_grid": [400.0 + index for index in range(size)],
        "R": [0.5 for _ in range(size)],
        "T": [0.4 for _ in range(size)],
        "A": [0.1 for _ in range(size)],
        "loss_history": [{"step": index, "loss": 0.1} for index in range(size)],
        "optimization_history": [{"step": index} for index in range(size)],
        "perturbation_offsets": [{"layer": index, "offset": 0.1} for index in range(size)],
        "draws": [{"draw": index} for index in range(size)],
        "cases": [{"case_id": f"case_{index}"} for index in range(size)],
        "independent_audit": [{"case_id": f"case_{index}"} for index in range(size)],
        "decision": {"ok": True, "score": 0.5},
    }


def test_compact_projection_removes_alias_arrays_and_scales_bounded() -> None:
    small = project_response(_large_response(32), detail="compact")
    large = project_response(_large_response(4000), detail="compact")
    small_bytes = len(json.dumps(small, ensure_ascii=False, separators=(",", ":")))
    large_bytes = len(json.dumps(large, ensure_ascii=False, separators=(",", ":")))

    assert not _walk_bulky_arrays(small)
    assert not _walk_bulky_arrays(large)
    assert large["children_count"] == 4000
    assert large["wavelengths_nm_count"] == 4000
    assert large["samples_count"] == 4000
    assert large_bytes <= COMPACT_MAX_BYTES
    assert large_bytes <= small_bytes * 2
    assert guard_context_budget(large) == len(
        json.dumps(large, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )


def test_all_profiles_externalize_bulky_arrays_and_full_keeps_richer_metadata() -> None:
    payload = _large_response(40)
    payload["scalar_metadata"] = {f"field_{index}": index for index in range(300)}
    payload["context"] = {f"context_{index}": index for index in range(300)}
    compact = project_response(payload, detail="compact")
    standard = project_response(payload, detail="standard")
    full = project_response(payload, detail="full")

    assert compact["response"]["profile"] == "compact"
    assert standard["response"]["profile"] == "standard"
    assert full["response"]["profile"] == "full"
    assert not _walk_bulky_arrays(compact)
    assert not _walk_bulky_arrays(standard)
    assert not _walk_bulky_arrays(full)
    assert "children" not in full
    assert "spectra" not in full
    assert full["children_count"] == 40
    assert len(full["scalar_metadata"]) == 300
    assert len(standard["scalar_metadata"]) <= 256
    assert len(full["context"]) == 300
    assert len(standard["context"]) <= 256
    assert full["response"]["detail_available_via_profile"] is False


def test_compact_failure_is_typed_actionable_without_false_artifact_claim() -> None:
    payload = {
        "schema_version": "veritmm-run-result-v1",
        "protocol_version": "veritmm-agent-v1",
        "ok": False,
        "run_id": "run_failure",
        "task_sha256": None,
        "task_hash_scope": "normalized_operation_wrapper",
        "input_sha256": None,
        "operation": "simulate",
        "status": "preflight_rejected",
        "summary": {"status": "preflight_rejected"},
        "warnings": [],
        "failures": [
            {
                "code": "material_ambiguity",
                "message": "Select one governed dataset.",
                "recoverable": True,
                "severity": "error",
                "requires_user_choice": True,
                "context": {"candidates": [{"dataset_id": index} for index in range(100)]},
                "actions": [
                    {
                        "action_id": "select_dataset",
                        "action_type": "material_selection_required",
                        "description": "Select an explicit dataset.",
                        "safety": "requires_user_input",
                        "patch": [{"path": "/dataset_id", "value": 418}],
                        "context": {"candidates": list(range(100))},
                    }
                ],
            }
        ],
        "certificate_id": None,
        "artifacts": [],
        "next_machine_actions": [],
    }
    compact = project_response(payload, detail="compact")
    failure = compact["failures"][0]
    action = failure["actions"][0]

    assert failure["code"] == "material_ambiguity"
    assert failure["requires_user_choice"] is True
    assert "context" not in failure
    assert action["safety"] == "requires_user_input"
    assert "context" not in action
    assert compact["summary"]["response"]["artifact_backed"] is False
    assert compact["summary"]["response"]["detail_available_via_profile"] is True
    RunResultEnvelope.model_validate(compact)


def test_compact_history_records_drop_large_user_metadata() -> None:
    payload = {
        "schema_version": "veritmm-history-v1",
        "ok": True,
        "runs": [
            {
                "run_id": "run_1",
                "operation": "simulate",
                "status": "completed",
                "user_metadata": {"blob": ["large"] * 100_000},
            }
        ],
        "artifacts": [],
    }
    compact = project_response(payload, detail="compact")
    encoded = json.dumps(compact, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    assert len(encoded) <= COMPACT_MAX_BYTES
    assert compact["runs"][0] == {
        "run_id": "run_1",
        "operation": "simulate",
        "status": "completed",
    }
    assert "user_metadata" not in compact["runs"][0]


def test_writer_emits_valid_artifact_references_and_truthful_markers(tmp_path: Path) -> None:
    write_json(tmp_path / "RESULT_SUMMARY.json", {"status": "completed"})
    write_json(tmp_path / "TOLERANCE_RESULT.json", {"samples": list(range(100))})
    payload = write_run_result(
        tmp_path,
        operation="simulate",
        task_sha256="a" * 64,
        status="completed",
        ok=True,
        summary={"samples": list(range(100))},
    )

    assert payload["summary"]["response"]["profile"] == "compact"
    assert payload["summary"]["samples_artifact_backed"] is True
    assert payload["summary"]["response"]["artifact_backed"] is True
    assert validate_artifact_references(payload, root=tmp_path)
    for reference in payload["artifacts"]:
        assert (tmp_path / reference["path"]).is_file()


def test_unrelated_artifact_does_not_claim_samples_are_backed(tmp_path: Path) -> None:
    write_json(tmp_path / "SIMULATION_RESULT.json", {"values": [1, 2, 3]})
    payload = write_run_result(
        tmp_path,
        operation="simulate",
        task_sha256="a" * 64,
        status="completed",
        ok=True,
        summary={"samples": list(range(100))},
    )

    assert payload["summary"]["samples_artifact_backed"] is False
    assert payload["summary"]["response"]["artifact_backed"] is True


def test_context_guard_rejects_unprojected_bulky_alias() -> None:
    with pytest.raises(ContextBudgetError, match="forbidden large-payload arrays"):
        guard_context_budget({"wavelengths_nm": list(range(10))})


def test_small_summary_intervals_and_actions_remain_inline() -> None:
    payload = {
        "interval": [0.2, 0.3],
        "R": [0.4, 0.5],
        "actions": [{"action_id": "retry", "safety": "safe"}],
        "artifacts": [],
    }
    compact = project_response(payload, detail="compact")

    assert compact["interval"] == [0.2, 0.3]
    assert compact["R"] == [0.4, 0.5]
    assert compact["actions"][0]["action_id"] == "retry"


def test_compact_summary_keeps_metrics_without_interval_or_layer_repetition() -> None:
    metric = {
        "finite": True,
        "minimum": 0.1,
        "minimum_wavelength_nm": 500.0,
        "maximum": 0.9,
        "maximum_wavelength_nm": 600.0,
        "mean": 0.5,
        "bands_ge_0_90_nm": {
            "intervals_nm": [[510.0, 520.0], [540.0, 550.0]],
            "total_interval_count": 2,
            "truncated": False,
        },
    }
    payload = {
        "spectral_features": {f"channel_{index}": {"R": metric} for index in range(20)},
        "materials": [
            {
                "stack_position": index,
                "provider": "bundled",
                "dataset_id": "SiO2",
                "material": "SiO2",
            }
            for index in range(100)
        ],
        "artifacts": [],
    }

    compact = project_response(payload)
    assert compact["spectral_features"]["channel_count"] == 20
    assert compact["spectral_features"]["included_channel_count"] == 12
    first_metric = compact["spectral_features"]["channel_0"]["R"]
    assert first_metric["maximum"] == 0.9
    assert "intervals_nm" not in first_metric["bands_ge_0_90_nm"]
    assert compact["materials"] == [
        {
            "provider": "bundled",
            "dataset_id": "SiO2",
            "material": "SiO2",
            "stack_occurrence_count": 100,
        }
    ]


def test_unknown_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown response detail"):
        project_response({"artifacts": []}, detail="verbose")
