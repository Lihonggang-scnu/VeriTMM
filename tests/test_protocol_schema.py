"""Deterministic tests for the AI-facing protocol models and schemas."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from tmm_engine.capabilities import (
    FailureAction as RuntimeFailureAction,
)
from tmm_engine.capabilities import (
    FailureCode,
    FailureRecord,
)
from tmm_engine.protocol import (
    PROTOCOL_VERSION,
    ArtifactRef,
    FailureAction,
    FailureRecordModel,
    OptimizationTaskContract,
    PreflightReport,
    RunResultEnvelope,
    SensitivityTaskContract,
    SimulationTaskContract,
    SweepTaskContract,
    ToleranceTaskContract,
    export_schema,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = ROOT / "examples" / "tmm_tasks"


@pytest.mark.parametrize("path", sorted(EXAMPLE_DIR.glob("*.json")), ids=lambda p: p.name)
def test_existing_examples_are_public_schema_valid(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["mode"] == "simulate":
        contract = SimulationTaskContract.model_validate(payload)
        schema_kind = "simulation"
    elif payload["mode"] == "optimize":
        contract = OptimizationTaskContract.model_validate(payload)
        schema_kind = "optimization"
    elif payload["mode"] == "sweep":
        contract = SweepTaskContract.model_validate(payload)
        schema_kind = "sweep"
    elif payload["mode"] == "sensitivity":
        contract = SensitivityTaskContract.model_validate(payload)
        schema_kind = "sensitivity"
    else:
        contract = ToleranceTaskContract.model_validate(payload)
        schema_kind = "tolerance"

    assert contract.mode == payload["mode"]
    assert json.loads(contract.model_dump_json())
    Draft202012Validator(export_schema(schema_kind)).validate(payload)


@pytest.mark.parametrize(
    ("kind", "model_type"),
    [
        ("simulation", SimulationTaskContract),
        ("optimization", OptimizationTaskContract),
        ("sweep", SweepTaskContract),
        ("sensitivity", SensitivityTaskContract),
        ("tolerance", ToleranceTaskContract),
    ],
)
def test_task_payload_round_trip(kind: str, model_type: type[object]) -> None:
    path = EXAMPLE_DIR / {
        "simulation": "periodic_dbr_simulation.json",
        "optimization": "antireflection_optimization.json",
        "sweep": "dbr_thickness_sweep.json",
        "sensitivity": "single_film_sensitivity.json",
        "tolerance": "single_film_tolerance.json",
    }[kind]
    payload = json.loads(path.read_text(encoding="utf-8"))
    model = model_type.model_validate(payload)  # type: ignore[attr-defined]
    restored = model_type.model_validate(model.model_dump(mode="json"))  # type: ignore[attr-defined]
    assert restored == model
    assert restored.model_dump(mode="json") == model.model_dump(mode="json")  # type: ignore[attr-defined]


def test_unwrapped_v01_payload_is_normalized_to_compatible_wrapper() -> None:
    payload = json.loads(
        (EXAMPLE_DIR / "periodic_dbr_simulation.json").read_text(encoding="utf-8")
    )["simulation"]
    contract = SimulationTaskContract.model_validate(payload)
    assert contract.mode == "simulate"
    assert contract.model_dump(mode="json", exclude_unset=True)["simulation"] == payload


def test_schema_exports_are_json_serializable_202012_documents() -> None:
    for kind in (
        "simulation",
        "optimization",
        "sweep",
        "sensitivity",
        "tolerance",
        "preflight",
        "failure",
        "run_result",
    ):
        schema = export_schema(kind)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        Draft202012Validator.check_schema(schema)
        json.dumps(schema)


def test_public_contract_rejects_runtime_invalid_structure() -> None:
    payload = json.loads(
        (EXAMPLE_DIR / "periodic_dbr_simulation.json").read_text(encoding="utf-8")
    )
    payload["simulation"]["stack"]["layers"][0]["thickness_nm"] = -1
    with pytest.raises(ValueError, match="greater than 0"):
        SimulationTaskContract.model_validate(payload)


def test_schema_export_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown schema kind"):
        export_schema("not_a_schema")


def test_protocol_version_is_stable() -> None:
    assert PROTOCOL_VERSION == "veritmm-agent-v1"


def test_public_failure_preflight_artifact_and_run_models_match_runtime_outputs() -> None:
    runtime_action = RuntimeFailureAction(
        action_id="select_explicit_dataset_id",
        action_type="material_selection_required",
        description="Select a dataset.",
        safety="requires_user_input",
        patch=({"path": "material.dataset_id", "value": 418},),
        context={"material": "TiO2"},
    )
    runtime_failure = FailureRecord(
        code=FailureCode.MATERIAL_AMBIGUITY,
        message="More than one dataset matched.",
        recoverable=True,
        context={"material": "TiO2"},
        severity="error",
        requires_user_choice=True,
        actions=(runtime_action,),
    )
    failure_payload = runtime_failure.to_dict()
    failure = FailureRecordModel.model_validate(failure_payload)
    assert failure.model_dump(mode="json") == failure_payload
    assert isinstance(failure.actions[0], FailureAction)

    preflight_payload = {
        "schema_version": "veritmm-preflight-v1",
        "protocol_version": PROTOCOL_VERSION,
        "ok": False,
        "operation": "preflight",
        "mode": "simulate",
        "status": "rejected",
        "contract_valid": True,
        "capability": None,
        "backend_resolution": {
            "requested_solver": "smatrix",
            "resolved_solver": None,
            "reason": "task_rejected_before_execution",
        },
        "materials": [],
        "warnings": [],
        "failures": [failure_payload],
        "estimated_work": {},
        "study": None,
    }
    preflight = PreflightReport.model_validate(preflight_payload)
    assert preflight.model_dump(mode="json") == preflight_payload

    artifact_payload = {
        "kind": "result_summary",
        "path": "RESULT_SUMMARY.json",
        "schema_version": "veritmm-result-summary-v1",
        "sha256": "a" * 64,
        "size_bytes": 123,
    }
    artifact = ArtifactRef.model_validate(artifact_payload)
    assert artifact.model_dump(mode="json") == artifact_payload

    result_payload = {
        "schema_version": "veritmm-run-result-v1",
        "protocol_version": PROTOCOL_VERSION,
        "ok": False,
        "run_id": "run_test",
        "task_sha256": None,
        "task_hash_scope": "normalized_operation_wrapper",
        "input_sha256": None,
        "operation": "simulate",
        "status": "preflight_rejected",
        "summary": {"status": "not_certified"},
        "warnings": [],
        "failures": [failure_payload],
        "certificate_id": None,
        "artifacts": [artifact_payload],
        "next_machine_actions": [runtime_action.to_dict()],
        "cache_hit": False,
        "source_run_id": None,
        "artifact_provenance": None,
    }
    result = RunResultEnvelope.model_validate(result_payload)
    assert result.model_dump(mode="json") == result_payload
