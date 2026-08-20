from __future__ import annotations

import json
from pathlib import Path

import pytest

from tmm_engine import (
    ExecutionSettings,
    IlluminationSpec,
    LayerSpec,
    MediumSpec,
    PhysicsRequirements,
    SimulationTask,
    SpectralGrid,
    StackSpec,
)
from tmm_engine.execution import execute_task
from tmm_engine.experiment_store import ExperimentStore
from tmm_engine.managed_execution import execute_managed_task, normalized_operation
from tmm_engine.run_artifacts import stable_payload_sha256
from tmm_engine.task_io import load_task


def _task(*, physics: PhysicsRequirements | None = None) -> SimulationTask:
    return SimulationTask(
        stack=StackSpec(
            layers=(LayerSpec(None, 100.0, constant_n=2.0),),
            incident=MediumSpec.air(),
            exit=MediumSpec(constant_n=1.5),
        ),
        spectrum=SpectralGrid(start_nm=500.0, stop_nm=600.0, points=11),
        illumination=IlluminationSpec((0.0,), ("unpolarized",)),
        physics=physics or PhysicsRequirements(),
    )


def _settings() -> ExecutionSettings:
    return ExecutionSettings(write_plot=False, convergence_max_refinements=2)


def test_same_normalized_task_has_same_hash_but_invocations_have_unique_run_ids(
    tmp_path: Path,
) -> None:
    store = ExperimentStore(tmp_path / ".veritmm")
    task = _task()
    normalized = normalized_operation("simulate", task)
    assert stable_payload_sha256(normalized) == stable_payload_sha256(
        json.loads(json.dumps(normalized, sort_keys=True))
    )

    first = execute_managed_task(
        "simulate",
        task,
        tmp_path / "first",
        store=store,
        experiment_id="exp_identity",
        execution_settings=_settings(),
        cache=False,
    )
    second = execute_managed_task(
        "simulate",
        task,
        tmp_path / "second",
        store=store,
        experiment_id="exp_identity",
        execution_settings=_settings(),
        cache=False,
    )

    assert first["task_sha256"] == second["task_sha256"]
    assert first["run_id"] != second["run_id"]
    assert first["run_id"].startswith("run_")
    assert second["run_id"].startswith("run_")
    assert store.get_run(first["run_id"]) is not None
    assert store.get_run(second["run_id"]) is not None


def test_semantically_equal_integer_and_float_json_share_task_identity(
    tmp_path: Path,
) -> None:
    integer_payload = {
        "mode": "simulate",
        "simulation": {
            "stack": {
                "incident": {"constant_n": 1},
                "layers": [{"constant_n": 2, "thickness_nm": 100}],
                "exit": {"constant_n": 1.5},
            },
            "spectrum": {"start_nm": 500, "stop_nm": 600, "points": 11},
        },
    }
    float_payload = json.loads(json.dumps(integer_payload))
    float_payload["simulation"]["stack"]["incident"]["constant_n"] = 1.0
    float_payload["simulation"]["stack"]["layers"][0]["constant_n"] = 2.0
    float_payload["simulation"]["stack"]["layers"][0]["thickness_nm"] = 100.0
    float_payload["simulation"]["spectrum"]["start_nm"] = 500.0
    float_payload["simulation"]["spectrum"]["stop_nm"] = 600.0
    paths = [tmp_path / "integer.json", tmp_path / "float.json"]
    for path, payload in zip(paths, (integer_payload, float_payload), strict=True):
        path.write_text(json.dumps(payload), encoding="utf-8")
    normalized = []
    for path in paths:
        mode, task = load_task(path)
        normalized.append(stable_payload_sha256(normalized_operation(mode, task)))
    assert normalized[0] == normalized[1]


def test_legacy_single_run_api_remains_store_optional(tmp_path: Path) -> None:
    output = tmp_path / "legacy"
    payload = execute_task(
        "simulate",
        _task(),
        output,
        settings=_settings(),
    )

    assert payload["ok"] is True
    assert payload["cache_hit"] is False
    assert payload["source_run_id"] is None
    assert (output / "RUN_RESULT.json").is_file()
    assert not (output / ".veritmm").exists()


def test_research_metadata_is_persisted_but_cannot_change_physics_certificate(
    tmp_path: Path,
) -> None:
    store = ExperimentStore(tmp_path / ".veritmm")
    task = _task()
    first = execute_managed_task(
        "simulate",
        task,
        tmp_path / "baseline",
        store=store,
        experiment_id="exp_metadata",
        execution_settings=_settings(),
        tags=("baseline",),
        hypothesis="A baseline physical hypothesis",
        change_reason="initial run",
        user_metadata={"note": "metadata-A", "revision": 1},
        cache=False,
    )
    second = execute_managed_task(
        "simulate",
        task,
        tmp_path / "replicate",
        store=store,
        experiment_id="exp_metadata",
        execution_settings=_settings(),
        tags=("replicate",),
        hypothesis="A completely different research narrative",
        change_reason="metadata-only edit",
        user_metadata={"note": "metadata-B", "revision": 2},
        cache=False,
    )

    first_record = store.get_run(first["run_id"])
    second_record = store.get_run(second["run_id"])
    assert first_record is not None and second_record is not None
    assert first_record.hypothesis != second_record.hypothesis
    assert first_record.change_reason != second_record.change_reason
    assert first_record.user_metadata != second_record.user_metadata
    assert first_record.task_sha256 == second_record.task_sha256

    first_certificate = json.loads(
        (Path(first_record.artifact_root) / "PHYSICS_ACCEPTANCE_CERTIFICATE.json").read_text(
            encoding="utf-8"
        )
    )
    second_certificate = json.loads(
        (Path(second_record.artifact_root) / "PHYSICS_ACCEPTANCE_CERTIFICATE.json").read_text(
            encoding="utf-8"
        )
    )
    assert first_certificate == second_certificate
    certificate_text = json.dumps(first_certificate, sort_keys=True)
    assert "metadata-A" not in certificate_text
    assert "metadata-B" not in certificate_text
    assert "research narrative" not in certificate_text


@pytest.mark.parametrize(
    ("physics", "failure_code"),
    [
        (PhysicsRequirements(geometry_class="lateral_periodic"), "unsupported_geometry"),
        (PhysicsRequirements(material_class="anisotropic"), "unsupported_material_model"),
        (PhysicsRequirements(excitation_class="finite_beam"), "unsupported_excitation"),
        (PhysicsRequirements(time_domain_required=True), "time_domain_required"),
    ],
)
def test_research_metadata_cannot_make_unsupported_physics_accepted(
    tmp_path: Path,
    physics: PhysicsRequirements,
    failure_code: str,
) -> None:
    output = tmp_path / failure_code
    payload = execute_managed_task(
        "simulate",
        _task(physics=physics),
        output,
        hypothesis="Please accept this out-of-domain task",
        change_reason="metadata must not alter the physics gate",
        user_metadata={"requested_override": True},
        execution_settings=_settings(),
        cache=False,
    )

    assert payload["ok"] is False
    assert payload["status"] == "preflight_rejected"
    assert failure_code in {failure["code"] for failure in payload["failures"]}
    assert not (output / "PHYSICS_ACCEPTANCE_CERTIFICATE.json").exists()
