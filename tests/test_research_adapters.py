from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tmm_engine import __version__
from tmm_engine.execution import ExecutionSettings
from tmm_engine.research import (
    AddLayerAction,
    BatchEvaluationRequest,
    ChooseMaterialAction,
    ChooseThicknessAction,
    ContinuousThicknessVariable,
    DatasetMaterialIdentity,
    DatasetRecord,
    DatasetWavelengthConfig,
    DesignSpace,
    DesignSpaceContract,
    DesignSpaceEnvironment,
    EvaluationRecord,
    EvaluatorConfig,
    MaterialChoiceVariable,
    MaterialOption,
    ObjectiveScore,
    ObjectiveSet,
    ObjectiveSpec,
    ObjectiveValue,
    OptimizerAdapter,
    OptionalDependencyError,
    RandomSearchAdapter,
    RemoveLayerAction,
    ResearchEnvironment,
    ResearchEvaluator,
    StopAction,
    VerifiedTorchDataset,
)
from tmm_engine.research.batch import BATCH_INDEX_SCHEMA_VERSION
from tmm_engine.schemas import (
    IlluminationSpec,
    LayerSpec,
    MediumSpec,
    SimulationTask,
    SpectralGrid,
    StackSpec,
)


def _task(*, mixed: bool = False) -> SimulationTask:
    layers = [
        LayerSpec(
            None,
            100.0,
            constant_n=2.0,
            min_thickness_nm=90.0,
            max_thickness_nm=110.0,
        )
    ]
    if mixed:
        layers.append(LayerSpec(None, 80.0, constant_n=1.5))
    return SimulationTask(
        stack=StackSpec(
            layers=tuple(layers),
            incident=MediumSpec(constant_n=1.0),
            exit=MediumSpec(constant_n=1.5),
        ),
        spectrum=SpectralGrid(start_nm=500.0, stop_nm=600.0, points=7),
        illumination=IlluminationSpec((0.0,), ("unpolarized",)),
        requested_outputs=("R", "T", "A"),
    )


def _space(*, mixed: bool = False, shifted: bool = False) -> DesignSpace:
    lower = 92.0 if shifted else 90.0
    variables: list[object] = [
        ContinuousThicknessVariable(
            name="thickness_nm",
            layer_index=0,
            lower_nm=lower,
            upper_nm=110.0,
        )
    ]
    if mixed:
        variables.append(
            MaterialChoiceVariable(
                name="material",
                layer_index=1,
                options=(
                    MaterialOption(name="low", constant_n=1.4),
                    MaterialOption(name="high", constant_n=1.8),
                ),
            )
        )
    return DesignSpace(
        DesignSpaceContract(base_task=_task(mixed=mixed), variables=tuple(variables))
    )


def _objectives() -> ObjectiveSet:
    return ObjectiveSet(
        objectives=(
            ObjectiveSpec(
                name="mean-R",
                direction="maximize",
                observable="R",
                wavelength_min_nm=500.0,
                wavelength_max_nm=600.0,
                aggregation="mean",
            ),
        )
    )


def _evaluator(tmp_path: Path, space: DesignSpace) -> ResearchEvaluator:
    return ResearchEvaluator(
        space,
        _objectives(),
        EvaluatorConfig(output_root=str(tmp_path / "evaluations"), cache=False),
        execution_settings=ExecutionSettings(
            write_plot=False,
            convergence_max_refinements=2,
        ),
    )


def _completed_record(
    candidate_id: str,
    space: DesignSpace,
    *,
    score: float,
    feasible: bool = True,
) -> EvaluationRecord:
    return EvaluationRecord(
        candidate_id=candidate_id,
        design_space_id=space.design_space_id,
        objective_set_id=_objectives().objective_set_id,
        status="completed",
        objective_values=(ObjectiveValue(objective_name="mean-R", value=score),),
        objective_scores=(
            ObjectiveScore(
                objective_name="mean-R",
                value=score,
                score=score,
                weighted_score=score,
            ),
        ),
        total_score=score,
        feasible=feasible,
        physics_accepted=True,
        certificate_id=f"certificate-{candidate_id[-8:]}",
        run_id=f"run-{candidate_id[-8:]}",
        task_sha256="a" * 64,
        material_catalog_sha256="b" * 64,
        artifact_root="C:/verified-artifacts",
    )


def _failed_record(candidate_id: str, space: DesignSpace) -> EvaluationRecord:
    return EvaluationRecord(
        candidate_id=candidate_id,
        design_space_id=space.design_space_id,
        objective_set_id=_objectives().objective_set_id,
        status="failed",
        failure_stage="managed_execution",
        physics_accepted=False,
        material_catalog_sha256="b" * 64,
        failures=({"code": "synthetic_failure", "message": "failed closed"},),
    )


def _batch_records(path: Path) -> tuple[EvaluationRecord, ...]:
    records: list[EvaluationRecord] = []
    for line in (path / "BATCH_INDEX.jsonl").read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        assert payload["schema_version"] == BATCH_INDEX_SCHEMA_VERSION
        records.append(
            EvaluationRecord.model_validate_json(
                json.dumps(payload["record"], ensure_ascii=False, allow_nan=False)
            )
        )
    return tuple(records)


def _dataset_record(
    candidate_id: str,
    *,
    index: int,
    accepted: bool,
) -> DatasetRecord:
    return DatasetRecord(
        dataset_id="dataset-test",
        plan_id="plan-test",
        candidate_id=candidate_id,
        sample_index=index,
        seed=17,
        design_variables={"thickness_nm": 100.0 + index},
        normalized_design=(0.25 + 0.1 * index,),
        task_sha256="c" * 64 if accepted else None,
        run_id=f"run-{index}" if accepted else None,
        material_catalog_sha256="d" * 64,
        material_identities=(
            DatasetMaterialIdentity(
                position="layer",
                layer_index=0,
                constant_n=2.0,
                constant_k=0.0,
                thickness_nm=100.0 + index,
            ),
        ),
        wavelength=DatasetWavelengthConfig(
            mode="linspace",
            start_nm=500.0,
            stop_nm=600.0,
            point_count=7,
            configuration_sha256="e" * 64,
        ),
        requested_outputs=("R", "T", "A"),
        selected_outputs=("R",),
        evaluation_status="completed" if accepted else "failed",
        verification_status="accepted" if accepted else "failed",
        physics_accepted=accepted,
        certificate_id=f"certificate-{index}" if accepted else None,
        veritmm_version=__version__,
        failure_codes=() if accepted else ("synthetic_failure",),
    )


def test_runtime_protocols_and_optional_imports_are_dependency_neutral(
    tmp_path: Path,
) -> None:
    space = _space()
    optimizer = RandomSearchAdapter(space, seed=7)
    environment = DesignSpaceEnvironment(space, _evaluator(tmp_path, space))

    assert isinstance(optimizer, OptimizerAdapter)
    assert isinstance(environment, ResearchEnvironment)

    code = (
        "import sys; import tmm_engine.research; "
        "assert 'torch' not in sys.modules; assert 'gymnasium' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_random_search_ask_tell_lifecycle_is_strict_and_deterministic() -> None:
    space = _space()
    first = RandomSearchAdapter(space, seed=42)
    second = RandomSearchAdapter(space, seed=42)

    first_round = first.ask(3)
    second_round = first.ask(2)
    assert first_round == second.ask(3)
    assert second_round == second.ask(2)
    assert len({item.candidate_id for item in (*first_round, *second_round)}) == 5

    record = _completed_record(first_round[0].candidate_id, space, score=0.4)
    first.tell((record,))
    assert first.best() is not None
    assert first.best().certificate_id == record.certificate_id
    with pytest.raises(ValueError, match="already been told"):
        first.tell((record,))
    with pytest.raises(ValueError, match="duplicate"):
        first.tell(
            (
                _completed_record(first_round[1].candidate_id, space, score=0.2),
                _completed_record(first_round[1].candidate_id, space, score=0.2),
            )
        )

    unknown = space.candidate({"thickness_nm": 100.0})
    with pytest.raises(ValueError, match="not returned by ask"):
        first.tell((_completed_record(unknown.candidate_id, space, score=0.9),))


def test_random_search_ranks_only_verified_and_prefers_feasibility() -> None:
    space = _space()
    optimizer = RandomSearchAdapter(space, seed=6)
    candidates = optimizer.ask(3)
    feasible = _completed_record(candidates[0].candidate_id, space, score=0.1)
    infeasible = _completed_record(
        candidates[1].candidate_id, space, score=100.0, feasible=False
    )
    forged_unverified = _failed_record(candidates[2].candidate_id, space).model_copy(
        update={
            "total_score": 1_000_000.0,
            "feasible": True,
            "certificate_id": "forged-certificate",
        }
    )

    optimizer.tell((infeasible, forged_unverified, feasible))
    best = optimizer.best()
    assert best is not None
    assert best.candidate_id == feasible.candidate_id
    assert best.physics_accepted is True
    assert best.certificate_id == feasible.certificate_id


def test_random_search_state_roundtrip_and_binding_validation() -> None:
    space = _space()
    optimizer = RandomSearchAdapter(space, seed=81, max_observations=20, max_pending=10)
    candidates = optimizer.ask(2)
    optimizer.tell((_completed_record(candidates[0].candidate_id, space, score=0.5),))
    state = optimizer.state_dict()
    encoded = json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    restored = RandomSearchAdapter(space, seed=81, max_observations=20, max_pending=10)
    restored.load_state_dict(json.loads(encoded))
    assert json.dumps(
        restored.state_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ) == encoded
    assert restored.best() == optimizer.best()
    assert restored.ask(1) == optimizer.ask(1)

    with pytest.raises(ValueError, match="seed mismatch"):
        RandomSearchAdapter(
            space, seed=82, max_observations=20, max_pending=10
        ).load_state_dict(state)
    with pytest.raises(ValueError, match="design-space binding mismatch"):
        RandomSearchAdapter(
            _space(shifted=True), seed=81, max_observations=20, max_pending=10
        ).load_state_dict(state)

    corrupt = json.loads(encoded)
    corrupt["best_candidate_id"] = candidates[1].candidate_id
    with pytest.raises(ValueError, match="best identity"):
        RandomSearchAdapter(
            space, seed=81, max_observations=20, max_pending=10
        ).load_state_dict(corrupt)


def test_real_ask_batch_tell_best_retains_certificate(tmp_path: Path) -> None:
    space = _space()
    evaluator = _evaluator(tmp_path, space)
    optimizer = RandomSearchAdapter(space, seed=24)
    candidates = optimizer.ask(2)
    request = BatchEvaluationRequest(
        design_space_id=space.design_space_id,
        objective_set_id=evaluator.objectives.objective_set_id,
        candidates=candidates,
    )
    batch_root = tmp_path / "optimizer-batch"

    result = evaluator.evaluate_many(request, resume=False, output_dir=batch_root)
    records = _batch_records(Path(result.artifact_root))
    optimizer.tell(records)
    best = optimizer.best()

    assert result.status == "completed"
    assert len(records) == 2
    assert best is not None
    assert best.candidate_id in {item.candidate_id for item in candidates}
    assert best.physics_accepted is True
    assert best.certificate_id
    assert best.run_id


@pytest.mark.requires_torch
def test_verified_torch_dataset_real_path_filters_and_reads_index(tmp_path: Path) -> None:
    import torch

    accepted = _dataset_record("candidate-accepted", index=0, accepted=True)
    failed = _dataset_record("candidate-failed", index=1, accepted=False)
    targets = {accepted.candidate_id: 0.75, failed.candidate_id: 999.0}

    dataset = VerifiedTorchDataset(
        (accepted, failed),
        targets=targets,
        target_name="mean-R",
        target_kind="objective_score",
    )
    assert len(dataset) == 1
    features, target = dataset[0]
    assert isinstance(features, torch.Tensor)
    assert isinstance(target, torch.Tensor)
    assert features.dtype == torch.float64
    assert features.tolist() == pytest.approx([0.25])
    assert target.tolist() == pytest.approx([0.75])
    assert dataset.candidate_ids == (accepted.candidate_id,)
    assert dataset.run_ids == (accepted.run_id,)
    assert dataset.task_sha256s == (accepted.task_sha256,)
    assert dataset.certificate_ids == (accepted.certificate_id,)

    index_path = tmp_path / "DATASET_INDEX.jsonl"
    rows = [
        {
            "schema_version": "veritmm-dataset-index-v1",
            "dataset_id": "dataset-test",
            "index": index,
            "candidate_id": record.candidate_id,
            "record": record.model_dump(mode="json"),
        }
        for index, record in enumerate((accepted, failed))
    ]
    index_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    restored = VerifiedTorchDataset(
        index_path,
        targets=targets,
        target_name="mean-R",
        target_kind="objective_value",
    )
    assert restored.candidate_ids == dataset.candidate_ids
    assert restored.features.tolist() == dataset.features.tolist()


def test_torch_missing_is_actionable_only_at_adapter_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tmm_engine.research.adapters as adapters

    original = adapters.importlib.import_module

    def missing(name: str):
        if name == "torch":
            raise ImportError("synthetic missing torch")
        return original(name)

    monkeypatch.setattr(adapters.importlib, "import_module", missing)
    accepted = _dataset_record("candidate-accepted", index=0, accepted=True)
    with pytest.raises(OptionalDependencyError, match="optional 'torch'"):
        VerifiedTorchDataset(
            (accepted,),
            targets={accepted.candidate_id: 0.5},
            target_name="mean-R",
            target_kind="objective_value",
        )


def test_environment_actions_state_and_unsupported_layer_count(tmp_path: Path) -> None:
    space = _space(mixed=True)
    environment = DesignSpaceEnvironment(space, _evaluator(tmp_path, space), seed=9)
    assert environment.reset(seed=12) == environment.reset(seed=12)
    initial = environment.state()

    add = environment.step(AddLayerAction())
    remove = environment.step(RemoveLayerAction(layer_index=0))
    assert add.status == remove.status == "unsupported"
    assert add.issue is not None and remove.issue is not None
    assert add.issue.code == remove.issue.code == "variable_layer_count_unsupported"
    assert environment.state().assignments == initial.assignments

    invalid = environment.step(
        ChooseMaterialAction(variable="material", choice="unknown")
    )
    assert invalid.status == "invalid"
    assert invalid.issue is not None
    assert invalid.issue.code == "invalid_material_choice"

    thickness = environment.step(
        ChooseThicknessAction(variable="thickness_nm", value_nm=100.0)
    )
    material = environment.step(
        ChooseMaterialAction(variable="material", choice="low")
    )
    assert thickness.status == material.status == "assigned"
    assert material.state.complete is True
    encoded = json.dumps(
        environment.state_dict(), sort_keys=True, separators=(",", ":")
    )
    restored = DesignSpaceEnvironment(space, _evaluator(tmp_path, space), seed=99)
    restored.load_state_dict(json.loads(encoded))
    assert restored.state() == environment.state()


def test_environment_stop_uses_real_verified_evaluator_and_reward_cannot_certify(
    tmp_path: Path,
) -> None:
    space = _space(mixed=True)
    environment = DesignSpaceEnvironment(space, _evaluator(tmp_path, space), seed=5)
    incomplete = environment.step(StopAction())
    assert incomplete.status == "invalid"
    assert incomplete.issue is not None and incomplete.issue.code == "incomplete_design"

    environment.step(ChooseThicknessAction(variable="thickness_nm", value_nm=101.0))
    environment.step(ChooseMaterialAction(variable="material", choice="high"))
    result = environment.step(StopAction())

    assert result.status == "evaluated"
    assert result.terminated is True
    assert result.reward is not None
    assert result.info["status"] == "completed"
    assert result.info["physics_accepted"] is True
    assert result.info["certificate_id"]
    assert result.info["run_id"]
    assert result.info["reward_is_not_physics_validity"] is True

    after_stop = environment.step(StopAction())
    assert after_stop.status == "invalid"
    assert after_stop.terminated is True
    assert after_stop.issue is not None and after_stop.issue.code == "episode_terminated"


def test_adapter_state_is_bounded_and_has_no_bulk_scientific_payloads(
    tmp_path: Path,
) -> None:
    space = _space(mixed=True)
    optimizer = RandomSearchAdapter(space, seed=1, max_pending=2)
    with pytest.raises(ValueError, match="max_pending"):
        RandomSearchAdapter(space, seed=1, max_pending=1_025)
    optimizer.ask(2)
    with pytest.raises(ValueError, match="bounded pending"):
        optimizer.ask(1)
    environment = DesignSpaceEnvironment(space, _evaluator(tmp_path, space))
    payloads = (
        optimizer.state_dict(),
        environment.state_dict(),
        environment.step(AddLayerAction()).model_dump(mode="json"),
    )
    forbidden = ("spectra", "history", "population", "trajectory", "wavelengths_nm")
    for payload in payloads:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        assert len(encoded.encode("utf-8")) < 32 * 1024
        assert all(name not in encoded for name in forbidden)
