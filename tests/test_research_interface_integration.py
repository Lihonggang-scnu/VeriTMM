from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pytest

from tmm_engine import __version__
from tmm_engine.cli import main as cli_main
from tmm_engine.execution import ExecutionSettings
from tmm_engine.experiment_store import ExperimentStore
from tmm_engine.managed_execution import execute_managed_task
from tmm_engine.protocol import describe_capabilities
from tmm_engine.protocol.responses import COMPACT_MAX_BYTES, validate_artifact_references
from tmm_engine.research import (
    AddLayerAction,
    BatchEvaluationRequest,
    ConstraintSpec,
    ContinuousThicknessVariable,
    DatasetConfig,
    DatasetFactory,
    DatasetRecord,
    DesignCandidate,
    DesignSpace,
    DesignSpaceContract,
    DesignSpaceEnvironment,
    DiscreteThicknessVariable,
    EvaluationRecord,
    EvaluatorConfig,
    MaterialChoiceVariable,
    MaterialOption,
    ObjectiveSet,
    ObjectiveSpec,
    OptimizerAdapter,
    RandomSearchAdapter,
    ResearchEnvironment,
    ResearchEvaluator,
    SamplingPlan,
    StopAction,
    VerifiedTorchDataset,
    sample_candidates,
)
from tmm_engine.schemas import (
    IlluminationSpec,
    LayerSpec,
    MediumSpec,
    PhysicsRequirements,
    SimulationTask,
    SpectralGrid,
    StackSpec,
)


def _task(*, supported: bool = True) -> SimulationTask:
    return SimulationTask(
        stack=StackSpec(
            layers=(
                LayerSpec(
                    None,
                    100.0,
                    constant_n=2.0,
                    min_thickness_nm=90.0,
                    max_thickness_nm=110.0,
                ),
                LayerSpec(None, 180.0, constant_n=1.45, optimizable=False),
            ),
            incident=MediumSpec(constant_n=1.0),
            exit=MediumSpec(constant_n=1.5),
        ),
        spectrum=SpectralGrid(start_nm=500.0, stop_nm=600.0, points=7),
        illumination=IlluminationSpec((0.0,), ("unpolarized",)),
        requested_outputs=("R", "T", "A"),
        physics=PhysicsRequirements(
            geometry_class="layered_planar" if supported else "lateral_periodic"
        ),
    )


def _space(*, supported: bool = True) -> DesignSpace:
    return DesignSpace(
        DesignSpaceContract(
            base_task=_task(supported=supported),
            variables=(
                ContinuousThicknessVariable(
                    name="thickness_nm",
                    layer_index=0,
                    lower_nm=90.0,
                    upper_nm=110.0,
                ),
            ),
            metadata={"purpose": "v0.6 integration"},
        )
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
        ),
        constraints=(
            ConstraintSpec(
                name="minimum-T",
                relation="at_least",
                observable="T",
                wavelength_min_nm=500.0,
                wavelength_max_nm=600.0,
                aggregation="min",
                threshold=0.0,
            ),
        ),
    )


def _settings() -> ExecutionSettings:
    return ExecutionSettings(write_plot=False, convergence_max_refinements=2)


def _evaluator(
    tmp_path: Path,
    space: DesignSpace,
    *,
    cache: bool = False,
    store: ExperimentStore | None = None,
) -> ResearchEvaluator:
    return ResearchEvaluator(
        space,
        _objectives(),
        EvaluatorConfig(
            output_root=str(tmp_path / "runs"),
            cache=cache,
            experiment_id="research-interface-integration",
        ),
        execution_settings=_settings(),
        store=store,
    )


def _evaluation_records(root: Path) -> tuple[EvaluationRecord, ...]:
    records: list[EvaluationRecord] = []
    for line in (root / "BATCH_INDEX.jsonl").read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        records.append(
            EvaluationRecord.model_validate_json(
                json.dumps(payload["record"], ensure_ascii=False, allow_nan=False)
            )
        )
    return tuple(records)


def _dataset_records(root: Path) -> tuple[DatasetRecord, ...]:
    records: list[DatasetRecord] = []
    for line in (root / "DATASET_INDEX.jsonl").read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        records.append(
            DatasetRecord.model_validate_json(
                json.dumps(payload["record"], ensure_ascii=False, allow_nan=False)
            )
        )
    return tuple(records)


def _keys(value: object) -> set[str]:
    if isinstance(value, dict):
        result = {str(key) for key in value}
        for item in value.values():
            result.update(_keys(item))
        return result
    if isinstance(value, (list, tuple)):
        result: set[str] = set()
        for item in value:
            result.update(_keys(item))
        return result
    return set()


def test_serialized_design_space_and_plan_roundtrip_to_legal_tasks() -> None:
    space = _space()
    restored_contract = DesignSpaceContract.model_validate_json(
        space.contract.canonical_json()
    )
    plan = SamplingPlan(strategy="sobol", sample_count=3, seed=19)
    restored_plan = SamplingPlan.model_validate_json(plan.canonical_json())

    candidates = sample_candidates(DesignSpace(restored_contract), restored_plan)

    assert restored_contract.design_space_id == space.design_space_id
    assert restored_plan == plan
    assert len(candidates) == 3
    for candidate in candidates:
        task = space.to_simulation_task(candidate)
        task.validate()
        assert len(task.stack.layers) == 2
        assert task.stack.layers[1] == _task().stack.layers[1]


def test_supported_end_to_end_dataset_batch_optimizer_is_certificate_bound(
    tmp_path: Path,
) -> None:
    space = _space()
    store = ExperimentStore(tmp_path / "store")
    evaluator = _evaluator(tmp_path, space, cache=True, store=store)
    optimizer = RandomSearchAdapter(space, seed=31)
    candidates = optimizer.ask(2)
    request = BatchEvaluationRequest(
        design_space_id=space.design_space_id,
        objective_set_id=evaluator.objectives.objective_set_id,
        candidates=candidates,
    )
    batch = evaluator.evaluate_many(
        request, resume=False, output_dir=tmp_path / "batch"
    )
    evaluations = _evaluation_records(Path(batch.artifact_root))

    plan = SamplingPlan(strategy="random", sample_count=2, seed=31)
    dataset_root = tmp_path / "dataset"
    dataset = DatasetFactory(space, evaluator).generate(
        plan,
        DatasetConfig(output_root=str(dataset_root), cache=True),
    )
    rows = _dataset_records(dataset_root)
    optimizer.tell(evaluations)
    best = optimizer.best()

    assert batch.status == dataset.status == "completed"
    assert [item.candidate_id for item in candidates] == [
        item.candidate_id for item in rows
    ]
    assert all(item.physics_accepted and item.certificate_id for item in evaluations)
    assert all(
        item.verification_status == "accepted"
        and item.physics_accepted
        and item.certificate_id
        and item.run_id
        and item.task_sha256
        for item in rows
    )
    assert all(item.provenance["cache_hit"] is True for item in rows)
    assert best is not None and best.physics_accepted and best.certificate_id
    for item in evaluations:
        assert item.artifact_root is not None
        assert validate_artifact_references(
            [ref.model_dump(mode="python") for ref in item.artifacts],
            root=item.artifact_root,
        )
    for item in rows:
        assert item.artifact_root is not None
        assert validate_artifact_references(
            [ref.model_dump(mode="python") for ref in item.artifacts],
            root=item.artifact_root,
        )
    assert validate_artifact_references(
        [item.model_dump(mode="python") for item in batch.artifacts],
        root=batch.artifact_root,
    )
    assert validate_artifact_references(
        [item.model_dump(mode="python") for item in dataset.artifacts],
        root=dataset.artifact_root,
    )

    forbidden = {
        "spectra",
        "wavelengths_nm",
        "channels",
        "population",
        "history",
        "trajectory",
    }
    assert not forbidden & _keys(batch.model_dump(mode="json"))
    assert not forbidden & _keys(dataset.model_dump(mode="json"))


def test_unsupported_physics_cannot_become_best_or_verified_dataset_row(
    tmp_path: Path,
) -> None:
    space = _space(supported=False)
    evaluator = _evaluator(tmp_path, space)
    optimizer = RandomSearchAdapter(space, seed=8)
    candidate = optimizer.ask(1)[0]
    evaluation = evaluator.evaluate(candidate, cache=False)
    optimizer.tell((evaluation,))

    dataset_root = tmp_path / "unsupported-dataset"
    result = DatasetFactory(space, evaluator).generate(
        SamplingPlan(strategy="random", sample_count=1, seed=8),
        DatasetConfig(output_root=str(dataset_root), cache=False),
    )
    row = _dataset_records(dataset_root)[0]

    assert evaluation.status == "failed"
    assert evaluation.physics_accepted is False
    assert evaluation.objective_values == ()
    assert optimizer.best() is None
    assert result.status == "failed"
    assert row.verification_status in {"rejected", "failed"}
    assert row.physics_accepted is False
    assert row.certificate_id is None


class _PartiallyFailingExecutor:
    name = "integration-partial"

    def execute(
        self,
        evaluator: ResearchEvaluator,
        candidates: tuple[DesignCandidate, ...],
        *,
        output_root: Path,
    ) -> Iterable[EvaluationRecord]:
        for index, candidate in enumerate(candidates):
            if index == 1:
                yield EvaluationRecord(
                    candidate_id=candidate.candidate_id,
                    design_space_id=evaluator.design_space.design_space_id,
                    objective_set_id=evaluator.objectives.objective_set_id,
                    status="failed",
                    failure_stage="managed_execution",
                    material_catalog_sha256="a" * 64,
                    failures=(
                        {
                            "code": "integration_isolated_failure",
                            "message": "one candidate failed closed",
                        },
                    ),
                )
            else:
                yield evaluator.evaluate(
                    candidate, cache=False, output_root=output_root
                )


def test_partial_batch_and_dataset_failure_is_isolated(tmp_path: Path) -> None:
    space = _space()
    evaluator = _evaluator(tmp_path, space)
    root = tmp_path / "partial-dataset"
    result = DatasetFactory(space, evaluator).generate(
        SamplingPlan(strategy="grid", sample_count=2, grid_levels=2),
        DatasetConfig(output_root=str(root), cache=False),
        executor=_PartiallyFailingExecutor(),
    )
    records = _dataset_records(root)
    batch_manifest = json.loads(
        (root / "evaluations" / "BATCH_MANIFEST.json").read_text(encoding="utf-8")
    )

    assert result.status == "partial"
    assert batch_manifest["status"] == "partial"
    assert [item.verification_status for item in records] == ["accepted", "failed"]
    assert records[0].certificate_id
    assert records[1].certificate_id is None
    assert records[1].physics_accepted is False


def test_capability_manifest_and_cli_report_v06_research_limits(
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = describe_capabilities()
    research = manifest.research_interface

    assert __version__ == manifest.package_version == research.version
    assert research.dataset.sampling_strategies == (
        "random",
        "grid",
        "latin_hypercube",
        "sobol",
    )
    assert research.dataset.sobol_core_max_dimension == 16
    assert research.objectives.score_confers_physics_validity is False
    assert research.evaluation.independent_certificate_required is True
    assert research.design_space.variable_layer_count is False
    assert research.adapters.gymnasium_required is False
    assert research.adapters.reserved_actions == ("add_layer", "remove_layer")

    assert cli_main(["describe", "--json"]) == 0
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)
    assert len(stdout.encode("utf-8")) <= COMPACT_MAX_BYTES
    assert payload["research_interface"]["version"] == __version__
    assert payload["research_interface"]["dataset"]["sobol_core_max_dimension"] == 16
    assert payload["response"]["profile"] == "compact"


def test_public_research_imports_are_lazy_and_dependency_neutral(tmp_path: Path) -> None:
    space = _space()
    evaluator = _evaluator(tmp_path, space)

    assert isinstance(RandomSearchAdapter(space, seed=1), OptimizerAdapter)
    assert isinstance(DesignSpaceEnvironment(space, evaluator), ResearchEnvironment)
    assert all(
        item is not None
        for item in (
            DesignSpaceContract,
            DesignCandidate,
            ContinuousThicknessVariable,
            DiscreteThicknessVariable,
            MaterialChoiceVariable,
            MaterialOption,
            ObjectiveSpec,
            ConstraintSpec,
            ObjectiveSet,
            ResearchEvaluator,
            BatchEvaluationRequest,
            DatasetFactory,
            SamplingPlan,
            RandomSearchAdapter,
            VerifiedTorchDataset,
            DesignSpaceEnvironment,
            AddLayerAction,
            StopAction,
        )
    )

    code = (
        "import sys; from tmm_engine import research; "
        "assert research.VerifiedTorchDataset; "
        "assert 'torch' not in sys.modules; assert 'gymnasium' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_evaluator_matches_direct_certified_simulation_without_numerical_regression(
    tmp_path: Path,
) -> None:
    space = _space()
    evaluator = _evaluator(tmp_path, space)
    candidate = space.candidate({"thickness_nm": 101.0})
    task = space.to_simulation_task(candidate)
    direct_root = tmp_path / "direct"
    envelope = execute_managed_task(
        "simulate",
        task,
        direct_root,
        execution_settings=_settings(),
        cache=False,
        detail="compact",
    )
    assert envelope["ok"] is True and envelope["certificate_id"]
    assert validate_artifact_references(envelope["artifacts"], root=direct_root)
    simulation_ref = next(
        item for item in envelope["artifacts"] if item["kind"] == "simulation_result"
    )
    simulation = json.loads(
        (direct_root / simulation_ref["path"]).read_text(encoding="utf-8")
    )
    expected = float(
        np.mean(simulation["channels"]["angle=0|pol=unpolarized"]["R"])
    )

    evaluated = evaluator.evaluate(candidate, cache=False)
    actual = next(
        item.value
        for item in evaluated.objective_values
        if item.objective_name == "mean-R"
    )
    assert evaluated.physics_accepted is True and evaluated.certificate_id
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_missing_certificate_artifact_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tmm_engine.research import evaluator as evaluator_module

    original = evaluator_module.execute_managed_task

    def remove_certificate(*args: object, **kwargs: object) -> dict[str, object]:
        envelope = original(*args, **kwargs)
        output_root = Path(args[2])
        certificate = next(
            item
            for item in envelope["artifacts"]
            if item["kind"] == "physics_certificate"
        )
        (output_root / certificate["path"]).unlink()
        return envelope

    monkeypatch.setattr(evaluator_module, "execute_managed_task", remove_certificate)
    space = _space()
    record = _evaluator(tmp_path, space).evaluate(
        space.candidate({"thickness_nm": 100.0}), cache=False
    )

    assert record.status == "failed"
    assert record.failure_stage == "artifact_integrity"
    assert record.physics_accepted is False
    assert record.certificate_id is None
    assert record.objective_values == ()
    assert record.total_score is None
