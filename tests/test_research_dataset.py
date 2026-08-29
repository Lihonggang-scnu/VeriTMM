from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pytest
from pydantic import ValidationError

from tmm_engine import __version__
from tmm_engine.execution import ExecutionSettings
from tmm_engine.protocol.responses import COMPACT_MAX_BYTES, validate_artifact_references
from tmm_engine.research import (
    ConstraintSpec,
    ContinuousThicknessVariable,
    DatasetConfig,
    DatasetFactory,
    DatasetMaterialIdentity,
    DatasetRecord,
    DatasetWavelengthConfig,
    DesignSpace,
    DesignSpaceContract,
    DiscreteThicknessVariable,
    EvaluationRecord,
    EvaluatorConfig,
    MaterialChoiceVariable,
    MaterialOption,
    ObjectiveScore,
    ObjectiveSet,
    ObjectiveSpec,
    ObjectiveValue,
    ResearchEvaluator,
    SamplingPlan,
    build_dataset_result,
    sample_candidates,
)
from tmm_engine.schemas import (
    IlluminationSpec,
    LayerSpec,
    MediumSpec,
    SimulationTask,
    SpectralGrid,
    StackSpec,
)


def _mixed_space() -> DesignSpace:
    task = SimulationTask(
        stack=StackSpec(
            layers=(
                LayerSpec(
                    None,
                    100.0,
                    constant_n=1.6,
                    min_thickness_nm=80.0,
                    max_thickness_nm=120.0,
                ),
                LayerSpec(
                    None,
                    90.0,
                    constant_n=2.0,
                    min_thickness_nm=70.0,
                    max_thickness_nm=110.0,
                ),
                LayerSpec("sio2", 100.0, provider="builtin"),
                LayerSpec("sio2", 200.0, provider="builtin", optimizable=False),
            ),
            incident=MediumSpec(constant_n=1.0),
            exit=MediumSpec(constant_n=1.5),
        ),
        spectrum=SpectralGrid(start_nm=500.0, stop_nm=600.0, points=7),
        illumination=IlluminationSpec((0.0,), ("unpolarized",)),
    )
    return DesignSpace(
        DesignSpaceContract(
            base_task=task,
            variables=(
                ContinuousThicknessVariable(
                    name="continuous_nm",
                    layer_index=0,
                    lower_nm=80.0,
                    upper_nm=120.0,
                ),
                DiscreteThicknessVariable(
                    name="discrete_nm",
                    layer_index=1,
                    values_nm=(70.0, 90.0, 110.0),
                ),
                MaterialChoiceVariable(
                    name="material",
                    layer_index=2,
                    options=(
                        MaterialOption(
                            name="catalog",
                            material="sio2",
                            provider="builtin",
                            dataset_id="local",
                        ),
                        MaterialOption(
                            name="constant", constant_n=1.8, constant_k=0.01
                        ),
                    ),
                ),
            ),
        )
    )


def _small_space() -> DesignSpace:
    task = SimulationTask(
        stack=StackSpec(
            layers=(
                LayerSpec(
                    None,
                    100.0,
                    constant_n=2.0,
                    min_thickness_nm=90.0,
                    max_thickness_nm=110.0,
                ),
            ),
            incident=MediumSpec(constant_n=1.0),
            exit=MediumSpec(constant_n=1.5),
        ),
        spectrum=SpectralGrid(start_nm=500.0, stop_nm=600.0, points=7),
        illumination=IlluminationSpec((0.0,), ("unpolarized",)),
        requested_outputs=("R", "T", "A"),
    )
    return DesignSpace(
        DesignSpaceContract(
            base_task=task,
            variables=(
                ContinuousThicknessVariable(
                    name="thickness_nm",
                    layer_index=0,
                    lower_nm=90.0,
                    upper_nm=110.0,
                ),
            ),
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


def _evaluator(tmp_path: Path, space: DesignSpace | None = None) -> ResearchEvaluator:
    space = space or _small_space()
    return ResearchEvaluator(
        space,
        _objectives(),
        EvaluatorConfig(output_root=str(tmp_path / "runs"), cache=False),
        execution_settings=ExecutionSettings(
            write_plot=False,
            convergence_max_refinements=2,
        ),
    )


def test_sampling_plan_serialization_and_identity_are_deterministic() -> None:
    first = SamplingPlan(
        strategy="latin_hypercube",
        sample_count=8,
        seed=42,
        options={"centered": True},
    )
    restored = SamplingPlan.model_validate_json(first.canonical_json())
    rebuilt = SamplingPlan(
        strategy="latin_hypercube",
        sample_count=8,
        seed=42,
        options={"centered": True},
    )
    changed = SamplingPlan(
        strategy="latin_hypercube",
        sample_count=8,
        seed=43,
        options={"centered": True},
    )

    assert restored == first
    assert restored.canonical_json() == first.canonical_json()
    assert rebuilt.plan_id == first.plan_id
    assert changed.plan_id != first.plan_id
    with pytest.raises(ValidationError, match="grid_levels"):
        SamplingPlan(strategy="grid", sample_count=2)
    with pytest.raises(ValidationError, match="unsupported"):
        SamplingPlan(
            strategy="sobol",
            sample_count=2,
            options={"silently_randomize": True},
        )


@pytest.mark.parametrize(
    ("strategy", "kwargs"),
    [
        ("random", {}),
        (
            "grid",
            {
                "grid_levels": {
                    "continuous_nm": 2,
                    "discrete_nm": 2,
                    "material": 2,
                }
            },
        ),
        ("latin_hypercube", {}),
        ("sobol", {"options": {"skip": 3}}),
    ],
)
def test_all_sampling_strategies_are_reproducible_bounded_and_decode_choices(
    strategy: str, kwargs: dict[str, object]
) -> None:
    space = _mixed_space()
    plan = SamplingPlan(
        strategy=strategy,  # type: ignore[arg-type]
        sample_count=4,
        seed=19,
        **kwargs,
    )
    first = sample_candidates(space, plan)
    unrelated = sample_candidates(
        space,
        SamplingPlan(
            strategy="random", sample_count=3, seed=999
        ),
    )
    second = sample_candidates(space, plan)

    assert unrelated
    assert [item.canonical_json() for item in first] == [
        item.canonical_json() for item in second
    ]
    assert [item.sample_index for item in first] == list(range(4))
    assert len({item.candidate_id for item in first}) == 4
    for candidate in first:
        assert all(0 <= value <= 1 for value in candidate.normalized_design)
        assert candidate.values["discrete_nm"] in {70.0, 90.0, 110.0}
        assert candidate.values["material"] in {"catalog", "constant"}
        converted = space.to_simulation_task(candidate)
        assert converted.stack.layers[3].thickness_nm == 200.0


@pytest.mark.parametrize("strategy", ["random", "latin_hypercube", "sobol"])
def test_stochastic_strategy_seed_changes_candidate_order(strategy: str) -> None:
    space = _mixed_space()
    first = sample_candidates(
        space,
        SamplingPlan(strategy=strategy, sample_count=4, seed=1),  # type: ignore[arg-type]
    )
    second = sample_candidates(
        space,
        SamplingPlan(strategy=strategy, sample_count=4, seed=2),  # type: ignore[arg-type]
    )

    assert [item.candidate_id for item in first] != [
        item.candidate_id for item in second
    ]


def test_grid_is_stable_across_seed_and_rejects_invalid_or_unbounded_designs() -> None:
    space = _mixed_space()
    levels = {"continuous_nm": 2, "discrete_nm": 2, "material": 2}
    first = sample_candidates(
        space,
        SamplingPlan(
            strategy="grid", sample_count=4, seed=1, grid_levels=levels
        ),
    )
    second = sample_candidates(
        space,
        SamplingPlan(
            strategy="grid", sample_count=4, seed=999, grid_levels=levels
        ),
    )
    assert [item.candidate_id for item in first] == [
        item.candidate_id for item in second
    ]

    with pytest.raises(ValueError, match="fewer points"):
        sample_candidates(
            space,
            SamplingPlan(strategy="grid", sample_count=5, grid_levels=1),
        )
    with pytest.raises(ValueError, match="cardinality"):
        sample_candidates(
            space,
            SamplingPlan(
                strategy="grid",
                sample_count=2,
                grid_levels={
                    "continuous_nm": 2,
                    "discrete_nm": 4,
                    "material": 2,
                },
            ),
        )


def _read_dataset_records(root: Path) -> list[DatasetRecord]:
    rows = []
    for line in (root / "DATASET_INDEX.jsonl").read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        rows.append(
            DatasetRecord.model_validate_json(
                json.dumps(payload["record"], ensure_ascii=False)
            )
        )
    return rows


def _collect_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        keys = {str(key) for key in value}
        for item in value.values():
            keys.update(_collect_keys(item))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for item in value:
            keys.update(_collect_keys(item))
        return keys
    return set()


def test_small_real_dataset_is_certificate_bound_complete_and_artifact_backed(
    tmp_path: Path,
) -> None:
    evaluator = _evaluator(tmp_path)
    factory = DatasetFactory(evaluator.design_space, evaluator)
    plan = SamplingPlan(
        strategy="grid", sample_count=2, seed=7, grid_levels=2
    )
    root = tmp_path / "d"
    result = factory.generate(
        plan,
        DatasetConfig(output_root=str(root), cache=False),
    )

    assert result.status == "completed"
    assert result.accepted_count == result.record_count == 2
    assert result.failed_count == result.rejected_count == 0
    assert {item.kind for item in result.artifacts} == {
        "research_dataset_manifest",
        "research_dataset_index",
    }
    assert validate_artifact_references(
        [item.model_dump(mode="python") for item in result.artifacts], root=root
    )
    records = _read_dataset_records(root)
    assert [item.sample_index for item in records] == [0, 1]
    for record in records:
        assert record.verification_status == "accepted"
        assert record.physics_accepted is True
        assert record.certificate_id and record.run_id and record.task_sha256
        assert record.veritmm_version == __version__
        assert record.archive_schema_version == 2
        assert len(record.material_identities) == 3
        assert [item.position for item in record.material_identities] == [
            "incident",
            "layer",
            "exit",
        ]
        assert record.wavelength.mode == "linspace"
        assert record.wavelength.point_count == 7
        assert record.requested_outputs == ("R", "T", "A")
        assert record.selected_outputs == ("R", "T")
        assert record.provenance["sampling_plan_id"] == plan.plan_id
        assert validate_artifact_references(
            [item.model_dump(mode="python") for item in record.artifacts],
            root=record.artifact_root,
        )

    manifest_payload = json.loads(
        (root / "DATASET_MANIFEST.json").read_text(encoding="utf-8")
    )
    index_payloads = [
        json.loads(line)
        for line in (root / "DATASET_INDEX.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    result_payload = json.loads(result.canonical_json())
    forbidden = {"wavelengths_nm", "channels", "spectra", "history", "samples"}
    assert not forbidden & _collect_keys(manifest_payload)
    assert not forbidden & _collect_keys(index_payloads)
    assert not forbidden & _collect_keys(result_payload)
    assert "candidate_ids" not in manifest_payload


def _synthetic_evaluation(
    evaluator: ResearchEvaluator,
    candidate_id: str,
    output_root: Path,
    *,
    failed: bool,
) -> EvaluationRecord:
    if failed:
        return EvaluationRecord(
            candidate_id=candidate_id,
            design_space_id=evaluator.design_space.design_space_id,
            objective_set_id=evaluator.objectives.objective_set_id,
            status="failed",
            failure_stage="managed_execution",
            material_catalog_sha256="a" * 64,
            failures=({"code": "synthetic_unverified", "message": "isolated"},),
        )
    output_root.mkdir(parents=True, exist_ok=True)
    return EvaluationRecord(
        candidate_id=candidate_id,
        design_space_id=evaluator.design_space.design_space_id,
        objective_set_id=evaluator.objectives.objective_set_id,
        status="completed",
        objective_values=(ObjectiveValue(objective_name="mean-R", value=0.5),),
        objective_scores=(
            ObjectiveScore(
                objective_name="mean-R",
                value=0.5,
                score=0.5,
                weighted_score=0.5,
            ),
        ),
        total_score=0.5,
        feasible=True,
        physics_accepted=True,
        certificate_id=f"cert-{candidate_id}",
        run_id=f"run-{candidate_id}",
        task_sha256="b" * 64,
        material_catalog_sha256="a" * 64,
        artifact_root=str(output_root),
    )


class _DatasetExecutor:
    name = "dataset-test-executor"

    def __init__(self, *, fail_index: int | None = None) -> None:
        self.fail_index = fail_index
        self.seen: list[str] = []

    def execute(
        self,
        evaluator: ResearchEvaluator,
        candidates: tuple[object, ...],
        *,
        output_root: Path,
    ) -> Iterable[EvaluationRecord]:
        for index, candidate in enumerate(candidates):
            self.seen.append(candidate.candidate_id)
            yield _synthetic_evaluation(
                evaluator,
                candidate.candidate_id,
                output_root,
                failed=index == self.fail_index,
            )


def test_partial_failure_is_isolated_and_never_upgraded_to_verified(
    tmp_path: Path,
) -> None:
    evaluator = _evaluator(tmp_path)
    factory = DatasetFactory(evaluator.design_space, evaluator)
    plan = SamplingPlan(
        strategy="grid", sample_count=3, grid_levels=3
    )
    executor = _DatasetExecutor(fail_index=1)
    result = factory.generate(
        plan,
        DatasetConfig(output_root=str(tmp_path / "partial")),
        executor=executor,
    )
    records = _read_dataset_records(tmp_path / "partial")

    assert result.status == "partial"
    assert result.accepted_count == 2
    assert result.failed_count == 1
    failed = records[1]
    assert failed.evaluation_status == "failed"
    assert failed.verification_status == "failed"
    assert failed.physics_accepted is False
    assert failed.certificate_id is None
    assert failed.failure_codes == ("synthetic_unverified",)
    assert records[0].verification_status == "accepted"
    assert records[2].verification_status == "accepted"


class _InterruptingDatasetExecutor(_DatasetExecutor):
    name = "dataset-interrupting-executor"

    def execute(
        self,
        evaluator: ResearchEvaluator,
        candidates: tuple[object, ...],
        *,
        output_root: Path,
    ) -> Iterable[EvaluationRecord]:
        for index, candidate in enumerate(candidates):
            if index == 1:
                raise RuntimeError("dataset interruption")
            self.seen.append(candidate.candidate_id)
            yield _synthetic_evaluation(
                evaluator, candidate.candidate_id, output_root, failed=False
            )


def test_dataset_resume_skips_batch_ledger_and_rejects_corruption(
    tmp_path: Path,
) -> None:
    evaluator = _evaluator(tmp_path)
    factory = DatasetFactory(evaluator.design_space, evaluator)
    plan = SamplingPlan(strategy="grid", sample_count=3, grid_levels=3)
    root = tmp_path / "resume"
    interrupted = _InterruptingDatasetExecutor()
    with pytest.raises(RuntimeError, match="dataset interruption"):
        factory.generate(
            plan,
            DatasetConfig(output_root=str(root), resume=True),
            executor=interrupted,
        )
    assert len(interrupted.seen) == 1

    resumed = _DatasetExecutor()
    result = factory.generate(
        plan,
        DatasetConfig(output_root=str(root), resume=True),
        executor=resumed,
    )
    assert result.status == "completed"
    expected = sample_candidates(evaluator.design_space, plan)
    assert resumed.seen == [item.candidate_id for item in expected[1:]]

    with (root / "DATASET_INDEX.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{corrupt\n")
    with pytest.raises(ValueError, match="stale|corrupt"):
        factory.generate(
            plan,
            DatasetConfig(output_root=str(root), resume=True),
            executor=_DatasetExecutor(),
        )


def test_dataset_resume_rejects_manifest_binding_mismatch(tmp_path: Path) -> None:
    evaluator = _evaluator(tmp_path)
    factory = DatasetFactory(evaluator.design_space, evaluator)
    plan = SamplingPlan(strategy="grid", sample_count=2, grid_levels=2)
    root = tmp_path / "mismatch"
    factory.generate(
        plan,
        DatasetConfig(output_root=str(root)),
        executor=_DatasetExecutor(),
    )
    manifest_path = root / "DATASET_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["objective_set_id"] = "objective_set_mismatch"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="binding mismatch"):
        factory.generate(
            plan,
            DatasetConfig(output_root=str(root), resume=True),
            executor=_DatasetExecutor(),
        )


def _synthetic_dataset_record(index: int, root: Path) -> DatasetRecord:
    return DatasetRecord(
        dataset_id="dataset_scale",
        plan_id="sampling_plan_scale",
        candidate_id=f"candidate_{index:05d}",
        sample_index=index,
        seed=1,
        design_variables={"thickness_nm": 100.0},
        normalized_design=(0.5,),
        task_sha256=None,
        run_id=None,
        material_catalog_sha256="a" * 64,
        material_identities=(
            DatasetMaterialIdentity(
                position="layer",
                layer_index=0,
                constant_n=2.0,
                constant_k=0.0,
                thickness_nm=100.0,
            ),
        ),
        wavelength=DatasetWavelengthConfig(
            mode="linspace",
            start_nm=500.0,
            stop_nm=600.0,
            point_count=7,
            configuration_sha256="b" * 64,
        ),
        requested_outputs=("R", "T", "A"),
        selected_outputs=("R",),
        evaluation_status="failed",
        verification_status="failed",
        physics_accepted=False,
        certificate_id=None,
        veritmm_version=__version__,
        artifact_root=None,
        provenance={"synthetic": True},
        failure_codes=("synthetic",),
    )


def test_dataset_record_load_marks_legacy_version_without_rewriting_history(
    tmp_path: Path,
) -> None:
    current = _synthetic_dataset_record(0, tmp_path)
    assert current.version_identity_status == "verified"

    legacy_payload = current.model_dump(mode="python")
    legacy_payload["veritmm_version"] = "0.6.0"
    legacy_payload.pop("version_identity_status")

    loaded = DatasetRecord.model_validate(legacy_payload)

    assert loaded.veritmm_version == "0.6.0"
    assert loaded.version_identity_status == "legacy_inconsistent"
    assert loaded.model_dump(mode="json")["version_identity_status"] == (
        "legacy_inconsistent"
    )


def test_dataset_compact_response_is_constant_size_for_ten_thousand_records(
    tmp_path: Path,
) -> None:
    records = tuple(_synthetic_dataset_record(index, tmp_path) for index in range(10_000))
    small = build_dataset_result(
        dataset_id="dataset_scale",
        plan_id="sampling_plan_scale",
        records=records[:10],
        artifact_root=tmp_path,
    )
    large = build_dataset_result(
        dataset_id="dataset_scale",
        plan_id="sampling_plan_scale",
        records=records,
        artifact_root=tmp_path,
    )
    small_json = small.canonical_json()
    large_json = large.canonical_json()
    small_size = len(small_json.encode("utf-8"))
    large_size = len(large_json.encode("utf-8"))

    assert small_size <= COMPACT_MAX_BYTES
    assert large_size <= COMPACT_MAX_BYTES
    assert large_size - small_size < 1_000
    assert large.preview_count == 12
    assert large.truncated_count == 9_988
    forbidden = {"wavelengths_nm", "channels", "spectra", "history", "samples"}
    assert not forbidden & _collect_keys(json.loads(large_json))


def test_sampling_core_has_no_scipy_or_heavy_dependency() -> None:
    from tmm_engine.research import sampling

    source = Path(sampling.__file__).read_text(encoding="utf-8").casefold()
    assert "scipy" not in source
    assert "torch" not in source
    assert "sklearn" not in source
