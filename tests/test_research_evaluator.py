from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pytest

from tmm_engine.execution import ExecutionSettings
from tmm_engine.experiment_store import ExperimentStore
from tmm_engine.protocol.responses import (
    COMPACT_MAX_BYTES,
    validate_artifact_references,
)
from tmm_engine.research import (
    BatchEvaluationRequest,
    ConstraintSpec,
    ContinuousThicknessVariable,
    DesignSpace,
    DesignSpaceContract,
    EvaluationRecord,
    EvaluatorConfig,
    ObjectiveScore,
    ObjectiveSet,
    ObjectiveSpec,
    ObjectiveValue,
    ResearchEvaluator,
    build_batch_result,
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
                    min_thickness_nm=80.0,
                    max_thickness_nm=120.0,
                ),
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
                    lower_nm=80.0,
                    upper_nm=120.0,
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
                weight=2.0,
            ),
            ObjectiveSpec(
                name="min-T",
                direction="minimize",
                observable="T",
                wavelength_min_nm=500.0,
                wavelength_max_nm=600.0,
                aggregation="min",
                weight=0.5,
            ),
            ObjectiveSpec(
                name="target-max-A",
                direction="target",
                observable="A",
                wavelength_min_nm=500.0,
                wavelength_max_nm=600.0,
                aggregation="max",
                weight=1.5,
                target=0.0,
            ),
        ),
        constraints=(
            ConstraintSpec(
                name="T-floor",
                relation="at_least",
                observable="T",
                wavelength_min_nm=500.0,
                wavelength_max_nm=600.0,
                aggregation="min",
                threshold=0.0,
                tolerance=0.0,
            ),
            ConstraintSpec(
                name="A-ceiling",
                relation="at_most",
                observable="A",
                wavelength_min_nm=500.0,
                wavelength_max_nm=600.0,
                aggregation="max",
                threshold=1.0,
                tolerance=0.0,
            ),
        ),
    )


def _settings() -> ExecutionSettings:
    return ExecutionSettings(
        write_plot=False,
        convergence_max_refinements=2,
    )


def _evaluator(
    tmp_path: Path,
    *,
    supported: bool = True,
    store: ExperimentStore | None = None,
) -> ResearchEvaluator:
    return ResearchEvaluator(
        _space(supported=supported),
        _objectives(),
        EvaluatorConfig(
            output_root=str(tmp_path / "research"),
            experiment_id="exp_research",
            cache=True,
            tags=("research",),
        ),
        execution_settings=_settings(),
        store=store,
    )


def _candidate(evaluator: ResearchEvaluator, thickness: float = 100.0):
    return evaluator.design_space.candidate({"thickness_nm": thickness})


def _artifact_payload(record: EvaluationRecord, kind: str) -> dict[str, object]:
    ref = next(item for item in record.artifacts if item.kind == kind)
    assert record.artifact_root is not None
    return json.loads(
        (Path(record.artifact_root) / ref.path).read_text(encoding="utf-8")
    )


def test_real_evaluation_is_certified_compact_and_aggregates_correctly(
    tmp_path: Path,
) -> None:
    evaluator = _evaluator(tmp_path)
    candidate = _candidate(evaluator)
    record = evaluator.evaluate(candidate, cache=False)

    assert record.status == "completed"
    assert record.candidate_id == candidate.candidate_id
    assert record.physics_accepted is True
    assert record.certificate_id
    assert record.run_id and record.task_sha256
    assert record.response_profile == "compact"
    assert record.artifact_root is not None
    assert validate_artifact_references(
        [item.model_dump(mode="python") for item in record.artifacts],
        root=record.artifact_root,
    )
    certificate = _artifact_payload(record, "physics_certificate")
    assert certificate["accepted"] is True
    assert certificate["certificate_id"] == record.certificate_id

    simulation = _artifact_payload(record, "simulation_result")
    wavelengths = np.asarray(simulation["wavelengths_nm"], dtype=float)
    channel = simulation["channels"]["angle=0|pol=unpolarized"]
    mask = (wavelengths >= 500.0) & (wavelengths <= 600.0)
    expected = {
        "mean-R": float(np.mean(np.asarray(channel["R"])[mask])),
        "min-T": float(np.min(np.asarray(channel["T"])[mask])),
        "target-max-A": float(np.max(np.asarray(channel["A"])[mask])),
    }
    actual = {item.objective_name: item.value for item in record.objective_values}
    assert actual == pytest.approx(expected)
    scores = {item.objective_name: item for item in record.objective_scores}
    assert scores["mean-R"].score == pytest.approx(expected["mean-R"])
    assert scores["min-T"].score == pytest.approx(-expected["min-T"])
    assert scores["target-max-A"].score == pytest.approx(
        -abs(expected["target-max-A"])
    )
    assert record.total_score == pytest.approx(
        2.0 * expected["mean-R"]
        - 0.5 * expected["min-T"]
        - 1.5 * abs(expected["target-max-A"])
    )
    assert record.feasible is True
    assert all(item.satisfied for item in record.constraint_statuses)
    encoded = record.canonical_json().casefold()
    assert "wavelengths_nm" not in encoded
    assert '"channels"' not in encoded


def test_real_unsupported_physics_fails_closed_without_objectives(tmp_path: Path) -> None:
    evaluator = _evaluator(tmp_path, supported=False)
    record = evaluator.evaluate(_candidate(evaluator), cache=False)

    assert record.status == "failed"
    assert record.failure_stage == "managed_execution"
    assert record.physics_accepted is False
    assert record.objective_values == ()
    assert record.objective_scores == ()
    assert record.total_score is None
    assert record.feasible is None
    assert record.run_id is not None
    assert any(item.get("code") == "unsupported_geometry" for item in record.failures)


def test_missing_certificate_artifact_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tmm_engine.research import evaluator as evaluator_module

    real_execute = evaluator_module.execute_managed_task

    def remove_certificate(*args: object, **kwargs: object) -> dict[str, object]:
        envelope = real_execute(*args, **kwargs)
        output = Path(args[2])
        (output / "PHYSICS_ACCEPTANCE_CERTIFICATE.json").unlink()
        return envelope

    monkeypatch.setattr(evaluator_module, "execute_managed_task", remove_certificate)
    evaluator = _evaluator(tmp_path)
    record = evaluator.evaluate(_candidate(evaluator), cache=False)

    assert record.status == "failed"
    assert record.failure_stage == "artifact_integrity"
    assert record.physics_accepted is False
    assert record.objective_values == ()


def test_certificate_id_mismatch_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tmm_engine.research import evaluator as evaluator_module

    real_execute = evaluator_module.execute_managed_task

    def mismatch_certificate(*args: object, **kwargs: object) -> dict[str, object]:
        envelope = dict(real_execute(*args, **kwargs))
        envelope["certificate_id"] = "forged-certificate"
        return envelope

    monkeypatch.setattr(evaluator_module, "execute_managed_task", mismatch_certificate)
    evaluator = _evaluator(tmp_path)
    record = evaluator.evaluate(_candidate(evaluator), cache=False)

    assert record.status == "failed"
    assert record.failure_stage == "certificate_validation"
    assert record.physics_accepted is False
    assert record.objective_values == ()
    assert any("does not match" in str(item["message"]) for item in record.failures)


def test_cache_reuse_has_fresh_run_identity_and_source_provenance(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / ".veritmm")
    evaluator = _evaluator(tmp_path, store=store)
    candidate = _candidate(evaluator)
    first = evaluator.evaluate(candidate, cache=True)
    second = evaluator.evaluate(candidate, cache=True)

    assert first.status == second.status == "completed"
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.run_id != first.run_id
    assert second.source_run_id == first.run_id
    assert second.task_sha256 == first.task_sha256
    assert second.certificate_id == first.certificate_id
    assert second.artifact_provenance == {
        "mode": "cache_copy",
        "source_run_id": first.run_id,
    }
    assert first.artifact_root != second.artifact_root
    assert validate_artifact_references(
        [item.model_dump(mode="python") for item in second.artifacts],
        root=second.artifact_root,
    )


def test_invalid_candidate_identity_fails_before_managed_execution(tmp_path: Path) -> None:
    evaluator = _evaluator(tmp_path)
    forged = _candidate(evaluator).model_copy(update={"candidate_id": "candidate_forged"})
    record = evaluator.evaluate(forged, cache=False)

    assert record.status == "failed"
    assert record.failure_stage == "candidate_validation"
    assert record.run_id is None
    assert record.artifacts == ()
    assert record.objective_values == ()


def _synthetic_record(
    evaluator: ResearchEvaluator,
    candidate_id: str,
    artifact_root: Path,
    *,
    failed: bool = False,
) -> EvaluationRecord:
    common = {
        "candidate_id": candidate_id,
        "design_space_id": evaluator.design_space.design_space_id,
        "objective_set_id": evaluator.objectives.objective_set_id,
        "material_catalog_sha256": "a" * 64,
    }
    if failed:
        return EvaluationRecord(
            **common,
            status="failed",
            failure_stage="managed_execution",
            failures=({"code": "synthetic_failure", "message": "isolated"},),
        )
    return EvaluationRecord(
        **common,
        status="completed",
        objective_values=(ObjectiveValue(objective_name="mean-R", value=0.5),),
        objective_scores=(
            ObjectiveScore(
                objective_name="mean-R",
                value=0.5,
                score=0.5,
                weighted_score=1.0,
            ),
        ),
        total_score=1.0,
        feasible=True,
        physics_accepted=True,
        certificate_id=f"cert-{candidate_id}",
        run_id=f"run-{candidate_id}",
        task_sha256="b" * 64,
        artifact_root=str(artifact_root),
    )


class _ReplaceableExecutor:
    name = "test-replaceable"

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
        output_root.mkdir(parents=True, exist_ok=True)
        for index, candidate in enumerate(candidates):
            candidate_id = candidate.candidate_id
            self.seen.append(candidate_id)
            yield _synthetic_record(
                evaluator,
                candidate_id,
                output_root,
                failed=index == self.fail_index,
            )


def _batch_request(evaluator: ResearchEvaluator, count: int = 3) -> BatchEvaluationRequest:
    candidates = tuple(
        _candidate(evaluator, 90.0 + index * 5.0) for index in range(count)
    )
    return BatchEvaluationRequest(
        design_space_id=evaluator.design_space.design_space_id,
        objective_set_id=evaluator.objectives.objective_set_id,
        candidates=candidates,
    )


def test_replaceable_batch_executor_preserves_order_and_isolates_partial_failure(
    tmp_path: Path,
) -> None:
    evaluator = _evaluator(tmp_path)
    request = _batch_request(evaluator)
    executor = _ReplaceableExecutor(fail_index=1)
    result = evaluator.evaluate_many(request, executor=executor)

    assert result.status == "partial"
    assert result.completed_count == 2
    assert result.failed_count == 1
    assert executor.seen == [item.candidate_id for item in request.candidates]
    assert [item.candidate_id for item in result.preview] == executor.seen
    assert {item.kind for item in result.artifacts} == {
        "research_batch_manifest",
        "research_batch_index",
    }
    assert validate_artifact_references(
        [item.model_dump(mode="python") for item in result.artifacts],
        root=result.artifact_root,
    )
    lines = (Path(result.artifact_root) / "BATCH_INDEX.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert [json.loads(line)["candidate_id"] for line in lines] == executor.seen


def test_reference_sequential_batch_keeps_real_run_artifacts_in_evaluation_subdir(
    tmp_path: Path,
) -> None:
    evaluator = _evaluator(tmp_path)
    request = _batch_request(evaluator, count=1)
    result = evaluator.evaluate_many(request, output_dir=tmp_path / "real-batch")

    assert result.status == "completed"
    assert result.executor == "sequential"
    assert result.preview[0].run_id
    evaluation_roots = list((Path(result.artifact_root) / "evaluations").iterdir())
    assert len(evaluation_roots) == 1
    assert (evaluation_roots[0] / "RUN_RESULT.json").is_file()
    assert (evaluation_roots[0] / "SIMULATION_RESULT.json").is_file()
    assert (evaluation_roots[0] / "PHYSICS_ACCEPTANCE_CERTIFICATE.json").is_file()


class _InterruptingExecutor(_ReplaceableExecutor):
    name = "test-interrupting"

    def execute(
        self,
        evaluator: ResearchEvaluator,
        candidates: tuple[object, ...],
        *,
        output_root: Path,
    ) -> Iterable[EvaluationRecord]:
        for index, candidate in enumerate(candidates):
            if index == 1:
                raise RuntimeError("synthetic interruption")
            self.seen.append(candidate.candidate_id)
            yield _synthetic_record(
                evaluator, candidate.candidate_id, output_root
            )


def test_batch_resume_skips_completed_ledger_records(tmp_path: Path) -> None:
    evaluator = _evaluator(tmp_path)
    request = _batch_request(evaluator)
    root = tmp_path / "resume-batch"
    interrupted = _InterruptingExecutor()
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        evaluator.evaluate_many(
            request, executor=interrupted, output_dir=root, resume=True
        )
    assert interrupted.seen == [request.candidates[0].candidate_id]

    resumed = _ReplaceableExecutor()
    result = evaluator.evaluate_many(
        request, executor=resumed, output_dir=root, resume=True
    )
    assert result.status == "completed"
    assert resumed.seen == [item.candidate_id for item in request.candidates[1:]]
    assert [item.candidate_id for item in result.preview] == [
        item.candidate_id for item in request.candidates
    ]


@pytest.mark.parametrize("target", ["index", "manifest"])
def test_batch_resume_rejects_corrupt_or_mismatched_ledger(
    tmp_path: Path, target: str
) -> None:
    evaluator = _evaluator(tmp_path)
    request = _batch_request(evaluator, count=2)
    root = tmp_path / f"corrupt-{target}"
    evaluator.evaluate_many(
        request, executor=_ReplaceableExecutor(), output_dir=root
    )
    if target == "index":
        with (root / "BATCH_INDEX.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("{corrupt\n")
    else:
        manifest_path = root / "BATCH_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["objective_set_id"] = "objective_set_mismatch"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="stale|mismatch|corrupt"):
        evaluator.evaluate_many(
            request,
            executor=_ReplaceableExecutor(),
            output_dir=root,
            resume=True,
        )


def test_compact_batch_response_is_bounded_for_ten_thousand_records(
    tmp_path: Path,
) -> None:
    evaluator = _evaluator(tmp_path)
    records = tuple(
        _synthetic_record(
            evaluator,
            f"candidate_{index:05d}",
            tmp_path,
            failed=index % 5 == 0,
        )
        for index in range(10_000)
    )
    small = build_batch_result(
        batch_id="batch_scale",
        records=records[:10],
        executor_name="synthetic",
        artifact_root=tmp_path,
    )
    large = build_batch_result(
        batch_id="batch_scale",
        records=records,
        executor_name="synthetic",
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
    lowered = large_json.casefold()
    assert "wavelength" not in lowered
    assert '"channels"' not in lowered
    assert '"samples"' not in lowered


def test_duplicate_candidate_ids_are_rejected(tmp_path: Path) -> None:
    evaluator = _evaluator(tmp_path)
    candidate = _candidate(evaluator)

    with pytest.raises(ValueError, match="unique"):
        BatchEvaluationRequest(
            design_space_id=evaluator.design_space.design_space_id,
            objective_set_id=evaluator.objectives.objective_set_id,
            candidates=(candidate, candidate),
        )
