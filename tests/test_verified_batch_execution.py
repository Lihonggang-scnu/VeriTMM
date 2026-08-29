from __future__ import annotations

import json
from pathlib import Path

from tmm_engine import (
    ExecutionSettings,
    IlluminationSpec,
    LayerSpec,
    MediumSpec,
    SimulationTask,
    SpectralGrid,
    StackSpec,
)
from tmm_engine.research import (
    BatchEvaluationRequest,
    ContinuousThicknessVariable,
    DesignSpace,
    DesignSpaceContract,
    EvaluatorConfig,
    ObjectiveSet,
    ObjectiveSpec,
    ResearchEvaluator,
)


def _evaluator(root: Path) -> ResearchEvaluator:
    task = SimulationTask(
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
    )
    space = DesignSpace(
        DesignSpaceContract(
            base_task=task,
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
    objectives = ObjectiveSet(
        objectives=(
            ObjectiveSpec(
                name="mean-R",
                direction="maximize",
                observable="R",
                wavelength_min_nm=500.0,
                wavelength_max_nm=600.0,
            ),
        )
    )
    return ResearchEvaluator(
        space,
        objectives,
        EvaluatorConfig(output_root=str(root / "research"), cache=False),
        execution_settings=ExecutionSettings(
            write_plot=False,
            convergence_max_refinements=1,
        ),
    )


def _request(evaluator: ResearchEvaluator, count: int = 5) -> BatchEvaluationRequest:
    candidates = evaluator.design_space.sample(count, seed=123)
    return BatchEvaluationRequest(
        design_space_id=evaluator.design_space.design_space_id,
        objective_set_id=evaluator.objectives.objective_set_id,
        candidates=candidates,
        metadata={"test_seed": 123},
    )


def _records(result) -> list[dict]:
    path = Path(result.artifact_root) / "BATCH_INDEX.jsonl"
    return [json.loads(line)["record"] for line in path.read_text(encoding="utf-8").splitlines()]


def _artifact(record: dict, kind: str) -> dict:
    reference = next(item for item in record["artifacts"] if item["kind"] == kind)
    return json.loads(
        (Path(record["artifact_root"]) / reference["path"]).read_text(encoding="utf-8")
    )


def _scientific_record_view(record: dict) -> dict:
    return {
        "candidate_id": record["candidate_id"],
        "status": record["status"],
        "physics_accepted": record["physics_accepted"],
        "certificate_id": record["certificate_id"],
        "task_sha256": record["task_sha256"],
        "objective_values": record["objective_values"],
        "objective_scores": record["objective_scores"],
        "certificate": _artifact(record, "physics_certificate"),
        "simulation": _artifact(record, "simulation_result"),
    }


def test_scalar_and_chunked_batch_certificates_and_spectra_match(tmp_path: Path) -> None:
    scalar_evaluator = _evaluator(tmp_path / "scalar")
    scalar_request = _request(scalar_evaluator)
    scalar = scalar_evaluator.evaluate_many(
        scalar_request,
        resume=False,
        output_dir=tmp_path / "scalar-batch",
    )

    batch_evaluator = _evaluator(tmp_path / "chunked")
    batch_request = _request(batch_evaluator)
    chunked = batch_evaluator.evaluate_many(
        batch_request,
        batch_size=2,
        resume=False,
        output_dir=tmp_path / "chunked-batch",
    )

    assert scalar.status == chunked.status == "completed"
    assert scalar.completed_count == chunked.completed_count == 5
    assert scalar.failed_count == chunked.failed_count == 0
    scalar_views = [_scientific_record_view(record) for record in _records(scalar)]
    chunked_views = [_scientific_record_view(record) for record in _records(chunked)]
    assert scalar_views == chunked_views


def test_chunk_size_does_not_change_verified_records(tmp_path: Path) -> None:
    first_evaluator = _evaluator(tmp_path / "first")
    first = first_evaluator.evaluate_many(
        _request(first_evaluator),
        batch_size=2,
        resume=False,
        output_dir=tmp_path / "batch-two",
    )
    second_evaluator = _evaluator(tmp_path / "second")
    second = second_evaluator.evaluate_many(
        _request(second_evaluator),
        batch_size=4,
        resume=False,
        output_dir=tmp_path / "batch-four",
    )

    assert first.executor == second.executor == "chunked_verified"
    assert [_scientific_record_view(record) for record in _records(first)] == [
        _scientific_record_view(record) for record in _records(second)
    ]


def test_fixed_seed_chunked_batch_is_reproducible(tmp_path: Path) -> None:
    first_evaluator = _evaluator(tmp_path / "replay-one")
    first = first_evaluator.evaluate_many(
        _request(first_evaluator),
        batch_size=2,
        resume=False,
        output_dir=tmp_path / "replay-batch-one",
    )
    second_evaluator = _evaluator(tmp_path / "replay-two")
    second = second_evaluator.evaluate_many(
        _request(second_evaluator),
        batch_size=2,
        resume=False,
        output_dir=tmp_path / "replay-batch-two",
    )

    assert [_scientific_record_view(record) for record in _records(first)] == [
        _scientific_record_view(record) for record in _records(second)
    ]
