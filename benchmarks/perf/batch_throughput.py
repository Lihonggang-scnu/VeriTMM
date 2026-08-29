"""Informational raw-solver versus verified-batch throughput benchmark."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from tmm_engine import (
    ExecutionSettings,
    IlluminationSpec,
    LayerSpec,
    MaterialRegistry,
    MediumSpec,
    SimulationTask,
    SpectralGrid,
    StackSpec,
    TMMWorkbench,
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


def _build_evaluator(root: Path, candidate_count: int) -> tuple[ResearchEvaluator, tuple[object, ...]]:
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
        spectrum=SpectralGrid(start_nm=500.0, stop_nm=600.0, points=21),
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
    evaluator = ResearchEvaluator(
        space,
        objectives,
        EvaluatorConfig(output_root=str(root / "research"), cache=False),
        execution_settings=ExecutionSettings(
            write_plot=False,
            convergence_max_refinements=1,
        ),
    )
    candidates = space.sample(candidate_count, seed=42)
    return evaluator, candidates


def run_benchmark(candidate_count: int, output: Path) -> dict[str, object]:
    if candidate_count < 1:
        raise ValueError("candidate count must be positive")
    with tempfile.TemporaryDirectory(prefix="veritmm_batch_perf_") as temporary:
        root = Path(temporary)
        evaluator, all_candidates = _build_evaluator(root, candidate_count)
        candidates = tuple(all_candidates)
        tasks = [evaluator.design_space.to_simulation_task(candidate) for candidate in candidates]
        workbench = TMMWorkbench(MaterialRegistry())

        raw_start = time.perf_counter()
        for task in tasks:
            workbench.simulate(task)
        raw_seconds = max(time.perf_counter() - raw_start, 1e-12)

        request = BatchEvaluationRequest(
            design_space_id=evaluator.design_space.design_space_id,
            objective_set_id=evaluator.objectives.objective_set_id,
            candidates=candidates,
            metadata={"benchmark": "batch_throughput", "seed": 42},
        )
        verified_start = time.perf_counter()
        verified = evaluator.evaluate_many(
            request,
            batch_size=10,
            resume=False,
            output_dir=root / "verified_batch",
        )
        verified_seconds = max(time.perf_counter() - verified_start, 1e-12)

    result = {
        "schema_version": "veritmm-batch-throughput-v1",
        "candidate_count": len(candidates),
        "batch_size": 10,
        "raw_solver_time_seconds": float(raw_seconds),
        "verified_evaluation_time_seconds": float(verified_seconds),
        "raw_solver_throughput": float(len(candidates) / raw_seconds),
        "verified_evaluations_per_second": float(len(candidates) / verified_seconds),
        "verified_status": verified.status,
        "verified_completed_count": verified.completed_count,
        "verified_failed_count": verified.failed_count,
        "verification_decisions_unchanged": True,
        "informational_only": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("perf_result.json"))
    args = parser.parse_args()
    result = run_benchmark(args.candidates, args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
