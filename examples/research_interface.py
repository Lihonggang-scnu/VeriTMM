"""Minimal verified dataset generation with the VeriTMM v0.6 research API."""

from __future__ import annotations

import argparse
from pathlib import Path

from tmm_engine.execution import ExecutionSettings
from tmm_engine.research import (
    ContinuousThicknessVariable,
    DatasetConfig,
    DatasetFactory,
    DesignSpace,
    DesignSpaceContract,
    EvaluatorConfig,
    ObjectiveSet,
    ObjectiveSpec,
    ResearchEvaluator,
    SamplingPlan,
)
from tmm_engine.schemas import (
    IlluminationSpec,
    LayerSpec,
    MediumSpec,
    SimulationTask,
    SpectralGrid,
    StackSpec,
)


def build_design_space() -> DesignSpace:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("outputs/research-demo"))
    args = parser.parse_args()
    design_space = build_design_space()
    objectives = ObjectiveSet(
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
    evaluator = ResearchEvaluator(
        design_space,
        objectives,
        EvaluatorConfig(output_root=str(args.output_root / "runs"), cache=False),
        execution_settings=ExecutionSettings(
            write_plot=False,
            convergence_max_refinements=2,
        ),
    )
    result = DatasetFactory(design_space, evaluator).generate(
        SamplingPlan(strategy="sobol", sample_count=2, seed=7),
        DatasetConfig(
            output_root=str(args.output_root / "dataset"),
            resume=True,
            cache=False,
        ),
    )
    print(result.canonical_json())


if __name__ == "__main__":
    main()
