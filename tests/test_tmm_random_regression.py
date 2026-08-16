from __future__ import annotations

from dataclasses import replace

import numpy as np

from tmm_engine import (
    IlluminationSpec,
    LayerSpec,
    MaterialRegistry,
    MediumSpec,
    SimulationTask,
    SpectralGrid,
    StackSpec,
    TMMWorkbench,
)


def test_seeded_random_passive_stacks_match_independent_solver() -> None:
    """Golden randomized oracle test for signs, branches and cache mistakes."""

    rng = np.random.default_rng(20260809)
    workbench = TMMWorkbench(MaterialRegistry())
    wavelengths = SpectralGrid(start_nm=430.0, stop_nm=980.0, points=19)
    for _ in range(24):
        layers = tuple(
            LayerSpec(
                None,
                float(rng.uniform(12.0, 520.0)),
                constant_n=float(rng.uniform(1.05, 3.2)),
                constant_k=float(rng.uniform(0.0, 0.12)),
                optimizable=False,
            )
            for _ in range(int(rng.integers(1, 9)))
        )
        task = SimulationTask(
            stack=StackSpec(
                layers=layers,
                incident=MediumSpec.air(),
                exit=MediumSpec(constant_n=float(rng.uniform(1.1, 2.0))),
            ),
            spectrum=wavelengths,
            illumination=IlluminationSpec(
                angles_deg=(0.0, float(rng.uniform(5.0, 72.0))),
                polarizations=("s", "p", "unpolarized"),
            ),
            solver="smatrix",
        )
        internal = workbench.simulate(task)
        reference = workbench.simulate(replace(task, solver="byrnes"))
        for channel_key, values in internal.channels.items():
            for observable in ("R", "T", "A"):
                np.testing.assert_allclose(
                    values[observable],
                    reference.channels[channel_key][observable],
                    rtol=2e-9,
                    atol=2e-10,
                )
