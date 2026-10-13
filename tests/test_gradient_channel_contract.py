"""The scalar API must never discard supplied illumination channels."""

import pytest

pytest.importorskip("torch")

from tmm_engine import (
    IlluminationSpec,
    LayerSpec,
    MediumSpec,
    SimulationTask,
    SpectralGrid,
    StackSpec,
)
from tmm_engine.gradient_api import GradientRequestError, compute_gradient, compute_sensitivity


@pytest.mark.parametrize(
    "angles,polarizations", [((0.0, 30.0), ("s",)), ((0.0,), ("s", "p"))]
)
def test_reject_multichannel(angles, polarizations):
    task = SimulationTask(
        stack=StackSpec(
            layers=(LayerSpec(None, 90.0, constant_n=2.0, optimizable=True),),
            incident=MediumSpec.air(),
            exit=MediumSpec(constant_n=1.5),
        ),
        spectrum=SpectralGrid(500.0, 700.0, 11),
        illumination=IlluminationSpec(angles, polarizations),
    )
    for callback in (compute_gradient, compute_sensitivity):
        with pytest.raises(GradientRequestError) as caught:
            callback(task, objective="mean_T")
        assert caught.value.code == "multichannel_objective_unsupported"
