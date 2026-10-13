from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from tmm_engine.protocol.models import SpectralTargetContract
from tmm_engine.robust_optimization import _objective_and_attainment
from tmm_engine.schemas import SpectralTarget
from tmm_engine.spectral_reduction import reduce_spectral_mean
from tmm_engine.task_io import optimization_task_from_dict


@pytest.mark.parametrize("reduction,expected", [("sample_mean", 0.3), ("wavelength_integral_mean", 0.36)])
def test_irregular_grid_reductions_are_distinct(reduction, expected):
    assert reduce_spectral_mean([450, 500, 700], [0, 0, 0.9], reduction) == pytest.approx(expected)


@pytest.mark.parametrize("reduction", [None, "sample_mean", "wavelength_integral_mean"])
def test_mean_reduction_survives_public_protocol_and_task_io(reduction):
    target = SpectralTarget("R", 0.0, 450, 700, constraint="at_most", mean_reduction=reduction)
    target.validate()
    protocol = SpectralTargetContract.model_validate(asdict(target))
    assert protocol.mean_reduction == reduction
    task = optimization_task_from_dict({
        "simulation": {
            "stack": {
                "layers": [{"constant_n": 1.4, "thickness_nm": 100.0}],
                "incident": {"constant_n": 1.0}, "exit": {"constant_n": 1.52},
            },
            "spectrum": {"start_nm": 450, "stop_nm": 700, "points": 51},
        },
        "targets": [protocol.model_dump()],
    })
    assert task.targets[0].mean_reduction == reduction


def test_worst_case_rejects_mean_reduction_in_both_contracts():
    target = SpectralTarget("R", 0.0, 450, 700, aggregation="worst_case", mean_reduction="sample_mean")
    with pytest.raises(ValueError, match="only valid for mean"):
        target.validate()
    with pytest.raises(ValidationError, match="only valid for mean"):
        SpectralTargetContract.model_validate(asdict(target))


@pytest.mark.parametrize("constraint,target_value,tolerance,values", [
    ("at_most", 0.25, None, [0.9, 0.0, 0.0]),
    ("at_least", 0.75, None, [0.1, 1.0, 1.0]),
    ("match", 0.0, 0.25, [0.9, 0.0, 0.0]),
])
def test_robust_attainment_honors_reduction_without_changing_surrogate(
    constraint, target_value, tolerance, values
):
    losses = []
    for reduction, expected_passed in ((None, False), ("sample_mean", False), ("wavelength_integral_mean", True)):
        target = SpectralTarget(
            "R", target_value, 450, 700, constraint=constraint,
            tolerance=tolerance, mean_reduction=reduction,
        )
        task = SimpleNamespace(
            simulation=SimpleNamespace(spectrum=SimpleNamespace(
                wavelengths_nm=lambda: np.array([450.0, 575.0, 700.0])
            )), targets=(target,),
        )
        forward = SimpleNamespace(channel=lambda angle, polarization: {"R": values})
        loss, passed = _objective_and_attainment(task, forward)
        losses.append(loss)
        assert passed is expected_passed
    np.testing.assert_allclose(losses, [0.27, 0.27, 0.27])
