from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.requires_torch

from tmm_engine import MaterialRegistry, TMMWorkbench  # noqa: E402
from tmm_engine.protocol import SensitivityTaskContract  # noqa: E402
from tmm_engine.scientific_analysis import execute_sensitivity  # noqa: E402
from tmm_engine.study_metrics import evaluate_metric  # noqa: E402
from tmm_engine.task_io import simulation_task_from_dict  # noqa: E402


def _request(*, zero_gradient: bool = False):
    layer_n = 1.0 if zero_gradient else 2.0
    exit_n = 1.0 if zero_gradient else 1.5
    document = {
        "schema_version": "sensitivity-task-v1",
        "mode": "sensitivity",
        "sensitivity": {
            "simulation": {
                "stack": {
                    "layers": [
                        {
                            "constant_n": layer_n,
                            "thickness_nm": 103.0,
                            "optimizable": True,
                            "label": "variable",
                        },
                        {
                            "constant_n": 1.4,
                            "thickness_nm": 80.0,
                            "optimizable": False,
                            "label": "fixed",
                        },
                    ],
                    "incident": {"constant_n": 1.0},
                    "exit": {"constant_n": exit_n},
                },
                "spectrum": {"start_nm": 500.0, "stop_nm": 600.0, "points": 31},
                "illumination": {
                    "angles_deg": [0.0],
                    "polarizations": ["unpolarized"],
                },
                "solver": "smatrix",
                "requested_outputs": ["R", "T", "A"],
            },
            "metric": {
                "name": "mean_R",
                "observable": "R",
                "wavelength_min_nm": 510.0,
                "wavelength_max_nm": 590.0,
                "aggregation": "mean",
            },
            "parameters": "optimizable_thicknesses",
            "finite_difference_step_nm": 0.01,
            "relative_error_tolerance": 1e-3,
            "absolute_error_tolerance": 1e-7,
        },
    }
    return SensitivityTaskContract.model_validate(document).sensitivity


def test_reported_fd_is_an_independent_numpy_central_difference_and_fixed_layers_are_excluded(
    tmp_path: Path,
) -> None:
    request = _request()
    output = tmp_path / "fixed-layer"
    execute_sensitivity(request, output, device="cpu")
    result = json.loads((output / "SENSITIVITY_RESULT.json").read_text(encoding="utf-8"))

    assert result["fixed_layer_indices"] == [1]
    assert [row["layer_index"] for row in result["parameters"]] == [0]
    row = result["parameters"][0]

    simulation = simulation_task_from_dict(request.simulation.model_dump(mode="python"))
    workbench = TMMWorkbench(MaterialRegistry())
    h = float(row["finite_difference_step_nm"])
    plus_layer = replace(simulation.stack.layers[0], thickness_nm=103.0 + h)
    minus_layer = replace(simulation.stack.layers[0], thickness_nm=103.0 - h)
    plus = workbench.simulate(
        replace(simulation, stack=replace(simulation.stack, layers=(plus_layer, *simulation.stack.layers[1:])))
    )
    minus = workbench.simulate(
        replace(simulation, stack=replace(simulation.stack, layers=(minus_layer, *simulation.stack.layers[1:])))
    )
    independent_fd = (
        evaluate_metric(plus, request.metric) - evaluate_metric(minus, request.metric)
    ) / (2.0 * h)
    assert row["finite_difference_derivative_per_nm"] == pytest.approx(
        independent_fd,
        rel=1e-9,
        abs=1e-10,
    )
    assert np.isfinite(row["autodiff_derivative_per_nm"])


def test_near_zero_gradient_uses_absolute_tolerance_without_fake_relative_error(
    tmp_path: Path,
) -> None:
    output = tmp_path / "near-zero"
    envelope = execute_sensitivity(_request(zero_gradient=True), output, device="cpu")
    result = json.loads((output / "SENSITIVITY_RESULT.json").read_text(encoding="utf-8"))
    row = result["parameters"][0]

    assert envelope["ok"] is True
    assert result["status"] == "passed"
    assert result["finite_difference_audit"]["passed"] is True
    assert row["near_zero_gradient"] is True
    assert row["relative_error"] is None
    assert row["absolute_error"] <= 1e-7
    assert row["audit_passed"] is True
