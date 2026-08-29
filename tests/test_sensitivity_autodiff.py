from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.requires_torch

from tmm_engine.protocol import RunResultEnvelope, SensitivityTaskContract  # noqa: E402
from tmm_engine.scientific_analysis import execute_sensitivity  # noqa: E402


def _request(*, layer_n: float = 2.0):
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
                        }
                    ],
                    "incident": {"constant_n": 1.0},
                    "exit": {"constant_n": 1.5},
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


def test_autodiff_and_independent_central_difference_agree(tmp_path: Path) -> None:
    output = tmp_path / "sensitivity"
    envelope = execute_sensitivity(_request(), output, device="cpu")
    result = json.loads((output / "SENSITIVITY_RESULT.json").read_text(encoding="utf-8"))
    certificate = json.loads(
        (output / "PHYSICS_ACCEPTANCE_CERTIFICATE.json").read_text(encoding="utf-8")
    )

    assert envelope["ok"] is True
    assert envelope["status"] == "completed"
    RunResultEnvelope.model_validate(envelope)
    assert result["status"] == "passed"
    assert result["ranking"] == [0]
    assert result["finite_difference_audit"]["passed"] is True
    assert certificate["uncertainty_budget"]["parameter_components"]
    assert certificate["evidence_coverage"]["uncertainty_quantified"] == "verified"
    assert {item["kind"] for item in envelope["artifacts"]} >= {
        "sensitivity_result",
        "physics_certificate",
    }

    row = result["parameters"][0]
    assert row["layer_index"] == 0
    assert row["audit_passed"] is True
    assert row["near_zero_gradient"] is False
    assert row["relative_error"] is not None
    assert row["relative_error"] <= 1e-3
    assert row["autodiff_derivative_per_nm"] == pytest.approx(
        row["finite_difference_derivative_per_nm"],
        rel=1e-3,
        abs=1e-7,
    )


def test_simple_uniform_medium_has_analytic_zero_reflection_gradient() -> None:
    from tmm_engine.differentiable import DifferentiableTMM

    wavelengths_um = torch.tensor([0.5, 0.55, 0.6], dtype=torch.float64)
    thickness_um = torch.tensor([[0.1]], dtype=torch.float64, requires_grad=True)
    nk = torch.ones((1, 3, 3), dtype=torch.complex128)
    result = DifferentiableTMM(polarization="s")(
        thickness_um,
        nk,
        wavelengths_um,
    )
    result.R.mean().backward()

    assert torch.allclose(result.R, torch.zeros_like(result.R), atol=1e-12)
    assert torch.allclose(result.T, torch.ones_like(result.T), atol=1e-12)
    assert float(thickness_um.grad.abs().max()) <= 1e-12
