from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tmm_engine import (  # noqa: E402
    IlluminationSpec,
    LayerSpec,
    MaterialRegistry,
    MediumSpec,
    OptimizationTask,
    OptimizerSpec,
    SimulationTask,
    SpectralGrid,
    SpectralTarget,
    StackSpec,
    TMMWorkbench,
)
from tmm_engine.differentiable import DifferentiableTMM  # noqa: E402
from tmm_engine.optimization import DifferentiableThicknessOptimizer  # noqa: E402


def _constant_nk(n_values: list[complex], wavelengths: np.ndarray) -> torch.Tensor:
    data = np.stack([np.full(wavelengths.shape, n, dtype=np.complex128) for n in n_values])
    return torch.tensor(data[None, :, :], dtype=torch.complex128)


@pytest.mark.parametrize("polarization", ["s", "p", "unpolarized"])
def test_torch_backend_matches_numpy_forward_solver(polarization: str) -> None:
    wavelengths_nm = np.linspace(450.0, 750.0, 61)
    wavelengths_um = torch.tensor(wavelengths_nm * 1e-3, dtype=torch.float64)
    thicknesses_um = torch.tensor([[0.091, 0.137]], dtype=torch.float64)
    nk = _constant_nk([1.0, 2.15, 1.43, 1.52], wavelengths_nm)
    torch_result = DifferentiableTMM(polarization=polarization)(
        thicknesses_um, nk, wavelengths_um, theta_rad=np.deg2rad(31.0)
    )

    task = SimulationTask(
        stack=StackSpec(
            layers=(
                LayerSpec(None, 91.0, constant_n=2.15),
                LayerSpec(None, 137.0, constant_n=1.43),
            ),
            incident=MediumSpec.air(),
            exit=MediumSpec(constant_n=1.52),
        ),
        spectrum=SpectralGrid(values_nm=tuple(wavelengths_nm.tolist())),
        illumination=IlluminationSpec(angles_deg=(31.0,), polarizations=(polarization,)),
    )
    numpy_result = TMMWorkbench(MaterialRegistry()).simulate(task).channel(31.0, polarization)
    np.testing.assert_allclose(
        torch_result.R.detach().numpy()[0], numpy_result["R"], rtol=2e-9, atol=2e-10
    )
    np.testing.assert_allclose(
        torch_result.T.detach().numpy()[0], numpy_result["T"], rtol=2e-9, atol=2e-10
    )


def test_autograd_thickness_gradient_matches_central_difference() -> None:
    wavelengths_nm = np.linspace(500.0, 650.0, 41)
    wavelengths_um = torch.tensor(wavelengths_nm * 1e-3, dtype=torch.float64)
    nk = _constant_nk([1.0, 2.0, 1.5], wavelengths_nm)
    thickness = torch.tensor([[0.103]], dtype=torch.float64, requires_grad=True)
    solver = DifferentiableTMM(polarization="s")
    objective = solver(thickness, nk, wavelengths_um).R.mean()
    objective.backward()
    automatic = float(thickness.grad[0, 0].item())

    step = 1e-6
    with torch.no_grad():
        plus = float(solver(thickness + step, nk, wavelengths_um).R.mean().item())
        minus = float(solver(thickness - step, nk, wavelengths_um).R.mean().item())
    finite_difference = (plus - minus) / (2.0 * step)
    assert automatic == pytest.approx(finite_difference, rel=2e-5, abs=2e-6)


def test_thickness_optimizer_improves_and_independent_solver_confirms() -> None:
    n_coating = float(np.sqrt(1.5))
    simulation = SimulationTask(
        stack=StackSpec(
            layers=(
                LayerSpec(
                    None,
                    200.0,
                    constant_n=n_coating,
                    min_thickness_nm=40.0,
                    max_thickness_nm=250.0,
                ),
            ),
            incident=MediumSpec.air(),
            exit=MediumSpec(constant_n=1.5),
        ),
        spectrum=SpectralGrid(start_nm=500.0, stop_nm=600.0, points=101),
    )
    task = OptimizationTask(
        simulation=simulation,
        targets=(SpectralTarget("R", 0.0, 540.0, 560.0, name="ar_center"),),
        optimizer=OptimizerSpec(
            method="adam",
            max_steps=100,
            learning_rate=0.08,
            starts=3,
            seed=7,
            early_stop_patience=30,
        ),
    )
    optimizer = DifferentiableThicknessOptimizer(MaterialRegistry())
    result = optimizer.optimize(task)
    assert result.status == "completed"
    assert result.evaluation_count > 0
    assert result.audit["evaluation_count"] == result.evaluation_count
    assert result.to_dict()["evaluation_count"] == result.evaluation_count
    assert len(result.candidate_designs) >= 2
    assert result.candidate_designs[0]["objective_loss"] <= result.candidate_designs[-1]["objective_loss"]
    assert result.optimized_loss < result.initial_loss * 0.05
    expected_quarter_wave = 550.0 / (4.0 * n_coating)
    assert result.optimized_thicknesses_nm[0] == pytest.approx(expected_quarter_wave, abs=8.0)

    validated_task = replace(
        simulation,
        stack=replace(
            simulation.stack,
            layers=(replace(simulation.stack.layers[0], thickness_nm=result.optimized_thicknesses_nm[0]),),
        ),
    )
    channel = TMMWorkbench(MaterialRegistry()).simulate(validated_task).channel()
    mask = (validated_task.spectrum.wavelengths_nm() >= 540.0) & (
        validated_task.spectrum.wavelengths_nm() <= 560.0
    )
    assert float(np.mean(channel["R"][mask])) < 2e-4


def test_optimizer_rejects_incoherent_differentiation() -> None:
    simulation = SimulationTask(
        stack=StackSpec(
            layers=(LayerSpec(None, 1000.0, coherence="incoherent", constant_n=1.5),),
            exit=MediumSpec.air(),
        ),
        spectrum=SpectralGrid(start_nm=500.0, stop_nm=600.0, points=11),
    )
    task = OptimizationTask(
        simulation=simulation,
        targets=(SpectralTarget("R", 0.0, 500.0, 600.0),),
    )
    with pytest.raises(ValueError, match="coherent"):
        DifferentiableThicknessOptimizer(MaterialRegistry()).optimize(task)


def test_unmet_soft_target_returns_physics_valid_best_effort() -> None:
    simulation = SimulationTask(
        stack=StackSpec(
            layers=(
                LayerSpec(
                    None,
                    180.0,
                    constant_n=float(np.sqrt(1.5)),
                    min_thickness_nm=40.0,
                    max_thickness_nm=250.0,
                ),
            ),
            incident=MediumSpec.air(),
            exit=MediumSpec(constant_n=1.5),
        ),
        spectrum=SpectralGrid(start_nm=500.0, stop_nm=600.0, points=51),
    )
    task = OptimizationTask(
        simulation=simulation,
        targets=(
            SpectralTarget(
                "R",
                0.0,
                500.0,
                600.0,
                constraint="at_most",
                aggregation="worst_case",
                name="impossible_zero_reflectance_band",
            ),
        ),
        optimizer=OptimizerSpec(method="adam", max_steps=12, starts=1, seed=2),
    )
    optimizer = DifferentiableThicknessOptimizer(MaterialRegistry())
    result = optimizer.optimize(task)
    _, _, validation = optimizer.validate_result(task, result, TMMWorkbench(MaterialRegistry()))
    assert validation["status"] == "passed"
    assert validation["design_outcome_status"] == "physically_valid_best_effort"
    target = validation["target_attainment"]["impossible_zero_reflectance_band"]
    assert target["shortfall"] > 0
    assert target["role"] == "soft_scoring_objective"
