from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import numpy as np

from tmm_engine import (
    AcceptanceSettings,
    IlluminationSpec,
    LayerSpec,
    MaterialRegistry,
    MediumSpec,
    SimulationTask,
    SpectralConvergenceSettings,
    SpectralGrid,
    StackSpec,
    TMMWorkbench,
    certify_simulation,
)
from tmm_engine.convergence import audit_spectral_convergence
from tmm_engine.schemas import dataclass_to_dict


def _dbr_stack() -> StackSpec:
    layers: list[LayerSpec] = []
    pair = (
        LayerSpec(None, 68.1818181818, constant_n=2.2),
        LayerSpec(None, 103.4482758621, constant_n=1.45),
    )
    for _ in range(3):
        layers.extend(pair)
    layers.append(LayerSpec(None, 206.9, constant_n=1.45))
    for _ in range(3):
        layers.extend(pair)
    return StackSpec(
        layers=tuple(layers),
        incident=MediumSpec.air(),
        exit=MediumSpec(constant_n=1.52),
        name="adaptive-defect-cavity",
    )


def _task(
    *,
    points: int = 7,
    angles: tuple[float, ...] = (0.0,),
    polarizations: tuple[str, ...] = ("unpolarized",),
) -> SimulationTask:
    return SimulationTask(
        stack=StackSpec(
            layers=(LayerSpec(None, 105.0, constant_n=1.38),),
            incident=MediumSpec.air(),
            exit=MediumSpec(constant_n=1.52),
        ),
        spectrum=SpectralGrid(start_nm=450.0, stop_nm=750.0, points=points),
        illumination=IlluminationSpec(angles, polarizations),
    )


def _narrow_task() -> SimulationTask:
    return SimulationTask(
        stack=_dbr_stack(),
        spectrum=SpectralGrid(start_nm=560.0, stop_nm=640.0, points=3),
    )


def _task_hash(task: SimulationTask) -> str:
    payload = json.dumps(
        dataclass_to_dict(task), sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _channel_snapshot(result):
    return {
        channel: {
            observable: np.asarray(values[observable], dtype=np.float64).copy()
            for observable in values
        }
        for channel, values in result.channels.items()
    }


def test_adaptive_refinement_preserves_declared_result_and_task_hash() -> None:
    task = _task(points=5, angles=(0.0, 45.0))
    workbench = TMMWorkbench(MaterialRegistry())
    initial = workbench.simulate(task)
    initial_wavelengths = initial.wavelengths_nm.copy()
    initial_channels = _channel_snapshot(initial)
    before_hash = _task_hash(task)

    outcome = audit_spectral_convergence(
        workbench,
        task,
        SpectralConvergenceSettings(
            max_refinements=3,
            max_pointwise_deviation=1e-4,
            max_integral_deviation=1e-4,
            max_intervals_per_round=1,
            max_angle_intervals_per_round=1,
        ),
        initial_result=initial,
    )

    assert outcome.final_result is initial
    np.testing.assert_array_equal(outcome.final_result.wavelengths_nm, initial_wavelengths)
    for channel, values in initial_channels.items():
        for observable, expected in values.items():
            np.testing.assert_array_equal(outcome.final_result.channels[channel][observable], expected)
    assert _task_hash(task) == before_hash
    assert outcome.declared_grid_sha256 != ""
    assert outcome.verification_grid_sha256 == hashlib.sha256(
        np.ascontiguousarray(outcome.verification_result.wavelengths_nm, dtype=np.float64).tobytes()
    ).hexdigest()
    assert outcome.report_dict()["declared_grid_sha256"] == outcome.declared_grid_sha256


def test_adaptive_refinement_replay_is_byte_deterministic() -> None:
    task = _task(points=7, angles=(0.0, 56.0, 82.0), polarizations=("p",))
    settings = SpectralConvergenceSettings(
        max_refinements=4,
        max_pointwise_deviation=2e-3,
        max_integral_deviation=2e-3,
        max_intervals_per_round=2,
        max_angle_intervals_per_round=2,
    )
    workbench = TMMWorkbench(MaterialRegistry())
    first = audit_spectral_convergence(workbench, task, settings)
    second = audit_spectral_convergence(workbench, task, settings)

    assert first.report_dict() == second.report_dict()
    assert [item["points_added"] for item in first.rounds] == [
        item["points_added"] for item in second.rounds
    ]
    assert first.verification_grid_sha256 == second.verification_grid_sha256
    assert first.verification_angle_grid_sha256 == second.verification_angle_grid_sha256


def test_narrow_resonance_adapts_with_fewer_points_than_dense_referee() -> None:
    task = _narrow_task()
    workbench = TMMWorkbench(MaterialRegistry())
    settings = SpectralConvergenceSettings(
        max_refinements=8,
        max_pointwise_deviation=5e-3,
        max_integral_deviation=1e-3,
        max_intervals_per_round=2,
    )
    outcome = audit_spectral_convergence(workbench, task, settings)
    dense = workbench.simulate(
        replace(task, spectrum=SpectralGrid(start_nm=560.0, stop_nm=640.0, points=161))
    )

    assert outcome.passed
    assert outcome.refinement_status == "converged"
    assert outcome.rounds[-1]["max_pointwise_deviation"] <= settings.max_pointwise_deviation
    assert outcome.verification_result.wavelengths_nm.size <= 8
    assert outcome.verification_result.wavelengths_nm.size < dense.wavelengths_nm.size // 4
    assert any(
        item["points_added"]["wavelengths"] > 0
        for item in outcome.rounds
    )


def test_angular_refinement_targets_brewster_and_near_grazing_intervals() -> None:
    task = _task(
        points=11,
        angles=(0.0, 45.0, 70.0, 85.0),
        polarizations=("p",),
    )
    settings = SpectralConvergenceSettings(
        max_refinements=6,
        max_pointwise_deviation=5e-3,
        max_integral_deviation=5e-3,
        max_angular_deviation=5e-3,
        max_intervals_per_round=1,
        max_angle_intervals_per_round=2,
    )
    outcome = audit_spectral_convergence(TMMWorkbench(MaterialRegistry()), task, settings)
    intervals = [
        item
        for round_ledger in outcome.rounds
        for item in round_ledger["angular_intervals"]
    ]
    sampled_angles = sorted(
        {float(key.split("|")[0].split("=")[1]) for key in outcome.verification_result.channels}
    )

    assert outcome.passed
    assert outcome.refinement_status == "converged"
    assert outcome.declared_angle_grid_sha256 != outcome.verification_angle_grid_sha256
    assert any(item["lower"] <= 56.66 <= item["upper"] for item in intervals)
    assert any(item["lower"] >= 70.0 and item["upper"] <= 85.0 for item in intervals)
    assert any(45.0 < angle < 70.0 for angle in sampled_angles)
    assert any(70.0 < angle < 85.0 for angle in sampled_angles)
    assert outcome.rounds[-1]["max_angular_deviation"] <= settings.max_angular_deviation


def test_budget_exhaustion_is_not_numerically_verified() -> None:
    task = SimulationTask(
        stack=StackSpec(
            layers=(LayerSpec(None, 900.0, constant_n=2.35),),
            incident=MediumSpec.air(),
            exit=MediumSpec(constant_n=1.45),
        ),
        spectrum=SpectralGrid(start_nm=400.0, stop_nm=800.0, points=3),
    )
    settings = AcceptanceSettings(
        require_independent_solver=False,
        convergence=SpectralConvergenceSettings(
            max_refinements=3,
            max_pointwise_deviation=1e-8,
            max_integral_deviation=1e-9,
            maximum_points=3,
        ),
    )

    certified = certify_simulation(TMMWorkbench(MaterialRegistry()), task, settings)
    ledger = certified.certificate["spectral_convergence"]

    assert ledger["refinement_status"] == "budget_exhausted"
    assert ledger["worst_unresolved_interval"] is not None
    assert ledger["worst_unresolved_interval"]["lower"] < ledger["worst_unresolved_interval"]["upper"]
    assert certified.certificate["evidence_coverage"]["numerical_convergence"] != "verified"
    np.testing.assert_array_equal(certified.result.wavelengths_nm, task.spectrum.wavelengths_nm())
