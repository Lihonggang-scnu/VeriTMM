"""Deterministic spectral convergence audit for multilayer simulations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, List, Tuple

import numpy as np

from .schemas import SimulationTask, SpectralGrid
from .workbench import ForwardSimulationResult, TMMWorkbench


@dataclass(frozen=True)
class SpectralConvergenceSettings:
    max_refinements: int = 3
    max_pointwise_deviation: float = 5e-3
    max_integral_deviation: float = 1e-3
    maximum_points: int = 20_001

    def validate(self) -> None:
        if self.max_refinements < 1:
            raise ValueError("max_refinements must be positive")
        if self.max_pointwise_deviation <= 0 or self.max_integral_deviation <= 0:
            raise ValueError("convergence tolerances must be positive")
        if self.maximum_points < 3:
            raise ValueError("maximum_points must be at least 3")


@dataclass
class SpectralConvergenceOutcome:
    status: str
    passed: bool
    rounds: List[Dict[str, Any]]
    settings: Dict[str, Any]
    final_points: int
    final_result: ForwardSimulationResult

    def report_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "passed": self.passed,
            "rounds": self.rounds,
            "settings": self.settings,
            "final_points": self.final_points,
        }


def _refined_grid(values: np.ndarray) -> np.ndarray:
    midpoints = 0.5 * (values[:-1] + values[1:])
    refined = np.empty(values.size + midpoints.size, dtype=np.float64)
    refined[0::2] = values
    refined[1::2] = midpoints
    return refined


def _normalized_integral(values: np.ndarray, wavelengths: np.ndarray) -> float:
    width = float(wavelengths[-1] - wavelengths[0])
    if width <= 0:
        return float(values[0])
    return float(np.trapezoid(values, wavelengths) / width)


def _compare_levels(
    coarse: ForwardSimulationResult, refined: ForwardSimulationResult
) -> Tuple[float, float, Dict[str, Any]]:
    max_pointwise = 0.0
    max_integral = 0.0
    channels: Dict[str, Any] = {}
    for channel_key, coarse_values in coarse.channels.items():
        channel_metrics: Dict[str, Any] = {}
        refined_values = refined.channels[channel_key]
        for observable in ("R", "T", "A"):
            old = np.asarray(coarse_values[observable], dtype=np.float64)
            new = np.asarray(refined_values[observable], dtype=np.float64)
            interpolated = np.interp(refined.wavelengths_nm, coarse.wavelengths_nm, old)
            pointwise = float(np.max(np.abs(new - interpolated)))
            integral = abs(
                _normalized_integral(new, refined.wavelengths_nm)
                - _normalized_integral(old, coarse.wavelengths_nm)
            )
            channel_metrics[observable] = {
                "max_interpolation_deviation": pointwise,
                "normalized_integral_deviation": integral,
            }
            max_pointwise = max(max_pointwise, pointwise)
            max_integral = max(max_integral, integral)
        channels[channel_key] = channel_metrics
    return max_pointwise, max_integral, channels


def audit_spectral_convergence(
    workbench: TMMWorkbench,
    task: SimulationTask,
    settings: SpectralConvergenceSettings | None = None,
    initial_result: ForwardSimulationResult | None = None,
) -> SpectralConvergenceOutcome:
    """Refine between every pair of wavelengths until spectra stop changing.

    The pointwise metric compares newly evaluated midpoints with interpolation
    from the coarser spectrum.  This catches unresolved resonances that an
    energy-conservation check alone cannot detect.
    """

    settings = settings or SpectralConvergenceSettings()
    settings.validate()
    current_task = task
    current = initial_result or workbench.simulate(current_task)
    rounds: List[Dict[str, Any]] = []
    passed = False
    status = "spectral_convergence_failure"
    for index in range(settings.max_refinements):
        old_grid = current.wavelengths_nm
        new_grid = _refined_grid(old_grid)
        if new_grid.size > settings.maximum_points:
            status = "point_budget_exhausted"
            break
        refined_task = replace(current_task, spectrum=SpectralGrid(values_nm=tuple(new_grid.tolist())))
        refined = workbench.simulate(refined_task)
        pointwise, integral, detail = _compare_levels(current, refined)
        round_passed = bool(
            pointwise <= settings.max_pointwise_deviation
            and integral <= settings.max_integral_deviation
        )
        rounds.append(
            {
                "round": index + 1,
                "coarse_points": int(old_grid.size),
                "refined_points": int(new_grid.size),
                "max_pointwise_deviation": pointwise,
                "max_integral_deviation": integral,
                "passed": round_passed,
                "channels": detail,
            }
        )
        current_task, current = refined_task, refined
        if round_passed:
            passed = True
            status = "passed"
            break
    return SpectralConvergenceOutcome(
        status=status,
        passed=passed,
        rounds=rounds,
        settings=asdict(settings),
        final_points=int(current.wavelengths_nm.size),
        final_result=current,
    )


__all__ = [
    "SpectralConvergenceOutcome",
    "SpectralConvergenceSettings",
    "audit_spectral_convergence",
]
