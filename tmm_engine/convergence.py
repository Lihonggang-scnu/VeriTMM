"""Deterministic local spectral and angular convergence auditing."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .schemas import IlluminationSpec, SimulationTask, SpectralGrid
from .workbench import ForwardSimulationResult, TMMWorkbench


@dataclass(frozen=True)
class SpectralConvergenceSettings:
    """Configuration for the verifier-internal adaptive refinement grid.

    ``maximum_points`` applies to wavelength verification points and
    ``maximum_angle_points`` applies to angle verification points.  Neither
    limit changes the user-declared task grid or the returned simulation
    result.
    """

    max_refinements: int = 6
    max_pointwise_deviation: float = 5e-3
    max_integral_deviation: float = 1e-3
    maximum_points: int = 20_001
    maximum_angle_points: int = 4_097
    max_intervals_per_round: int = 8
    max_angle_intervals_per_round: int = 8
    max_angular_deviation: float = 5e-2

    def validate(self) -> None:
        if self.max_refinements < 1:
            raise ValueError("max_refinements must be positive")
        if (
            self.max_pointwise_deviation <= 0
            or self.max_integral_deviation <= 0
            or self.max_angular_deviation <= 0
        ):
            raise ValueError("convergence tolerances must be positive")
        if self.maximum_points < 3:
            raise ValueError("maximum_points must be at least 3")
        if self.maximum_angle_points < 2:
            raise ValueError("maximum_angle_points must be at least 2")
        if self.max_intervals_per_round < 1:
            raise ValueError("max_intervals_per_round must be positive")
        if self.max_angle_intervals_per_round < 1:
            raise ValueError("max_angle_intervals_per_round must be positive")


@dataclass
class SpectralConvergenceOutcome:
    """Convergence evidence and both declared and verifier-internal results."""

    status: str
    passed: bool
    rounds: List[Dict[str, Any]]
    settings: Dict[str, Any]
    final_points: int
    final_result: ForwardSimulationResult
    verification_result: ForwardSimulationResult
    declared_grid_sha256: str
    verification_grid_sha256: str
    declared_angle_grid_sha256: str
    verification_angle_grid_sha256: str
    refinement_status: str
    worst_unresolved_interval: Dict[str, Any] | None

    def report_dict(self) -> Dict[str, Any]:
        """Return the JSON-ready additive convergence ledger."""

        return {
            "status": self.status,
            "passed": self.passed,
            "rounds": self.rounds,
            "settings": self.settings,
            # ``final_points`` is retained for compatibility and means the
            # number of internal wavelength verification points.
            "final_points": self.final_points,
            "verification_points": int(self.verification_result.wavelengths_nm.size),
            "verification_angle_points": int(self._verification_angles().size),
            "declared_grid_sha256": self.declared_grid_sha256,
            "verification_grid_sha256": self.verification_grid_sha256,
            "declared_angle_grid_sha256": self.declared_angle_grid_sha256,
            "verification_angle_grid_sha256": self.verification_angle_grid_sha256,
            "refinement_status": self.refinement_status,
            "worst_unresolved_interval": self.worst_unresolved_interval,
            "grid_hash_scope": "contiguous float64 wavelength and angle arrays",
        }

    def _verification_angles(self) -> np.ndarray:
        """Read the internal angle grid recorded in the result's channels."""

        angles: set[float] = set()
        for key in self.verification_result.channels:
            if key.startswith("angle=") and "|pol=" in key:
                value = key[len("angle=") : key.index("|pol=")]
                try:
                    angles.add(float(value))
                except ValueError:
                    continue
        if not angles:
            return np.asarray([0.0], dtype=np.float64)
        return np.asarray(sorted(angles), dtype=np.float64)


def _grid_sha256(values: Sequence[float] | np.ndarray) -> str:
    """Hash a canonical contiguous float64 grid, including exact bytes."""

    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64).reshape(-1))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _channel_key(angle_deg: float, polarization: str) -> str:
    return f"angle={float(angle_deg):g}|pol={polarization}"


def _normalized_integral(values: np.ndarray, wavelengths: np.ndarray) -> float:
    width = float(wavelengths[-1] - wavelengths[0])
    if width <= 0:
        return float(values[0])
    return float(np.trapezoid(values, wavelengths) / width)


def _compare_levels(
    coarse: ForwardSimulationResult, refined: ForwardSimulationResult
) -> Tuple[float, float, Dict[str, Any]]:
    """Compare spectra on a wavelength grid that contains the coarse grid."""

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


def _max_abs(values: Sequence[np.ndarray]) -> float:
    if not values:
        return 0.0
    return float(max(float(np.max(np.abs(value))) for value in values))


def _spectral_interval_priorities(
    coarse: ForwardSimulationResult,
    probe: ForwardSimulationResult,
    coarse_grid: np.ndarray,
    *,
    cross_solver_probe: ForwardSimulationResult | None = None,
) -> list[dict[str, Any]]:
    """Compute deterministic priority metrics for every wavelength interval."""

    priorities: list[dict[str, Any]] = []
    for index in range(coarse_grid.size - 1):
        lower = float(coarse_grid[index])
        upper = float(coarse_grid[index + 1])
        midpoint_index = 2 * index + 1
        midpoint = float(probe.wavelengths_nm[midpoint_index])
        residual = 0.0
        curvature = 0.0
        extremum = False
        cross_disagreement = 0.0
        for channel_key, coarse_values in coarse.channels.items():
            probe_values = probe.channels[channel_key]
            for observable in ("R", "T", "A"):
                left = np.asarray(coarse_values[observable][index], dtype=np.float64)
                right = np.asarray(coarse_values[observable][index + 1], dtype=np.float64)
                middle = np.asarray(probe_values[observable][midpoint_index], dtype=np.float64)
                linear_middle = 0.5 * (left + right)
                residual = max(residual, float(abs(middle - linear_middle)))
                # The midpoint residual is the local second-difference signal;
                # the slope change against neighboring intervals is a separate
                # curvature proxy and remains meaningful on nonuniform grids.
                slope = (right - left) / max(upper - lower, np.finfo(float).eps)
                if index > 0:
                    previous = np.asarray(coarse_values[observable][index - 1], dtype=np.float64)
                    previous_width = lower - float(coarse_grid[index - 1])
                    previous_slope = (left - previous) / max(previous_width, np.finfo(float).eps)
                    curvature = max(curvature, _max_abs((slope - previous_slope,)))
                if index + 2 < coarse_grid.size:
                    following = np.asarray(coarse_values[observable][index + 2], dtype=np.float64)
                    following_width = float(coarse_grid[index + 2]) - upper
                    following_slope = (following - right) / max(following_width, np.finfo(float).eps)
                    curvature = max(curvature, _max_abs((following_slope - slope,)))
                if bool(np.any((middle - left) * (right - middle) < 0.0)):
                    extremum = True
                if cross_solver_probe is not None:
                    reference = cross_solver_probe.channels.get(channel_key)
                    if (
                        reference is not None
                        and observable in reference
                        and len(reference[observable]) > midpoint_index
                    ):
                        reference_middle = np.asarray(
                            reference[observable][midpoint_index], dtype=np.float64
                        )
                        cross_disagreement = max(
                            cross_disagreement,
                            float(abs(middle - reference_middle)),
                        )
        # The weights are fixed, positive, and dimensionless.  Residual and
        # curvature dominate; extrema and optional solver disagreement prevent
        # a sharp feature from losing a tie to a smooth interval.
        extremum_signal = float(extremum) * min(1.0, residual + curvature)
        priority = residual + 0.25 * curvature + 0.10 * extremum_signal + cross_disagreement
        priorities.append(
            {
                "axis": "wavelength",
                "index": int(index),
                "lower": lower,
                "upper": upper,
                "midpoint": midpoint,
                "residual": float(residual),
                "midpoint_interpolation_residual": float(residual),
                "curvature_proxy": float(curvature),
                "extremum_proximity": bool(extremum),
                "cross_solver_disagreement": float(cross_disagreement)
                if cross_solver_probe is not None
                else None,
                "priority": float(priority),
            }
        )
    return priorities


def _compare_angle_probe(
    coarse: ForwardSimulationResult,
    probe: ForwardSimulationResult,
    coarse_angles: np.ndarray,
    polarizations: Sequence[str],
) -> tuple[float, list[dict[str, Any]]]:
    """Return midpoint interpolation error and local angular priorities."""

    maximum = 0.0
    priorities: list[dict[str, Any]] = []
    for index in range(coarse_angles.size - 1):
        lower = float(coarse_angles[index])
        upper = float(coarse_angles[index + 1])
        midpoint = 0.5 * (lower + upper)
        residual = 0.0
        curvature = 0.0
        extremum = False
        for polarization in polarizations:
            left_values = coarse.channels[_channel_key(lower, polarization)]
            right_values = coarse.channels[_channel_key(upper, polarization)]
            middle_values = probe.channels[_channel_key(midpoint, polarization)]
            previous_values = (
                coarse.channels[_channel_key(float(coarse_angles[index - 1]), polarization)]
                if index > 0
                else None
            )
            following_values = (
                coarse.channels[_channel_key(float(coarse_angles[index + 2]), polarization)]
                if index + 2 < coarse_angles.size
                else None
            )
            for observable in ("R", "T", "A"):
                left = np.asarray(left_values[observable], dtype=np.float64)
                right = np.asarray(right_values[observable], dtype=np.float64)
                middle = np.asarray(middle_values[observable], dtype=np.float64)
                residual = max(residual, _max_abs((middle - 0.5 * (left + right),)))
                slope = (right - left) / max(upper - lower, np.finfo(float).eps)
                if previous_values is not None:
                    previous = np.asarray(previous_values[observable], dtype=np.float64)
                    previous_width = lower - float(coarse_angles[index - 1])
                    previous_slope = (left - previous) / max(
                        previous_width, np.finfo(float).eps
                    )
                    curvature = max(curvature, _max_abs((slope - previous_slope,)))
                if following_values is not None:
                    following = np.asarray(following_values[observable], dtype=np.float64)
                    following_width = float(coarse_angles[index + 2]) - upper
                    following_slope = (following - right) / max(
                        following_width, np.finfo(float).eps
                    )
                    curvature = max(curvature, _max_abs((following_slope - slope,)))
                if bool(np.any((middle - left) * (right - middle) < 0.0)):
                    extremum = True
        maximum = max(maximum, residual)
        extremum_signal = float(extremum) * min(1.0, residual + curvature)
        priority = residual + 0.25 * curvature + 0.10 * extremum_signal
        priorities.append(
            {
                "axis": "angle",
                "index": int(index),
                "lower": lower,
                "upper": upper,
                "midpoint": midpoint,
                "residual": float(residual),
                "midpoint_interpolation_residual": float(residual),
                "curvature_proxy": float(curvature),
                "extremum_proximity": bool(extremum),
                "cross_solver_disagreement": None,
                "priority": float(priority),
            }
        )
    return maximum, priorities


def _select_intervals(
    priorities: Sequence[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """Select high-priority intervals with a stable index tie-break."""

    candidates = [item for item in priorities if float(item["priority"]) > 0.0]
    ordered = sorted(candidates, key=lambda item: (-float(item["priority"]), int(item["index"])))
    return ordered[: int(limit)]


def _insert_midpoints(values: np.ndarray, selected: Sequence[Mapping[str, Any]]) -> np.ndarray:
    if not selected:
        return values.copy()
    midpoints = np.asarray([float(item["midpoint"]) for item in selected], dtype=np.float64)
    return np.asarray(np.sort(np.concatenate((values, midpoints))), dtype=np.float64)


def _worst_interval(
    spectral: Sequence[Mapping[str, Any]], angular: Sequence[Mapping[str, Any]]
) -> Dict[str, Any] | None:
    candidates = [dict(item) for item in (*spectral, *angular)]
    if not candidates:
        return None
    return dict(max(candidates, key=lambda item: (float(item["priority"]), -int(item["index"]))))


def _trigger_reasons(
    *,
    pointwise: float,
    integral: float,
    angular: float,
    settings: SpectralConvergenceSettings,
    selected: Sequence[Mapping[str, Any]],
) -> list[str]:
    reasons: list[str] = []
    if pointwise > settings.max_pointwise_deviation:
        reasons.append("spectral_pointwise_threshold")
    if integral > settings.max_integral_deviation:
        reasons.append("spectral_integral_threshold")
    if angular > settings.max_angular_deviation:
        reasons.append("angular_pointwise_threshold")
    for item in selected:
        if float(item["midpoint_interpolation_residual"]) > 0.0:
            reasons.append(f"{item['axis']}_midpoint_residual")
        if float(item["curvature_proxy"]) > 0.0:
            reasons.append(f"{item['axis']}_curvature")
        if bool(item["extremum_proximity"]):
            reasons.append(f"{item['axis']}_extremum_proximity")
    # Keep the ledger compact while preserving deterministic insertion order.
    return list(dict.fromkeys(reasons)) or ["error_threshold_met"]


def audit_spectral_convergence(
    workbench: TMMWorkbench,
    task: SimulationTask,
    settings: SpectralConvergenceSettings | None = None,
    initial_result: ForwardSimulationResult | None = None,
    *,
    cross_solver_probe: ForwardSimulationResult | None = None,
) -> SpectralConvergenceOutcome:
    """Audit local spectral and angular convergence on a private grid.

    Every round evaluates all candidate midpoints to measure local residuals,
    then commits only the highest-priority intervals.  Candidate evaluation is
    verifier-internal; ``final_result`` is always the original declared-grid
    result, while ``verification_result`` is the adaptive private result.
    """

    settings = settings or SpectralConvergenceSettings()
    settings.validate()
    declared_result = initial_result or workbench.simulate(task)
    declared_grid = np.asarray(declared_result.wavelengths_nm, dtype=np.float64).copy()
    declared_angles = np.asarray(task.illumination.angles_deg, dtype=np.float64).copy()
    internal_angles = tuple(sorted({float(angle) for angle in task.illumination.angles_deg}))
    current_task = replace(
        task,
        illumination=IlluminationSpec(
            angles_deg=internal_angles,
            polarizations=task.illumination.polarizations,
        ),
    )
    current = declared_result
    rounds: List[Dict[str, Any]] = []
    passed = False
    status = "spectral_convergence_failure"
    refinement_status = "max_depth_reached"
    worst_unresolved: Dict[str, Any] | None = None

    for index in range(settings.max_refinements):
        coarse_grid = np.asarray(current.wavelengths_nm, dtype=np.float64)
        coarse_angles = np.asarray(current_task.illumination.angles_deg, dtype=np.float64)
        spectral_probe: ForwardSimulationResult | None = None
        angular_probe: ForwardSimulationResult | None = None
        spectral_priorities: list[dict[str, Any]] = []
        angular_priorities: list[dict[str, Any]] = []
        spectral_detail: Dict[str, Any] = {}
        spectral_pointwise = 0.0
        spectral_integral = 0.0
        angular_pointwise = 0.0

        if coarse_grid.size > 1:
            candidate_grid = np.empty(coarse_grid.size * 2 - 1, dtype=np.float64)
            candidate_grid[0::2] = coarse_grid
            candidate_grid[1::2] = 0.5 * (coarse_grid[:-1] + coarse_grid[1:])
            probe_task = replace(
                current_task,
                spectrum=SpectralGrid(values_nm=tuple(candidate_grid.tolist())),
            )
            spectral_probe = workbench.simulate(probe_task)
            spectral_pointwise, spectral_integral, spectral_detail = _compare_levels(
                current, spectral_probe
            )
            spectral_priorities = _spectral_interval_priorities(
                current,
                spectral_probe,
                coarse_grid,
                cross_solver_probe=cross_solver_probe,
            )

        if coarse_angles.size > 1:
            candidate_angles = np.empty(coarse_angles.size * 2 - 1, dtype=np.float64)
            candidate_angles[0::2] = coarse_angles
            candidate_angles[1::2] = 0.5 * (coarse_angles[:-1] + coarse_angles[1:])
            probe_task = replace(
                current_task,
                illumination=IlluminationSpec(
                    angles_deg=tuple(candidate_angles.tolist()),
                    polarizations=current_task.illumination.polarizations,
                ),
            )
            angular_probe = workbench.simulate(probe_task)
            angular_pointwise, angular_priorities = _compare_angle_probe(
                current,
                angular_probe,
                coarse_angles,
                current_task.illumination.polarizations,
            )

        round_passed = bool(
            spectral_pointwise <= settings.max_pointwise_deviation
            and spectral_integral <= settings.max_integral_deviation
            and angular_pointwise <= settings.max_angular_deviation
        )
        selected_spectral = _select_intervals(
            spectral_priorities, settings.max_intervals_per_round
        )
        selected_angular = _select_intervals(
            angular_priorities, settings.max_angle_intervals_per_round
        )
        selected = [*selected_spectral, *selected_angular]
        worst_unresolved = _worst_interval(spectral_priorities, angular_priorities)

        if round_passed:
            passed = True
            status = "passed"
            refinement_status = "converged"
            rounds.append(
                {
                    "round": index + 1,
                    "coarse_points": int(coarse_grid.size),
                    "coarse_angle_points": int(coarse_angles.size),
                    "candidate_midpoint_points": int(len(spectral_priorities)),
                    "candidate_midpoint_angle_points": int(len(angular_priorities)),
                    "refined_points": int(coarse_grid.size),
                    "refined_angle_points": int(coarse_angles.size),
                    "points_added": {"wavelengths": 0, "angles": 0},
                    "max_pointwise_deviation": spectral_pointwise,
                    "max_integral_deviation": spectral_integral,
                    "max_angular_deviation": angular_pointwise,
                    "passed": True,
                    "trigger_reason": "error_threshold_met",
                    "trigger_reasons": ["error_threshold_met"],
                    "selected_intervals": [],
                    "channels": spectral_detail,
                    "angular_intervals": angular_priorities,
                }
            )
            worst_unresolved = None
            break

        available_wavelengths = max(0, settings.maximum_points - coarse_grid.size)
        available_angles = max(0, settings.maximum_angle_points - coarse_angles.size)
        selected_spectral = selected_spectral[:available_wavelengths]
        selected_angular = selected_angular[:available_angles]
        selected = [*selected_spectral, *selected_angular]

        budget_blocked = (
            bool(spectral_priorities)
            and available_wavelengths == 0
        ) or (
            bool(angular_priorities)
            and available_angles == 0
        )
        if not selected:
            refinement_status = "budget_exhausted" if budget_blocked else "max_depth_reached"
            trigger = "point_budget_exhausted" if budget_blocked else "max_depth_reached"
            rounds.append(
                {
                    "round": index + 1,
                    "coarse_points": int(coarse_grid.size),
                    "coarse_angle_points": int(coarse_angles.size),
                    "candidate_midpoint_points": int(len(spectral_priorities)),
                    "candidate_midpoint_angle_points": int(len(angular_priorities)),
                    "refined_points": int(coarse_grid.size),
                    "refined_angle_points": int(coarse_angles.size),
                    "points_added": {"wavelengths": 0, "angles": 0},
                    "max_pointwise_deviation": spectral_pointwise,
                    "max_integral_deviation": spectral_integral,
                    "max_angular_deviation": angular_pointwise,
                    "passed": False,
                    "trigger_reason": trigger,
                    "trigger_reasons": [trigger],
                    "selected_intervals": [],
                    "channels": spectral_detail,
                    "angular_intervals": angular_priorities,
                }
            )
            break

        next_grid = _insert_midpoints(coarse_grid, selected_spectral)
        next_angles = _insert_midpoints(coarse_angles, selected_angular)
        refined_task = replace(
            current_task,
            spectrum=SpectralGrid(values_nm=tuple(next_grid.tolist())),
            illumination=IlluminationSpec(
                angles_deg=tuple(next_angles.tolist()),
                polarizations=current_task.illumination.polarizations,
            ),
        )
        refined = workbench.simulate(refined_task)
        rounds.append(
            {
                "round": index + 1,
                "coarse_points": int(coarse_grid.size),
                "coarse_angle_points": int(coarse_angles.size),
                "candidate_midpoint_points": int(len(spectral_priorities)),
                "candidate_midpoint_angle_points": int(len(angular_priorities)),
                "refined_points": int(next_grid.size),
                "refined_angle_points": int(next_angles.size),
                "points_added": {
                    "wavelengths": int(next_grid.size - coarse_grid.size),
                    "angles": int(next_angles.size - coarse_angles.size),
                },
                "max_pointwise_deviation": spectral_pointwise,
                "max_integral_deviation": spectral_integral,
                "max_angular_deviation": angular_pointwise,
                "passed": False,
                "trigger_reason": _trigger_reasons(
                    pointwise=spectral_pointwise,
                    integral=spectral_integral,
                    angular=angular_pointwise,
                    settings=settings,
                    selected=selected,
                )[0],
                "trigger_reasons": _trigger_reasons(
                    pointwise=spectral_pointwise,
                    integral=spectral_integral,
                    angular=angular_pointwise,
                    settings=settings,
                    selected=selected,
                ),
                "selected_intervals": [dict(item) for item in selected_spectral],
                "angular_intervals": [dict(item) for item in selected_angular],
                "channels": spectral_detail,
            }
        )
        current_task, current = refined_task, refined

    if not passed and refinement_status == "max_depth_reached":
        # A non-converged run that used its final allowed round is distinct
        # from a point budget failure, even when there are still intervals.
        if len(rounds) >= settings.max_refinements:
            refinement_status = "max_depth_reached"
    if not passed and refinement_status == "max_depth_reached" and not rounds:
        refinement_status = "max_depth_reached"

    verification_grid = np.asarray(current.wavelengths_nm, dtype=np.float64)
    verification_angles = np.asarray(current_task.illumination.angles_deg, dtype=np.float64)
    return SpectralConvergenceOutcome(
        status=status,
        passed=passed,
        rounds=rounds,
        settings=asdict(settings),
        final_points=int(verification_grid.size),
        final_result=declared_result,
        verification_result=current,
        declared_grid_sha256=_grid_sha256(declared_grid),
        verification_grid_sha256=_grid_sha256(verification_grid),
        declared_angle_grid_sha256=_grid_sha256(declared_angles),
        verification_angle_grid_sha256=_grid_sha256(verification_angles),
        refinement_status=refinement_status,
        worst_unresolved_interval=None if passed else worst_unresolved,
    )


__all__ = [
    "SpectralConvergenceOutcome",
    "SpectralConvergenceSettings",
    "audit_spectral_convergence",
]
