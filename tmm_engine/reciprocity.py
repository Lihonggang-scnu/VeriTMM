"""Lorentz reciprocity check for reciprocal planar multilayers (opt-in).

A linear, isotropic, non-magneto-optic stack exchanges its power transmittance
with the reversed stack under conservation of the in-plane wave vector:
``T_forward(theta, pol) == T_backward(theta_b, pol)`` with
``sin(theta_b) = n_incident(theta) * sin(theta) / n_exit`` — i.e. the incident
and exit media swap, the layer order reverses, and the incidence angle is
mirrored about the conserved ``kx`` (for identical surroundings this reduces
to the same numeric angle, the ``180° - theta`` mirror in the global frame).

This is an internal consistency property of the solver's math, so the check
runs the same declared solver in both directions; a failure is a numerical or
implementation defect, never a capability boundary.  The check is opt-in
(``AcceptanceSettings.require_reciprocity``) because it roughly doubles the
simulations of the acceptance pass on a wavelength subsample; CI anchors run
it unconditionally.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from .schemas import IlluminationSpec, SimulationTask, SpectralGrid, StackSpec
from .workbench import _channel_key

RECIPROCITY_TOLERANCE = 1e-10
DEFAULT_MAX_SAMPLES = 11

_ABSORPTION_EPS = 1e-12
_TINY_TRANSMITTANCE = 1e-12


def _transmissions(workbench: Any, task: SimulationTask) -> Dict[str, np.ndarray]:
    """Simulate one single-illumination task and return its T per channel."""

    result = workbench.simulate(task)
    return {
        key: np.asarray(values["T"], dtype=np.float64)
        for key, values in result.channels.items()
    }


def _wavelength_subset(wavelengths_nm: np.ndarray, max_samples: int) -> np.ndarray:
    """Bounded uniform subsample of the declared grid."""

    count = int(min(max(1, int(max_samples)), wavelengths_nm.size))
    if count >= wavelengths_nm.size:
        return np.asarray(wavelengths_nm, dtype=np.float64)
    indices = np.unique(
        np.round(np.linspace(0, wavelengths_nm.size - 1, count)).astype(int)
    )
    return np.asarray(wavelengths_nm, dtype=np.float64)[indices]


def _reversed_stack(stack: StackSpec) -> StackSpec:
    """Reverse the stack with the public constructors so validation applies."""

    reversed_stack = StackSpec(
        layers=tuple(reversed(stack.layers)),
        incident=stack.exit,
        exit=stack.incident,
        name=stack.name,
    )
    reversed_stack.validate()
    return reversed_stack


def check_reciprocity(
    workbench: Any,
    stack: StackSpec,
    wavelengths_nm: np.ndarray,
    angles_deg: Sequence[float] = (0.0,),
    polarizations: Sequence[str] = ("s",),
    *,
    solver: str = "smatrix",
    max_samples: int = DEFAULT_MAX_SAMPLES,
    tolerance: float = RECIPROCITY_TOLERANCE,
) -> Dict[str, Any]:
    """Compare forward and reversed power transmittance on a bounded subset.

    Returns ``{status: passed|failed|unavailable, max_relative_deviation,
    worst_wavelength_nm, reason}``.  ``unavailable`` is used honestly — for
    incoherent stacks (outside the certified check in this release), absorbing
    incident/exit media (the mirrored angle is ill-defined for complex
    indices), a mirrored angle beyond the exit medium's light cone, or
    non-finite solver output — and is never a disguised pass.
    """

    wavelengths = _wavelength_subset(wavelengths_nm, max_samples)
    probe_stack = _reversed_stack(stack)
    forward_probe = SimulationTask(
        stack=stack,
        spectrum=SpectralGrid(values_nm=tuple(float(x) for x in wavelengths)),
        illumination=IlluminationSpec(
            tuple(float(a) for a in angles_deg),
            tuple(str(p) for p in polarizations),
        ),
        solver=str(solver),
        requested_outputs=("R", "T", "A"),
    )
    backward_probe = SimulationTask(
        stack=probe_stack,
        spectrum=forward_probe.spectrum,
        illumination=forward_probe.illumination,
        solver=str(solver),
        requested_outputs=("R", "T", "A"),
    )
    unavailable: Dict[str, Any] = {
        "status": "unavailable",
        "max_relative_deviation": None,
        "worst_wavelength_nm": None,
        "reason": None,
    }
    if stack.has_incoherent_layers or probe_stack.has_incoherent_layers:
        unavailable["reason"] = (
            "mixed-coherence stacks are outside the certified reciprocity "
            "check in this release"
        )
        return unavailable
    try:
        forward_media, _, _ = workbench._resolve_stack(forward_probe)
        backward_media, _, _ = workbench._resolve_stack(backward_probe)
    except Exception as exc:
        unavailable["reason"] = f"stack resolution failed: {exc}"
        return unavailable
    forward_incident = forward_media[0]
    backward_incident = backward_media[0]
    forward_exit = forward_media[-1]
    if (
        np.max(np.abs(np.imag(forward_incident))) > _ABSORPTION_EPS
        or np.max(np.abs(np.imag(backward_incident))) > _ABSORPTION_EPS
        or np.max(np.abs(np.imag(forward_exit))) > _ABSORPTION_EPS
    ):
        unavailable["reason"] = (
            "reciprocity mirror angle is ill-defined for absorbing incident "
            "or exit media"
        )
        return unavailable

    maximum_deviation = 0.0
    worst_wavelength: Optional[float] = None
    for wavelength_index, wavelength in enumerate(wavelengths):
        forward_task = SimulationTask(
            stack=stack,
            spectrum=SpectralGrid(values_nm=(float(wavelength),)),
            illumination=forward_probe.illumination,
            solver=str(solver),
            requested_outputs=("R", "T", "A"),
        )
        forward_transmissions = _transmissions(workbench, forward_task)
        for angle in angles_deg:
            angle = float(angle)
            sin_forward = np.sin(np.radians(angle))
            sin_backward = (
                float(np.real(forward_incident[wavelength_index])) * sin_forward
            ) / float(np.real(backward_incident[wavelength_index]))
            if abs(sin_backward) > 1.0 + _ABSORPTION_EPS:
                unavailable["reason"] = (
                    "mirrored incidence %.4f deg is beyond the exit medium's "
                    "light cone at %.4f nm; the reciprocal partner is not a "
                    "propagating wave" % (angle, float(wavelength))
                )
                return unavailable
            backward_angle = float(np.degrees(np.arcsin(sin_backward)))
            backward_task = SimulationTask(
                stack=probe_stack,
                spectrum=SpectralGrid(values_nm=(float(wavelength),)),
                illumination=IlluminationSpec(
                    (backward_angle,), forward_probe.illumination.polarizations
                ),
                solver=str(solver),
                requested_outputs=("R", "T", "A"),
            )
            backward_transmissions = _transmissions(workbench, backward_task)
            for polarization in polarizations:
                polarization = str(polarization)
                forward_key = _channel_key(angle, polarization)
                backward_key = _channel_key(backward_angle, polarization)
                if forward_key not in forward_transmissions or (
                    backward_key not in backward_transmissions
                ):
                    unavailable["reason"] = (
                        "solver did not emit the requested channel at %.4f nm"
                        % float(wavelength)
                    )
                    return unavailable
                forward_t = float(forward_transmissions[forward_key][0])
                backward_t = float(backward_transmissions[backward_key][0])
                if not (
                    np.isfinite(forward_t)
                    and np.isfinite(backward_t)
                ):
                    unavailable["reason"] = (
                        "non-finite transmittance at %.4f nm" % float(wavelength)
                    )
                    return unavailable
                scale = max(forward_t, backward_t, _TINY_TRANSMITTANCE)
                deviation = abs(forward_t - backward_t) / scale
                if deviation > maximum_deviation:
                    maximum_deviation = deviation
                    worst_wavelength = float(wavelength)
    status = "passed" if maximum_deviation <= tolerance else "failed"
    return {
        "status": status,
        "max_relative_deviation": maximum_deviation,
        "worst_wavelength_nm": worst_wavelength,
        "reason": None if status == "passed" else (
            "max relative transmittance deviation %.3e exceeds tolerance %.3e"
            % (maximum_deviation, tolerance)
        ),
    }


__all__ = [
    "DEFAULT_MAX_SAMPLES",
    "RECIPROCITY_TOLERANCE",
    "check_reciprocity",
]
