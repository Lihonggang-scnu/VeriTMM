"""Analytic validation oracle suite — exact closed forms for CI anchors only.

This module implements reference electromagnetic results from first
principles (Fresnel interface coefficients composed through the Airy
formulas), deliberately **without** using any of the engine's matrix,
scattering, or referee code.  It exists to anchor the solvers and the
high-precision referee in ``tests/test_analytic_oracle.py``.

Positioning (frozen 1.1 plan): the oracle is a **CI validation** asset.  It
is never imported by the runtime acceptance path and never influences a
physics certificate; a runtime oracle contract (when to invoke analytic
checks inside certification) is explicitly out of scope for 1.1.

Everything is formulated in the tangential-field admittance framework, which
is convention-independent for power:

- s polarization: ``Y = n * cos(theta)``
- p polarization: ``Y = n / cos(theta)``
- interface amplitudes (tangential E): ``r = (Y_from - Y_to)/(Y_from + Y_to)``,
  ``t = 2 Y_from / (Y_from + Y_to)``
- power: ``R = |r|^2``, ``T = Re(Y_exit) / Re(Y_incident) * |t|^2``

Snell's law uses the forward-decaying branch of the refracted cosine
(principal square root), which consistently handles absorbing films and, at
the interface level, total internal reflection.
"""

from __future__ import annotations

import math
from typing import Literal, Tuple

import numpy as np

Polarization = Literal["s", "p"]


def _cos_from_sin(sin_theta: complex) -> complex:
    """Forward-decaying branch of the refracted cosine."""

    return np.sqrt(1.0 - sin_theta**2 + 0j)


def _admittance(n: complex, sin_theta: complex, polarization: str) -> complex:
    cos_theta = _cos_from_sin(sin_theta)
    if polarization == "s":
        return n * cos_theta
    if polarization == "p":
        return n / cos_theta
    raise ValueError(f"polarization must be 's' or 'p', got {polarization!r}")


def _interface(
    n_from: complex,
    n_to: complex,
    sin_from: complex,
    polarization: str,
) -> Tuple[complex, complex]:
    """Fresnel amplitude coefficients (tangential E) of one interface."""

    sin_to = n_from * sin_from / n_to
    y_from = _admittance(n_from, sin_from, polarization)
    y_to = _admittance(n_to, sin_to, polarization)
    r = (y_from - y_to) / (y_from + y_to)
    t = 2.0 * y_from / (y_from + y_to)
    return r, t


def interface_power_rt(
    n_from: float,
    n_to: float,
    angle_deg: float,
    polarization: Polarization,
) -> Tuple[float, float]:
    """Exact Fresnel power reflectance/transmittance of a bare interface.

    Both media must be lossless (real indices); the angle must be below the
    critical angle for total internal reflection.
    """

    sin_from = math.sin(math.radians(angle_deg))
    sin_to = n_from * sin_from / n_to
    if abs(sin_to) > 1.0:
        raise ValueError("angle exceeds the critical angle for total internal reflection")
    r, t = _interface(n_from, n_to, sin_from, polarization)
    y_from = _admittance(n_from, sin_from, polarization)
    y_to = _admittance(n_to, sin_to, polarization)
    reflectance = float(abs(r) ** 2)
    transmittance = float((y_to / y_from).real * abs(t) ** 2)
    return reflectance, transmittance


def single_film_power_rt(
    n_incident: float,
    n_film: complex,
    n_exit: float,
    thickness_nm: float,
    wavelength_nm: float,
    angle_deg: float,
    polarization: Polarization,
) -> Tuple[float, float]:
    """Exact Airy-formula power R/T of one film between two media.

    The incident and exit media must be lossless (real indices) and the angle
    below their critical angle; the film may be absorbing.  Energy conservation
    gives ``A = 1 - R - T >= 0``.
    """

    if n_incident.imag != 0 or n_exit.imag != 0:
        raise ValueError("incident and exit media must be lossless (real indices)")
    sin0 = math.sin(math.radians(angle_deg))
    sin2 = n_incident * sin0 / n_exit
    if abs(sin2) > 1.0:
        raise ValueError("angle exceeds the critical angle for total internal reflection")

    n0, n1, n2 = complex(n_incident), complex(n_film), complex(n_exit)
    sin1 = n0 * sin0 / n1

    r01, t01 = _interface(n0, n1, sin0, polarization)
    r12, t12 = _interface(n1, n2, sin1, polarization)

    beta = (2.0 * np.pi / wavelength_nm) * n1 * _cos_from_sin(sin1) * thickness_nm
    phase = np.exp(2j * beta)
    r = (r01 + r12 * phase) / (1.0 + r01 * r12 * phase)
    t = (t01 * t12 * np.exp(1j * beta)) / (1.0 + r01 * r12 * phase)

    y0 = _admittance(n0, sin0, polarization)
    y2 = _admittance(n2, sin2, polarization)
    reflectance = float(abs(r) ** 2)
    transmittance = float((y2 / y0).real * abs(t) ** 2)
    return reflectance, transmittance


def quarter_wave_stack_reflectance_at_design(
    n_incident: float,
    n_high: float,
    n_low: float,
    n_exit: float,
    periods: int,
    design_wavelength_nm: float,
) -> float:
    """Exact reflectance of ``(H L)^periods`` at the quarter-wave design wavelength.

    At the design wavelength every layer is a quarter wave, so the pair
    characteristic matrix is diagonal and the input admittance is the closed
    form ``n_exit * (n_high / n_low)^(2 * periods)`` (high-index layer facing
    the incident medium).
    """

    if periods < 1:
        raise ValueError("periods must be a positive integer")
    admittance_in = n_exit * (n_high / n_low) ** (2 * periods)
    return float(
        ((n_incident - admittance_in) / (n_incident + admittance_in)) ** 2
    )


def quarter_wave_stack_band_edges(
    n_high: float,
    n_low: float,
    design_wavelength_nm: float,
) -> Tuple[float, float]:
    """Exact stopband edges of an infinite quarter-wave stack.

    For equal phase thickness ``delta = (pi/2) * design/lambda``, the
    dispersion relation ``cos(K*Lambda) = cos^2(delta) - M sin^2(delta)`` with
    ``M = (n_high^2 + n_low^2) / (2 n_high n_low)`` gives the first stopband
    ``delta in [delta_e, pi - delta_e]`` with ``sin^2(delta_e) = 2/(M+1)``.
    """

    if n_high <= n_low:
        raise ValueError("n_high must exceed n_low for a quarter-wave stack")
    m_ratio = (n_high**2 + n_low**2) / (2.0 * n_high * n_low)
    delta_edge = math.asin(math.sqrt(2.0 / (m_ratio + 1.0)))
    short_edge = design_wavelength_nm * (math.pi / 2.0) / (math.pi - delta_edge)
    long_edge = design_wavelength_nm * (math.pi / 2.0) / delta_edge
    return short_edge, long_edge


def fabry_perot_resonance_wavelengths(
    n_film: float,
    n_surrounding: float,
    thickness_nm: float,
    angle_deg: float,
    orders: Tuple[int, ...] = (1, 2, 3),
) -> Tuple[float, ...]:
    """Resonance wavelengths of a lossless Fabry-Perot cavity.

    For a film of index ``n_film`` between two identical surrounding media of
    index ``n_surrounding`` the Airy formula gives exactly ``T = 1`` when the
    round-trip phase is a multiple of ``2*pi``:
    ``lambda_m = 2 * n_film * d * cos(theta_film) / m`` with
    ``sin(theta_film) = n_surrounding * sin(theta_0) / n_film``.
    """

    sin_film = n_surrounding * math.sin(math.radians(angle_deg)) / n_film
    if abs(sin_film) > 1.0:
        raise ValueError("angle exceeds the critical angle inside the film")
    cos_film = _cos_from_sin(sin_film).real
    return tuple(
        2.0 * n_film * thickness_nm * cos_film / order for order in orders
    )


__all__ = [
    "fabry_perot_resonance_wavelengths",
    "interface_power_rt",
    "quarter_wave_stack_band_edges",
    "quarter_wave_stack_reflectance_at_design",
    "single_film_power_rt",
]
