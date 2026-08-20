"""Deterministic analysis helpers for multilayer and 1D-PhC experiments."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .schemas import SimulationTask

_SPEED_OF_LIGHT_M_PER_S = 299_792_458.0


def phase_dispersion_from_amplitude(
    wavelengths_nm: Sequence[float], amplitude: np.ndarray
) -> Dict[str, np.ndarray]:
    """Return unwrapped phase, group delay and GDD for a complex amplitude.

    The engine convention is ``exp(i k z - i omega t)`` and group delay is
    ``d phase / d omega``.  The final axis of ``amplitude`` must match the
    strictly increasing wavelength grid; leading axes (for example s/p) are
    preserved.
    """

    wavelengths = np.asarray(wavelengths_nm, dtype=np.float64)
    values = np.asarray(amplitude, dtype=np.complex128)
    if wavelengths.ndim != 1 or wavelengths.size < 3:
        raise ValueError("phase dispersion requires at least three wavelengths")
    if values.shape[-1] != wavelengths.size:
        raise ValueError("amplitude's final axis must match wavelengths")
    if np.any(np.diff(wavelengths) <= 0):
        raise ValueError("wavelengths must be strictly increasing")
    omega = 2.0 * np.pi * _SPEED_OF_LIGHT_M_PER_S / (wavelengths * 1e-9)
    phase = np.unwrap(np.angle(values), axis=-1)
    edge_order = 2 if wavelengths.size >= 3 else 1
    group_delay_s = np.gradient(phase, omega, axis=-1, edge_order=edge_order)
    gdd_s2 = np.gradient(group_delay_s, omega, axis=-1, edge_order=edge_order)
    return {
        "phase_rad": phase,
        "group_delay_fs": group_delay_s * 1e15,
        "gdd_fs2": gdd_s2 * 1e30,
    }


@dataclass(frozen=True)
class SpectralBand:
    start_nm: float
    stop_nm: float
    center_nm: float
    width_nm: float
    peak_value: float
    mean_value: float


def find_threshold_bands(
    wavelengths_nm: Sequence[float],
    values: Sequence[float],
    *,
    threshold: float,
    above: bool = True,
    min_width_nm: float = 0.0,
) -> List[SpectralBand]:
    wavelengths = np.asarray(wavelengths_nm, dtype=np.float64)
    spectrum = np.asarray(values, dtype=np.float64)
    if wavelengths.shape != spectrum.shape or wavelengths.ndim != 1:
        raise ValueError("wavelengths and values must have matching 1D shapes")
    mask = spectrum >= float(threshold) if above else spectrum <= float(threshold)
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    edges = np.diff(padded)
    starts = np.where(edges == 1)[0]
    stops = np.where(edges == -1)[0] - 1
    bands: List[SpectralBand] = []
    for start, stop in zip(starts, stops):
        width = float(wavelengths[stop] - wavelengths[start])
        if width < float(min_width_nm):
            continue
        segment = spectrum[start : stop + 1]
        peak_index = int(np.argmax(segment) if above else np.argmin(segment))
        bands.append(
            SpectralBand(
                start_nm=float(wavelengths[start]),
                stop_nm=float(wavelengths[stop]),
                center_nm=float(wavelengths[start + peak_index]),
                width_nm=width,
                peak_value=float(segment[peak_index]),
                mean_value=float(np.mean(segment)),
            )
        )
    return bands


def hemispherical_average(
    angles_deg: Sequence[float],
    spectra: Sequence[Sequence[float]],
) -> np.ndarray:
    """Integrate an azimuthally symmetric quantity over a hemisphere.

    The normalized average is ``2 integral f(theta) sin(theta) cos(theta)dtheta``.
    Endpoints at 0 and 90 degrees are inserted by constant extension when absent.
    """

    angles = np.asarray(angles_deg, dtype=np.float64)
    values = np.asarray(spectra, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != angles.size:
        raise ValueError("spectra must have shape [angles,wavelengths]")
    order = np.argsort(angles)
    angles = angles[order]
    values = values[order]
    if angles[0] > 0:
        angles = np.concatenate([[0.0], angles])
        values = np.concatenate([values[:1], values], axis=0)
    if angles[-1] < 90:
        angles = np.concatenate([angles, [90.0]])
        values = np.concatenate([values, values[-1:]], axis=0)
    theta = np.deg2rad(angles)
    integrand = 2.0 * values * np.sin(theta)[:, None] * np.cos(theta)[:, None]
    return np.trapezoid(integrand, theta, axis=0)


def spectrum_similarity(reference: Sequence[float], prediction: Sequence[float]) -> Dict[str, float]:
    ref = np.asarray(reference, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    if ref.shape != pred.shape or ref.ndim != 1:
        raise ValueError("spectra must be matching 1D arrays")
    mae = float(np.mean(np.abs(ref - pred)))
    rmse = float(np.sqrt(np.mean((ref - pred) ** 2)))
    if np.std(ref) <= 1e-15 or np.std(pred) <= 1e-15:
        correlation = 1.0 if np.allclose(ref, pred) else 0.0
    else:
        correlation = float(np.corrcoef(ref, pred)[0, 1])
    return {"mae": mae, "rmse": rmse, "pearson_correlation": correlation}


def thickness_tolerance_monte_carlo(
    workbench: Any,
    task: SimulationTask,
    *,
    sigma_nm: float,
    samples: int = 100,
    seed: int = 42,
    angle_deg: float = 0.0,
    polarization: str = "unpolarized",
) -> Dict[str, Any]:
    if float(sigma_nm) < 0 or int(samples) < 1:
        raise ValueError("sigma_nm must be non-negative and samples positive")
    rng = np.random.default_rng(int(seed))
    spectra = []
    thickness_draws = []
    for _ in range(int(samples)):
        layers = []
        draw = []
        for layer in task.stack.layers:
            value = max(0.1, float(layer.thickness_nm) + float(rng.normal(0.0, sigma_nm)))
            draw.append(value)
            layers.append(replace(layer, thickness_nm=value))
        perturbed = replace(task, stack=replace(task.stack, layers=tuple(layers)))
        result = workbench.simulate(perturbed)
        spectra.append(result.channel(angle_deg, polarization)["R"])
        thickness_draws.append(draw)
    arr = np.asarray(spectra, dtype=np.float64)
    return {
        "samples": int(samples),
        "sigma_nm": float(sigma_nm),
        "seed": int(seed),
        "thickness_draws_nm": np.asarray(thickness_draws, dtype=np.float64),
        "R_mean": np.mean(arr, axis=0),
        "R_std": np.std(arr, axis=0),
        "R_p05": np.quantile(arr, 0.05, axis=0),
        "R_p95": np.quantile(arr, 0.95, axis=0),
    }


def bloch_trace_bilayer_normal_incidence(
    n_a: Sequence[complex],
    d_a_nm: float,
    n_b: Sequence[complex],
    d_b_nm: float,
    wavelengths_nm: Sequence[float],
) -> Dict[str, np.ndarray]:
    """Return ``cos(K Lambda)=Tr(M_cell)/2`` for a bilayer unit cell."""

    wavelengths = np.asarray(wavelengths_nm, dtype=np.float64)
    na = np.asarray(n_a, dtype=np.complex128)
    nb = np.asarray(n_b, dtype=np.complex128)
    if wavelengths.shape != na.shape or wavelengths.shape != nb.shape:
        raise ValueError("n_a, n_b, and wavelengths must have identical shapes")

    def matrix(n: np.ndarray, thickness_nm: float) -> Tuple[np.ndarray, ...]:
        delta = 2.0 * np.pi * n * float(thickness_nm) / wavelengths
        c = np.cos(delta)
        s = 1j * np.sin(delta)
        return c, s / n, s * n, c

    a11, a12, a21, a22 = matrix(na, d_a_nm)
    b11, b12, b21, b22 = matrix(nb, d_b_nm)
    m11 = a11 * b11 + a12 * b21
    m22 = a21 * b12 + a22 * b22
    trace_half = 0.5 * (m11 + m22)
    bloch_k = np.arccos(trace_half) / (float(d_a_nm) + float(d_b_nm))
    forbidden = np.abs(np.real(trace_half)) > 1.0
    return {"trace_half": trace_half, "bloch_k_per_nm": bloch_k, "forbidden_band_mask": forbidden}


__all__ = [
    "SpectralBand",
    "bloch_trace_bilayer_normal_incidence",
    "find_threshold_bands",
    "hemispherical_average",
    "phase_dispersion_from_amplitude",
    "spectrum_similarity",
    "thickness_tolerance_monte_carlo",
]
