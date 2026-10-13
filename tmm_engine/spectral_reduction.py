"""Explicit wavelength reductions shared by metric reports."""

from __future__ import annotations

from typing import Literal

import numpy as np

MeanReduction = Literal["sample_mean", "wavelength_integral_mean"]


def reduce_spectral_mean(wavelengths_nm, values, reduction: MeanReduction) -> float:
    wavelengths = np.asarray(wavelengths_nm, dtype=np.float64)
    vector = np.asarray(values, dtype=np.float64)
    if wavelengths.ndim != 1 or vector.shape != wavelengths.shape:
        raise ValueError("spectral mean requires matching one-dimensional arrays")
    if not vector.size or not np.all(np.isfinite(vector)) or not np.all(np.isfinite(wavelengths)):
        raise ValueError("spectral mean requires finite samples")
    if reduction not in ("sample_mean", "wavelength_integral_mean"):
        raise ValueError(f"unsupported mean reduction: {reduction}")
    if reduction == "sample_mean" or vector.size == 1:
        return float(np.mean(vector))
    if np.any(np.diff(wavelengths) <= 0):
        raise ValueError("wavelength_integral_mean requires strictly increasing wavelengths")
    return float(np.trapezoid(vector, wavelengths) / (wavelengths[-1] - wavelengths[0]))
