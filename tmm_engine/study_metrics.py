"""Deterministic scalar metrics shared by sweeps and scientific analyses."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def channel_key(angle_deg: float, polarization: str) -> str:
    return "angle=%g|pol=%s" % (float(angle_deg), str(polarization))


def evaluate_metric(result: Any, metric: Any) -> float:
    """Evaluate a public MetricContract against a forward result or JSON form."""

    if hasattr(metric, "model_dump"):
        spec = metric.model_dump(mode="python")
    else:
        spec = dict(metric)
    if hasattr(result, "wavelengths_nm"):
        wavelengths = np.asarray(result.wavelengths_nm, dtype=np.float64)
        channels = result.channels
    else:
        wavelengths = np.asarray(result["wavelengths_nm"], dtype=np.float64)
        channels = result["channels"]
    key = channel_key(spec.get("angle_deg", 0.0), spec.get("polarization", "unpolarized"))
    if key not in channels:
        raise KeyError(f"metric channel is not present: {key}")
    observable = str(spec["observable"])
    if observable not in channels[key]:
        raise KeyError(f"metric observable is not present: {observable}")
    values = np.asarray(channels[key][observable], dtype=np.float64)
    mask = np.ones(wavelengths.shape, dtype=bool)
    lo = spec.get("wavelength_min_nm")
    hi = spec.get("wavelength_max_nm")
    if lo is not None:
        mask &= wavelengths >= float(lo)
    if hi is not None:
        mask &= wavelengths <= float(hi)
    if not np.any(mask):
        raise ValueError("metric wavelength selection does not overlap the simulation grid")
    selected_wavelengths = wavelengths[mask]
    selected = values[mask]
    aggregation = str(spec.get("aggregation", "mean"))
    if aggregation == "mean":
        return float(np.mean(selected))
    if aggregation == "min":
        return float(np.min(selected))
    if aggregation == "max":
        return float(np.max(selected))
    if aggregation == "worst_case":
        return float(
            np.min(selected)
            if spec.get("threshold_direction", "at_least") == "at_least"
            else np.max(selected)
        )
    if aggregation == "value_at_wavelength":
        wavelength = float(spec["wavelength_nm"])
        return float(np.interp(wavelength, wavelengths, values))
    if aggregation == "threshold_band_width":
        threshold = float(spec["threshold"])
        passing = (
            selected >= threshold
            if spec.get("threshold_direction", "at_least") == "at_least"
            else selected <= threshold
        )
        if not np.any(passing):
            return 0.0
        # Sum contiguous sampled intervals.  This avoids claiming an unobserved
        # interpolation through a failing gap.
        interval_mask = passing[:-1] & passing[1:]
        return float(np.sum(np.diff(selected_wavelengths)[interval_mask]))
    raise ValueError(f"unsupported metric aggregation: {aggregation}")


def metric_constraint_passes(value: float, target: Mapping[str, Any]) -> bool:
    constraint = str(target.get("constraint", "at_least"))
    threshold = float(target["value"])
    if constraint == "at_least":
        return float(value) >= threshold
    if constraint == "at_most":
        return float(value) <= threshold
    raise ValueError("constraint must be at_least or at_most")


__all__ = ["channel_key", "evaluate_metric", "metric_constraint_passes"]
