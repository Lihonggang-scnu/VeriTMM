"""Published multilayer cases used as trend-level physics acceptance tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .schemas import LayerSpec, MediumSpec, SimulationTask, SpectralGrid, StackSpec

PMC9147317_REFERENCE = {
    "title": "Distributed Bragg Reflectors Employed in Sensors and Filters Based on Cavity-Mode Spectral-Domain Resonances",
    "doi": "10.3390/s22103627",
    "pmcid": "PMC9147317",
    "reported_stopband_nm": [590.0, 870.0],
    "reported_cavity_dip_nm": 639.1,
    "reported_minimum_field_enhancement": 145.0,
    "cavity_thickness_nm": 600.0,
}


def pmc9147317_stack() -> StackSpec:
    """Two measured six-bilayer DBRs separated by a 600 nm air cavity."""

    tio2_nm = [87.7, 79.1, 77.3, 80.7, 80.9, 76.9]
    sio2_nm = [120.2, 101.8, 109.2, 108.0, 127.3, 125.0]
    first: List[LayerSpec] = []
    for high, low in zip(tio2_nm, sio2_nm):
        first.extend((LayerSpec("tio2", high), LayerSpec("sio2", low)))
    first.append(LayerSpec("tio2", 64.4, label="termination"))
    layers = first + [
        LayerSpec(None, 600.0, constant_n=1.0, optimizable=False, label="air_cavity")
    ] + list(reversed(first))
    # The paper uses float-glass substrates and measured dispersions.  The
    # validation deliberately uses the local TiO2/SiO2 datasets and a constant
    # glass approximation; acceptance is trend-level rather than curve fitting.
    return StackSpec(
        layers=tuple(layers),
        incident=MediumSpec(constant_n=1.52),
        exit=MediumSpec(constant_n=1.52),
        name="pmc9147317_two_dbr_air_cavity",
    )


def _rolling_max(values: np.ndarray, half_window: int) -> np.ndarray:
    return np.asarray(
        [
            np.max(values[max(0, i - half_window) : min(values.size, i + half_window + 1)])
            for i in range(values.size)
        ],
        dtype=np.float64,
    )


def _largest_band(wavelengths: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    transitions = np.diff(np.concatenate([[False], mask, [False]]).astype(np.int8))
    starts = np.where(transitions == 1)[0]
    stops = np.where(transitions == -1)[0] - 1
    if starts.size == 0:
        raise RuntimeError("no simulated stopband detected")
    index = int(np.argmax(wavelengths[stops] - wavelengths[starts]))
    return float(wavelengths[starts[index]]), float(wavelengths[stops[index]])


def _interval_iou(first: Tuple[float, float], second: Tuple[float, float]) -> float:
    intersection = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    union = max(first[1], second[1]) - min(first[0], second[0])
    return intersection / union if union > 0 else 0.0


def validate_pmc9147317(workbench: Any, output_dir: str | Path | None = None) -> Dict[str, Any]:
    task = SimulationTask(
        stack=pmc9147317_stack(),
        spectrum=SpectralGrid(start_nm=500.0, stop_nm=900.0, points=1601),
        solver="smatrix",
    )
    result = workbench.simulate(task)
    wavelengths = result.wavelengths_nm
    reflectance = np.asarray(result.channel()["R"], dtype=np.float64)

    # A 10 nm rolling envelope ignores the deliberately narrow cavity dip while
    # retaining the broad DBR stopband.  This mirrors how the paper distinguishes
    # the two physical features.
    envelope = _rolling_max(reflectance, half_window=40)
    predicted_band = _largest_band(wavelengths, envelope >= 0.9)
    reported_band = tuple(PMC9147317_REFERENCE["reported_stopband_nm"])
    overlap = _interval_iou(predicted_band, reported_band)
    cavity_mask = (wavelengths >= 620.0) & (wavelengths <= 655.0)
    cavity_indices = np.where(cavity_mask)[0]
    dip_index = int(cavity_indices[np.argmin(reflectance[cavity_mask])])
    dip_nm = float(wavelengths[dip_index])
    dip_error_nm = abs(dip_nm - float(PMC9147317_REFERENCE["reported_cavity_dip_nm"]))

    profile = workbench.field_profile(
        task, dip_nm, polarization="s", points_per_layer=30
    )
    intensity = (
        np.abs(profile["Ex"]) ** 2
        + np.abs(profile["Ey"]) ** 2
        + np.abs(profile["Ez"]) ** 2
    )
    maximum_field_enhancement = float(np.max(intensity))
    checks = {
        "stopband_overlap_iou_at_least_0_75": overlap >= 0.75,
        "cavity_dip_error_at_most_10_nm": dip_error_nm <= 10.0,
        "strong_cavity_field_enhancement_at_least_50": maximum_field_enhancement >= 50.0,
        "forward_passivity": bool(result.audit.get("passivity_check_passed")),
    }
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "validation_level": "trend_reproduction_not_exact_curve_fit",
        "reference": PMC9147317_REFERENCE,
        "model_assumptions": [
            "published physical thicknesses were used",
            "local dispersive TiO2 and SiO2 datasets replace the paper's unpublished ellipsometric curves",
            "glass is approximated by n=1.52",
            "the reported 7 nm roughness layer is omitted",
        ],
        "predicted_stopband_nm": list(predicted_band),
        "stopband_interval_iou": float(overlap),
        "predicted_cavity_dip_nm": dip_nm,
        "cavity_dip_error_nm": float(dip_error_nm),
        "predicted_cavity_dip_reflectance": float(reflectance[dip_index]),
        "maximum_normalized_field_intensity": maximum_field_enhancement,
        "checks": checks,
        "physics_audit": result.audit,
        "material_provenance": result.material_provenance,
    }

    if output_dir is not None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "VALIDATION_REPORT.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        with (target / "SIMULATED_SPECTRUM.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["wavelength_nm", "reflectance", "rolling_envelope"])
            writer.writerows(zip(wavelengths.tolist(), reflectance.tolist(), envelope.tolist()))
        with (target / "FIELD_PROFILE.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["z_nm", "layer_index", "normalized_field_intensity"])
            writer.writerows(
                zip(profile["z_nm"].tolist(), profile["layer_index"].tolist(), intensity.tolist())
            )
    return report


__all__ = ["PMC9147317_REFERENCE", "pmc9147317_stack", "validate_pmc9147317"]
