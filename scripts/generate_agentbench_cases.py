"""Generate the maintained VeriTMM AgentBench v1 case catalogue.

The catalogue is deterministic and intentionally contains both executable
physics tasks and fail-closed boundary cases.  Expected failure codes are
declared here rather than learned from the runtime under test.  Generation
also performs an independent preflight audit and refuses to write a catalogue
whose observed contract differs from those declarations.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tmm_engine.agent_bench import BenchmarkCase  # noqa: E402
from tmm_engine.preflight import preflight_path  # noqa: E402
from tmm_engine.run_artifacts import write_json  # noqa: E402

DEFAULT_OUTPUT = ROOT / "benchmarks" / "cases"


def _layer(n: float, d: float, **extra: Any) -> dict[str, Any]:
    return {"constant_n": n, "thickness_nm": d, **extra}


def _simulation(
    *,
    layers: list[dict[str, Any]] | None = None,
    start: float = 450.0,
    stop: float = 750.0,
    points: int = 61,
    angles: list[float] | None = None,
    polarizations: list[str] | None = None,
    solver: str = "smatrix",
    outputs: list[str] | None = None,
    physics: dict[str, Any] | None = None,
    incident_n: float = 1.0,
    exit_n: float = 1.52,
    name: str = "agentbench-stack",
) -> dict[str, Any]:
    return {
        "mode": "simulate",
        "simulation": {
            "stack": {
                "name": name,
                "incident": {"constant_n": incident_n},
                "layers": layers or [_layer(1.38, 105.0, optimizable=False)],
                "exit": {"constant_n": exit_n},
            },
            "spectrum": {"start_nm": start, "stop_nm": stop, "points": points},
            "illumination": {
                "angles_deg": angles or [0.0],
                "polarizations": polarizations or ["unpolarized"],
            },
            "solver": solver,
            "requested_outputs": outputs or ["R", "T", "A"],
            "physics": physics or {},
        },
    }


def _dbr(periods: int = 3, center_nm: float = 600.0) -> list[dict[str, Any]]:
    high, low = 2.2, 1.45
    layers: list[dict[str, Any]] = []
    for _ in range(periods):
        layers.extend(
            [
                _layer(high, center_nm / (4.0 * high), optimizable=False),
                _layer(low, center_nm / (4.0 * low), optimizable=False),
            ]
        )
    return layers


def _sweep(*, scenario: str = "standard", angle_axis: bool = False) -> dict[str, Any]:
    parameter = (
        {"path": "/illumination/angles_deg/0", "values": [0.0, 20.0, 40.0]}
        if angle_axis
        else {"path": "/stack/layers/1/thickness_nm", "values": [90.0, 103.4, 116.0]}
    )
    task = {
        "schema_version": "sweep-task-v1",
        "mode": "sweep",
        "sweep": {
            "base_simulation": _simulation(layers=_dbr(2), start=500.0, stop=700.0, points=41)[
                "simulation"
            ],
            "parameters": [parameter],
            "metrics": [
                {
                    "name": "peak_R",
                    "observable": "R",
                    "wavelength_min_nm": 520.0,
                    "wavelength_max_nm": 680.0,
                    "aggregation": "max",
                }
            ],
        },
    }
    task["_scenario_for_generator"] = scenario
    return task


def _sensitivity(layer_count: int = 1) -> dict[str, Any]:
    layers = [
        _layer(
            1.38 if index % 2 == 0 else 2.05,
            100.0 if index % 2 == 0 else 70.0,
            optimizable=True,
            min_thickness_nm=30.0,
            max_thickness_nm=180.0,
        )
        for index in range(layer_count)
    ]
    return {
        "schema_version": "sensitivity-task-v1",
        "mode": "sensitivity",
        "sensitivity": {
            "simulation": _simulation(layers=layers, start=500.0, stop=650.0, points=41)[
                "simulation"
            ],
            "metric": {
                "name": "mean_R",
                "observable": "R",
                "wavelength_min_nm": 520.0,
                "wavelength_max_nm": 620.0,
                "aggregation": "mean",
            },
            "parameters": "optimizable_thicknesses",
            "finite_difference_step_nm": 0.01,
            "relative_error_tolerance": 0.001,
            "absolute_error_tolerance": 1e-7,
        },
    }


def _tolerance(distribution: str = "normal", *, correlated: bool = False) -> dict[str, Any]:
    uncertainty = (
        {"layer_index": 0, "distribution": "normal", "sigma_nm": 2.0}
        if distribution == "normal"
        else {"layer_index": 0, "distribution": "uniform", "half_width_nm": 3.0}
    )
    metric = {
        "name": "mean_T",
        "observable": "T",
        "wavelength_min_nm": 520.0,
        "wavelength_max_nm": 620.0,
        "aggregation": "mean",
    }
    payload: dict[str, Any] = {
        "simulation": _simulation(start=500.0, stop=650.0, points=41)["simulation"],
        "uncertainties": [uncertainty],
        "metric": metric,
        "target": {"metric": metric, "constraint": "at_least", "value": 0.85},
        "sample_count": 12,
        "seed": 42,
    }
    if correlated:
        payload["global_correlated_bias_nm"] = 1.0
    return {
        "schema_version": "tolerance-task-v1",
        "mode": "tolerance",
        "tolerance": payload,
    }


def _optimization(*, robust: str | None = None, quantized: bool = False) -> dict[str, Any]:
    simulation = _simulation(
        layers=[
            _layer(
                1.38,
                105.0,
                optimizable=True,
                min_thickness_nm=50.0,
                max_thickness_nm=180.0,
            )
        ],
        start=500.0,
        stop=650.0,
        points=31,
    )["simulation"]
    optimization: dict[str, Any] = {
        "simulation": simulation,
        "targets": [
            {
                "name": "broadband_transmission",
                "observable": "T",
                "target": 0.96,
                "constraint": "at_least",
                "aggregation": "mean",
                "wavelength_min_nm": 520.0,
                "wavelength_max_nm": 620.0,
                "angle_deg": 0.0,
                "polarization": "unpolarized",
            }
        ],
        "optimizer": {
            "method": "adam",
            "max_steps": 4,
            "learning_rate": 0.05,
            "starts": 1,
            "seed": 7,
            "early_stop_patience": 3,
            "quantization_nm": 1.0 if quantized else None,
        },
    }
    if robust is not None:
        optimization["robustness"] = {
            "enabled": True,
            "objective": robust,
            "samples_per_step": 2,
            "final_samples": 8,
            "seed": 7,
            "thickness_sigma_nm": 2.0,
            "k_sigma": 2.0,
        }
    return {"mode": "optimize", "optimization": optimization}


def _near_zero_sensitivity() -> dict[str, Any]:
    task = _sensitivity(1)
    simulation = task["sensitivity"]["simulation"]
    simulation["stack"]["incident"] = {"constant_n": 1.0}
    simulation["stack"]["layers"][0]["constant_n"] = 1.0
    simulation["stack"]["exit"] = {"constant_n": 1.0}
    return task


def _low_yield_tolerance() -> dict[str, Any]:
    task = _tolerance("normal")
    task["tolerance"]["target"]["value"] = 0.9999
    return task


def _assert_ok() -> list[dict[str, Any]]:
    return [{"source": "run_result", "path": "ok", "operator": "eq", "expected": True}]


def _case(
    case_id: str,
    category: str,
    natural_language_task: str,
    task: dict[str, Any],
    *,
    mode: str,
    capability: str = "supported",
    failure_codes: list[str] | None = None,
    execution: str = "preflight_only",
    scenario: str = "standard",
    artifacts: list[str] | None = None,
    assertions: list[dict[str, Any]] | None = None,
    difficulty: str = "intermediate",
    tags: list[str] | None = None,
) -> BenchmarkCase:
    clean_task = copy.deepcopy(task)
    clean_task.pop("_scenario_for_generator", None)
    return BenchmarkCase.model_validate(
        {
            "case_id": case_id,
            "category": category,
            "natural_language_task": natural_language_task,
            "task": clean_task,
            "expected_mode": mode,
            "expected_capability": capability,
            "expected_failure_codes": failure_codes or [],
            "expected_artifacts": artifacts or [],
            "physics_assertions": assertions or [],
            "difficulty": difficulty,
            "tags": tags or [],
            "execution": execution,
            "scenario": scenario,
            "reproducibility_runs": 2,
        }
    )


def build_cases() -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []

    # Broad, valid planar simulation surface.  Only a representative subset is
    # executed; the rest stress contract/preflight routing without inflating CI.
    simulations = [
        (
            "single_film_normal",
            "Simulate a single dielectric film at normal incidence.",
            _simulation(),
        ),
        (
            "single_film_oblique_s",
            "Simulate a single film at 55 degrees with s polarization.",
            _simulation(angles=[55.0], polarizations=["s"]),
        ),
        (
            "single_film_oblique_p",
            "Simulate a single film at 55 degrees with p polarization.",
            _simulation(angles=[55.0], polarizations=["p"]),
        ),
        (
            "single_film_unpolarized",
            "Evaluate unpolarized reflectance of a passive film.",
            _simulation(polarizations=["unpolarized"]),
        ),
        (
            "multi_angle_channels",
            "Compare normal and oblique s and p channels.",
            _simulation(angles=[0.0, 30.0, 60.0], polarizations=["s", "p"]),
        ),
        (
            "dbr_three_period",
            "Simulate a three-period quarter-wave Bragg reflector.",
            _simulation(layers=_dbr(3)),
        ),
        (
            "dbr_six_period",
            "Simulate a six-period quarter-wave Bragg reflector.",
            _simulation(layers=_dbr(6), points=81),
        ),
        (
            "defect_cavity",
            "Simulate a Bragg mirror with a half-wave central defect.",
            _simulation(layers=_dbr(2) + [_layer(1.45, 206.9, optimizable=False)] + _dbr(2)),
        ),
        (
            "chirped_stack",
            "Simulate a chirped alternating-index multilayer.",
            _simulation(
                layers=[
                    _layer(2.1, d, optimizable=False)
                    if i % 2 == 0
                    else _layer(1.45, d, optimizable=False)
                    for i, d in enumerate([55, 85, 62, 95, 70, 105])
                ]
            ),
        ),
        (
            "absorbing_layer",
            "Simulate a weakly absorbing film and verify passive RTA.",
            _simulation(layers=[_layer(2.0, 80.0, constant_k=0.08, optimizable=False)]),
        ),
        (
            "strong_absorber",
            "Simulate a strongly absorbing scalar film.",
            _simulation(layers=[_layer(2.4, 120.0, constant_k=0.7, optimizable=False)]),
        ),
        (
            "finite_incoherent_slab",
            "Simulate a coherent coating on a thick incoherent slab.",
            _simulation(
                layers=[
                    _layer(2.0, 90.0, optimizable=False),
                    _layer(1.5, 500000.0, coherence="incoherent", optimizable=False),
                ],
                solver="byrnes",
            ),
        ),
        (
            "layer_absorption",
            "Resolve absorption by layer in a coherent absorbing stack.",
            _simulation(
                layers=[
                    _layer(2.0, 80.0, constant_k=0.1, optimizable=False),
                    _layer(1.5, 100.0, optimizable=False),
                ],
                outputs=["R", "T", "A", "layer_absorption"],
            ),
        ),
        (
            "system_emissivity",
            "Calculate system emissivity for a passive multilayer.",
            _simulation(
                layers=[_layer(2.0, 80.0, constant_k=0.1, optimizable=False)],
                outputs=["R", "T", "A", "system_emissivity"],
            ),
        ),
        (
            "ellipsometry",
            "Calculate ellipsometric observables for a coherent film.",
            _simulation(
                angles=[65.0], polarizations=["s", "p"], outputs=["R", "T", "A", "ellipsometry"]
            ),
        ),
        (
            "amplitudes",
            "Return coherent complex amplitude observables.",
            _simulation(outputs=["R", "T", "A", "amplitudes"]),
        ),
        (
            "phase_dispersion",
            "Calculate phase and dispersion on a sufficiently dense grid.",
            _simulation(points=31, outputs=["R", "T", "A", "phase_dispersion"]),
        ),
        (
            "characteristic_backend",
            "Run the characteristic-matrix backend for a benign film.",
            _simulation(solver="characteristic"),
        ),
        (
            "byrnes_backend",
            "Run the independent Byrnes backend for RTA outputs.",
            _simulation(solver="byrnes"),
        ),
        (
            "ir_constant_indices",
            "Simulate a constant-index infrared multilayer.",
            _simulation(layers=_dbr(3, 1550.0), start=1200.0, stop=1900.0, points=71),
        ),
        (
            "uv_constant_indices",
            "Simulate a constant-index ultraviolet stack.",
            _simulation(start=300.0, stop=400.0, points=41),
        ),
        (
            "irregular_grid",
            "Simulate on an explicit irregular wavelength grid.",
            {
                "mode": "simulate",
                "simulation": {
                    **_simulation()["simulation"],
                    "spectrum": {"values_nm": [450.0, 470.0, 510.0, 590.0, 750.0]},
                },
            },
        ),
        (
            "sparse_grid_warning",
            "Preflight a sparse but valid spectrum and report its warning.",
            _simulation(points=7),
        ),
        (
            "near_grazing_warning",
            "Preflight a valid near-grazing plane-wave case.",
            _simulation(angles=[82.0], polarizations=["s", "p"]),
        ),
        (
            "very_thick_warning",
            "Preflight an extremely thick coherent film without silently changing coherence.",
            _simulation(layers=[_layer(1.5, 100000.0, optimizable=False)]),
        ),
        (
            "extremely_thin_warning",
            "Preflight an extremely thin positive film.",
            _simulation(layers=[_layer(2.0, 0.01, optimizable=False)]),
        ),
        (
            "many_layer_stack",
            "Preflight a 40-layer planar dielectric stack.",
            _simulation(layers=_dbr(20), points=31),
        ),
        ("air_symmetric", "Simulate a film between equal air media.", _simulation(exit_n=1.0)),
        (
            "index_matched",
            "Simulate an index-matched film and substrate.",
            _simulation(layers=[_layer(1.52, 100.0, optimizable=False)], exit_n=1.52),
        ),
        (
            "narrow_resonance_grid",
            "Simulate a dense narrow-band cavity spectrum.",
            _simulation(
                layers=_dbr(3) + [_layer(1.45, 206.9, optimizable=False)] + _dbr(3),
                start=560.0,
                stop=640.0,
                points=161,
            ),
        ),
        (
            "broadband_visible",
            "Simulate an alternating stack across the visible band.",
            _simulation(layers=_dbr(4), start=380.0, stop=780.0, points=101),
        ),
        (
            "multi_output_full",
            "Request RTA, layer absorption and system emissivity together.",
            _simulation(
                layers=[_layer(2.0, 80.0, constant_k=0.05, optimizable=False)],
                outputs=["R", "T", "A", "layer_absorption", "system_emissivity"],
            ),
        ),
        (
            "brewster_neighborhood",
            "Resolve p-polarized response around the Brewster-angle regime.",
            _simulation(
                angles=[50.0, 56.0, 62.0],
                polarizations=["p"],
                points=51,
            ),
        ),
        (
            "lossy_multilayer",
            "Simulate a passive multilayer containing two absorption levels.",
            _simulation(
                layers=[
                    _layer(2.1, 65.0, constant_k=0.02, optimizable=False),
                    _layer(1.45, 110.0, optimizable=False),
                    _layer(2.3, 45.0, constant_k=0.2, optimizable=False),
                ],
                points=71,
            ),
        ),
    ]
    executed_simulations = {
        "single_film_normal",
        "single_film_oblique_s",
        "dbr_three_period",
        "dbr_six_period",
        "absorbing_layer",
        "finite_incoherent_slab",
        "layer_absorption",
        "system_emissivity",
        "ellipsometry",
        "amplitudes",
        "phase_dispersion",
        "characteristic_backend",
        "narrow_resonance_grid",
    }
    for name, text, task in simulations:
        execute = name in executed_simulations
        cases.append(
            _case(
                f"sim_{name}",
                "valid_simulation",
                text,
                task,
                mode="simulate",
                execution="run" if execute else "preflight_only",
                artifacts=(
                    ["physics_certificate", "result_summary", "spectrum_table"] if execute else []
                ),
                assertions=_assert_ok() if execute else [],
                tags=["planar", "simulation", name],
            )
        )

    # Studies and inverse design.  Cache/restart cases verify runtime semantics,
    # not merely schema acceptance.
    study_specs = [
        (
            "sweep_thickness",
            "sweep",
            _sweep(),
            "run",
            "standard",
            ["sweep_result", "sweep_table", "result_summary"],
        ),
        ("sweep_angle", "sweep", _sweep(angle_axis=True), "preflight_only", "standard", []),
        (
            "sweep_cache_replay",
            "sweep",
            _sweep(),
            "run",
            "cache_replay",
            ["sweep_result", "sweep_table", "result_summary"],
        ),
        (
            "sweep_resume",
            "sweep",
            _sweep(),
            "run",
            "sweep_resume",
            ["sweep_result", "sweep_table", "result_summary"],
        ),
        (
            "sensitivity_one_layer",
            "sensitivity",
            _sensitivity(1),
            "run",
            "standard",
            ["sensitivity_result", "result_summary"],
        ),
        (
            "sensitivity_two_layers",
            "sensitivity",
            _sensitivity(2),
            "preflight_only",
            "standard",
            [],
        ),
        (
            "sensitivity_near_zero",
            "sensitivity",
            _near_zero_sensitivity(),
            "run",
            "standard",
            ["sensitivity_result", "result_summary"],
        ),
        (
            "tolerance_normal",
            "tolerance",
            _tolerance("normal"),
            "run",
            "standard",
            ["tolerance_result", "robustness_report", "result_summary"],
        ),
        ("tolerance_uniform", "tolerance", _tolerance("uniform"), "preflight_only", "standard", []),
        (
            "tolerance_correlated",
            "tolerance",
            _tolerance("normal", correlated=True),
            "preflight_only",
            "standard",
            [],
        ),
        (
            "tolerance_low_yield",
            "tolerance",
            _low_yield_tolerance(),
            "run",
            "standard",
            ["tolerance_result", "robustness_report", "result_summary"],
        ),
        (
            "optimize_ar",
            "optimize",
            _optimization(),
            "run",
            "standard",
            ["optimization_result", "physics_certificate", "result_summary", "design_portfolio"],
        ),
        (
            "optimize_quantized",
            "optimize",
            _optimization(quantized=True),
            "preflight_only",
            "standard",
            [],
        ),
        (
            "robust_expected",
            "optimize",
            _optimization(robust="expected_loss"),
            "preflight_only",
            "standard",
            [],
        ),
        (
            "robust_worst_case",
            "optimize",
            _optimization(robust="worst_case_loss"),
            "preflight_only",
            "standard",
            [],
        ),
        (
            "robust_mean_sigma",
            "optimize",
            _optimization(robust="mean_plus_k_sigma"),
            "run",
            "standard",
            [
                "optimization_result",
                "physics_certificate",
                "result_summary",
                "design_portfolio",
                "robustness_report",
            ],
        ),
    ]
    for name, mode, task, execution, scenario, artifacts in study_specs:
        assertions = _assert_ok() if execution == "run" else []
        if name.startswith("sweep_") and execution == "run":
            assertions.append(
                {
                    "source": "sweep_result",
                    "path": "successful_child_count",
                    "operator": "eq",
                    "expected": 3,
                }
            )
        if scenario == "cache_replay":
            assertions.append(
                {
                    "source": "run_result",
                    "path": "cache_hit",
                    "operator": "eq",
                    "expected": True,
                }
            )
        if scenario == "sweep_resume":
            assertions.extend(
                [
                    {
                        "source": "sweep_result",
                        "path": "resumed_child_count",
                        "operator": "eq",
                        "expected": 3,
                    },
                    {
                        "source": "sweep_result",
                        "path": "executed_child_count",
                        "operator": "eq",
                        "expected": 0,
                    },
                ]
            )
        if mode == "sensitivity" and execution == "run":
            assertions.append(
                {
                    "source": "sensitivity_result",
                    "path": "finite_difference_audit.passed",
                    "operator": "eq",
                    "expected": True,
                }
            )
            if name == "sensitivity_near_zero":
                assertions.append(
                    {
                        "source": "sensitivity_result",
                        "path": "parameters.0.near_zero_gradient",
                        "operator": "eq",
                        "expected": True,
                    }
                )
        if mode == "tolerance" and execution == "run":
            assertions.extend(
                [
                    {
                        "source": "tolerance_result",
                        "path": "completed_sample_count",
                        "operator": "eq",
                        "expected": 12,
                    },
                    {
                        "source": "tolerance_result",
                        "path": "failed_sample_count",
                        "operator": "eq",
                        "expected": 0,
                    },
                ]
            )
            if name == "tolerance_low_yield":
                assertions.append(
                    {
                        "source": "tolerance_result",
                        "path": "yield",
                        "operator": "eq",
                        "expected": 0.0,
                    }
                )
        if mode == "optimize" and execution == "run":
            assertions.append(
                {
                    "source": "optimization_result",
                    "path": "status",
                    "operator": "eq",
                    "expected": "completed",
                }
            )
        if name == "robust_mean_sigma":
            assertions.extend(
                [
                    {
                        "source": "robustness_report",
                        "path": "status",
                        "operator": "eq",
                        "expected": "evaluated",
                    },
                    {
                        "source": "robustness_report",
                        "path": "physics_validity_is_separate",
                        "operator": "eq",
                        "expected": True,
                    },
                ]
            )
        cases.append(
            _case(
                name,
                "valid_study" if mode != "optimize" else "valid_inverse_design",
                f"Execute or validate the {name.replace('_', ' ')} workflow.",
                task,
                mode=mode,
                execution=execution,
                scenario=scenario,
                artifacts=artifacts,
                assertions=assertions,
                difficulty="advanced",
                tags=[mode, scenario],
            )
        )

    # Capability boundaries: these are scientifically out of scope, not weak
    # inputs that should be coerced into scalar planar TMM.
    boundary_fields = [
        ("grating", {"geometry_class": "lateral_periodic"}, "unsupported_geometry"),
        ("arbitrary_2d", {"geometry_class": "arbitrary_2d"}, "unsupported_geometry"),
        ("arbitrary_3d", {"geometry_class": "arbitrary_3d"}, "unsupported_geometry"),
        ("anisotropic", {"material_class": "anisotropic"}, "unsupported_material_model"),
        ("magneto_optic", {"material_class": "magneto_optic"}, "unsupported_material_model"),
        ("nonlinear", {"material_class": "nonlinear"}, "unsupported_material_model"),
        ("finite_beam", {"excitation_class": "finite_beam"}, "unsupported_excitation"),
        ("dipole", {"excitation_class": "dipole"}, "unsupported_excitation"),
        ("mode_source", {"excitation_class": "mode_source"}, "unsupported_excitation"),
        ("time_domain", {"time_domain_required": True}, "time_domain_required"),
    ]
    for name, physics, code in boundary_fields:
        cases.append(
            _case(
                f"reject_{name}",
                "unsupported_physics",
                f"Attempt a {name.replace('_', ' ')} task that scalar planar TMM must reject.",
                _simulation(physics=physics),
                mode="simulate",
                capability="unsupported",
                failure_codes=[code],
                difficulty="adversarial",
                tags=["fail_closed", name],
            )
        )

    output_boundaries = [
        (
            "mixed_amplitudes",
            _simulation(
                layers=[_layer(1.5, 500000.0, coherence="incoherent", optimizable=False)],
                solver="byrnes",
                outputs=["R", "T", "A", "amplitudes"],
            ),
        ),
        (
            "mixed_ellipsometry",
            _simulation(
                layers=[_layer(1.5, 500000.0, coherence="incoherent", optimizable=False)],
                solver="byrnes",
                outputs=["R", "T", "A", "ellipsometry"],
            ),
        ),
        (
            "mixed_phase",
            _simulation(
                layers=[_layer(1.5, 500000.0, coherence="incoherent", optimizable=False)],
                solver="byrnes",
                outputs=["R", "T", "A", "phase_dispersion"],
            ),
        ),
        ("byrnes_amplitudes", _simulation(solver="byrnes", outputs=["R", "T", "A", "amplitudes"])),
        ("byrnes_phase", _simulation(solver="byrnes", outputs=["R", "T", "A", "phase_dispersion"])),
        (
            "phase_too_few_points",
            _simulation(points=2, outputs=["R", "T", "A", "phase_dispersion"]),
        ),
        (
            "amplitudes_with_ellipsometry",
            _simulation(outputs=["R", "T", "A", "amplitudes", "ellipsometry"]),
        ),
        (
            "phase_with_layer_absorption",
            _simulation(outputs=["R", "T", "A", "phase_dispersion", "layer_absorption"]),
        ),
    ]
    for name, task in output_boundaries:
        cases.append(
            _case(
                f"reject_{name}",
                "unsupported_output_combination",
                f"Reject the incompatible output request {name.replace('_', ' ')}.",
                task,
                mode="simulate",
                capability="unsupported",
                failure_codes=["unsupported_output_combination"],
                difficulty="adversarial",
                tags=["fail_closed", "output_contract"],
            )
        )

    # Invalid contracts and unresolved material references.  These remain
    # distinct from unsupported physics so benchmark metrics expose both.
    fixture_root = ROOT / "tests" / "fixtures" / "agent_invalid"
    fixture_cases = [
        ("material_not_found", "material_not_found"),
        ("material_ambiguity", "material_ambiguity"),
        ("material_out_of_range", "material_range_error"),
        ("negative_thickness", "invalid_task"),
    ]
    for name, code in fixture_cases:
        task = json.loads((fixture_root / f"{name}.json").read_text(encoding="utf-8"))
        cases.append(
            _case(
                f"invalid_{name}",
                "invalid_or_unresolved_input",
                f"Diagnose the {name.replace('_', ' ')} input without guessing a replacement.",
                task,
                mode="simulate",
                capability="invalid",
                failure_codes=[code],
                difficulty="adversarial",
                tags=["typed_failure", name],
            )
        )

    invalid_tasks: list[tuple[str, dict[str, Any], str]] = []
    bad_grid = _simulation()
    bad_grid["simulation"]["spectrum"] = {"start_nm": 700.0, "stop_nm": 500.0, "points": 21}
    invalid_tasks.append(("descending_grid", bad_grid, "simulate"))
    bad_angle = _simulation(angles=[90.0])
    invalid_tasks.append(("angle_90", bad_angle, "simulate"))
    bad_solver = _simulation()
    bad_solver["simulation"]["solver"] = "fdtd"
    invalid_tasks.append(("unknown_solver", bad_solver, "simulate"))
    bad_output = _simulation()
    bad_output["simulation"]["requested_outputs"] = ["R", "diffraction_orders"]
    invalid_tasks.append(("unknown_output", bad_output, "simulate"))
    no_opt = _optimization()
    no_opt["optimization"]["simulation"]["stack"]["layers"][0]["optimizable"] = False
    invalid_tasks.append(("optimization_no_variable", no_opt, "optimize"))
    off_grid = _optimization()
    off_grid["optimization"]["targets"][0]["wavelength_min_nm"] = 800.0
    off_grid["optimization"]["targets"][0]["wavelength_max_nm"] = 900.0
    invalid_tasks.append(("optimization_target_outside_grid", off_grid, "optimize"))
    wrong_channel = _optimization()
    wrong_channel["optimization"]["targets"][0]["polarization"] = "s"
    invalid_tasks.append(("optimization_channel_missing", wrong_channel, "optimize"))
    bad_sweep = _sweep()
    bad_sweep["sweep"]["parameters"][0]["path"] = "/stack/layers/99/thickness_nm"
    bad_sweep.pop("_scenario_for_generator", None)
    invalid_tasks.append(("sweep_layer_outside_stack", bad_sweep, "sweep"))
    bad_tolerance = _tolerance()
    bad_tolerance["tolerance"]["uncertainties"][0]["layer_index"] = 9
    invalid_tasks.append(("tolerance_layer_outside_stack", bad_tolerance, "tolerance"))
    bad_sensitivity = _sensitivity()
    bad_sensitivity["sensitivity"]["metric"]["wavelength_min_nm"] = 800.0
    bad_sensitivity["sensitivity"]["metric"]["wavelength_max_nm"] = 900.0
    invalid_tasks.append(("sensitivity_metric_outside_grid", bad_sensitivity, "sensitivity"))
    for name, task, mode in invalid_tasks:
        cases.append(
            _case(
                f"invalid_{name}",
                "invalid_contract",
                f"Reject and diagnose the malformed {name.replace('_', ' ')} task.",
                task,
                mode=mode,
                capability="invalid",
                failure_codes=["invalid_task"],
                difficulty="adversarial",
                tags=["typed_failure", "invalid_contract"],
            )
        )

    if len(cases) < 80:
        raise AssertionError(f"catalogue must contain at least 80 cases, got {len(cases)}")
    return cases


def _folder(case: BenchmarkCase) -> str:
    if case.expected_capability == "unsupported":
        return "unsupported"
    if case.expected_capability == "invalid":
        return "invalid"
    if case.expected_mode == "optimize":
        return "optimization"
    if case.expected_mode in {"sensitivity", "tolerance"}:
        return "robustness"
    if case.expected_mode == "sweep":
        return "sweeps"
    return "simulation"


def _audit_expectations(cases: list[BenchmarkCase]) -> None:
    mismatches: list[str] = []
    with tempfile.TemporaryDirectory(prefix="veritmm_case_audit_") as temporary:
        root = Path(temporary)
        for case in cases:
            task_path = root / f"{case.case_id}.json"
            write_json(task_path, case.task)
            report = preflight_path(task_path)
            observed_codes = sorted({str(item.get("code")) for item in report.get("failures", [])})
            expected_codes = sorted(set(case.expected_failure_codes))
            # Structurally invalid tasks can fail before the loader can safely
            # identify a mode.  Supported and capability-boundary tasks must
            # retain their declared mode through preflight.
            if case.expected_capability != "invalid" and report.get("mode") != case.expected_mode:
                mismatches.append(
                    f"{case.case_id}: mode={report.get('mode')} expected={case.expected_mode}"
                )
            if observed_codes != expected_codes:
                mismatches.append(
                    f"{case.case_id}: failures={observed_codes} expected={expected_codes}"
                )
            accepted = bool(report.get("ok"))
            if accepted != (case.expected_capability == "supported"):
                mismatches.append(
                    f"{case.case_id}: accepted={accepted} capability={case.expected_capability}"
                )
    if mismatches:
        raise RuntimeError("AgentBench expectation audit failed:\n" + "\n".join(mismatches))


def generate(output: Path) -> dict[str, Any]:
    cases = build_cases()
    _audit_expectations(cases)
    resolved = output.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    for path in resolved.rglob("*.json"):
        if resolved not in path.resolve().parents:
            raise RuntimeError(f"refusing to remove path outside case root: {path}")
        path.unlink()
    for case in cases:
        write_json(
            resolved / _folder(case) / f"{case.case_id}.json",
            case.model_dump(mode="json"),
        )
    return {
        "case_count": len(cases),
        "mode_counts": dict(Counter(case.expected_mode for case in cases)),
        "capability_counts": dict(Counter(case.expected_capability for case in cases)),
        "run_case_count": sum(case.execution == "run" for case in cases),
        "scenario_counts": dict(Counter(case.scenario for case in cases)),
        "output": str(resolved),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(generate(args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
