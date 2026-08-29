"""Tests locking the public capability boundary to the current runtime."""

from __future__ import annotations

import json

from tmm_engine import __version__
from tmm_engine.protocol import (
    PROTOCOL_VERSION,
    SUPPORTED_REQUESTED_OUTPUTS,
    SUPPORTED_SOLVERS,
    describe_capabilities,
)


def test_manifest_is_stable_and_machine_readable() -> None:
    first = describe_capabilities()
    second = describe_capabilities()
    first_payload = first.model_dump(mode="json")
    assert first == second
    assert first_payload == second.model_dump(mode="json")
    json.dumps(first_payload)
    assert first_payload["protocol_version"] == PROTOCOL_VERSION
    assert first_payload["package_version"] == __version__


def test_manifest_declares_the_current_supported_surface() -> None:
    manifest = describe_capabilities()
    assert manifest.modes == (
        "simulate",
        "optimize",
        "sweep",
        "sensitivity",
        "tolerance",
    )
    assert manifest.solvers == SUPPORTED_SOLVERS == ("smatrix", "characteristic", "byrnes")
    assert manifest.geometry == ("layered_planar",)
    assert manifest.excitation == ("plane_wave",)
    assert manifest.material_models[0].model_dump(mode="json") == {
        "material_class": "isotropic",
        "passivity": "passive",
        "representation": "scalar_nk",
    }
    assert manifest.units.model_dump(mode="json") == {
        "wavelength": "nm",
        "thickness": "nm",
        "angle": "deg",
    }
    assert manifest.requested_outputs == SUPPORTED_REQUESTED_OUTPUTS


def test_manifest_describes_mixed_coherence_and_optimization_without_overclaiming() -> None:
    manifest = describe_capabilities()
    mixed = manifest.mixed_coherence
    assert mixed.supported
    assert mixed.layer_coherence == ("coherent", "incoherent")
    assert mixed.routing_solver == "byrnes"
    assert "amplitudes" in mixed.unsupported_outputs
    assert "ellipsometry" in mixed.unsupported_outputs
    assert "phase_dispersion" in mixed.unsupported_outputs

    optimization = manifest.optimization
    assert optimization.supported
    assert optimization.parameterization == "layer_thickness_nm"
    assert optimization.fixed_material_selection
    assert optimization.not_formally_supported == ()
    assert manifest.scientific_analysis.physics_validity_is_separate_from_robustness
    assert manifest.scientific_analysis.uncertainty_boundary_policies == ("truncate",)
    assert manifest.scientific_analysis.yield_denominator == "completed_sample_count"


def test_manifest_artifact_types_are_explicit() -> None:
    artifact_types = set(describe_capabilities().artifact_types)
    assert {
        "normalized_task",
        "simulation_result",
        "optimization_result",
        "physics_certificate",
        "preflight_report",
        "spectrum_table",
        "run_result",
        "sweep_result",
        "sweep_table",
        "sensitivity_result",
        "tolerance_result",
        "robustness_report",
        "benchmark_result",
        "agent_trajectory",
        "agent_ab_result",
    } <= artifact_types


def test_manifest_declares_offline_agentbench_without_claiming_mcp() -> None:
    agent_bench = describe_capabilities().agent_bench
    assert agent_bench.supported is True
    assert agent_bench.offline_deterministic is True
    assert agent_bench.llm_required is False
    assert agent_bench.network_required is False
    assert agent_bench.minimum_release_gate_cases == 80
    assert agent_bench.unsupported_false_accept_required == 0.0
    assert agent_bench.mcp_status == "optional_deferred"
