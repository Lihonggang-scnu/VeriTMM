from __future__ import annotations

from dataclasses import replace

from tmm_engine import (
    AcceptanceSettings,
    FailureCode,
    IlluminationSpec,
    LayerSpec,
    MaterialRegistry,
    MediumSpec,
    PhysicsRequirements,
    SimulationTask,
    SpectralConvergenceSettings,
    SpectralGrid,
    StackSpec,
    TMMWorkbench,
    assess_tmm_capability,
    certify_simulation,
)
from tmm_engine.convergence import audit_spectral_convergence


def _task(points: int = 31) -> SimulationTask:
    return SimulationTask(
        stack=StackSpec(
            layers=(LayerSpec(None, 100.0, constant_n=2.1),),
            incident=MediumSpec.air(),
            exit=MediumSpec(constant_n=1.5),
        ),
        spectrum=SpectralGrid(start_nm=450.0, stop_nm=750.0, points=points),
        illumination=IlluminationSpec((0.0, 45.0), ("s", "p")),
    )


def test_capability_routes_lateral_periodicity_to_rcwa() -> None:
    task = replace(_task(), physics=PhysicsRequirements(geometry_class="lateral_periodic"))
    assessment = assess_tmm_capability(task)
    assert not assessment.supported
    assert assessment.failures[0].code == FailureCode.UNSUPPORTED_GEOMETRY
    assert assessment.failures[0].suggested_solver_family == "rcwa"


def test_capability_rejects_incoherent_complex_amplitude() -> None:
    task = _task()
    layer = replace(task.stack.layers[0], coherence="incoherent")
    task = replace(task, stack=replace(task.stack, layers=(layer,)), requested_outputs=("R", "amplitudes"))
    assessment = assess_tmm_capability(task)
    assert not assessment.supported
    assert assessment.failures[0].code == FailureCode.UNSUPPORTED_OUTPUT_COMBINATION


def test_layer_absorption_is_routed_instead_of_silently_omitted() -> None:
    task = replace(_task(), requested_outputs=("R", "T", "A", "layer_absorption"))
    result = TMMWorkbench(MaterialRegistry()).simulate(task)
    assert result.solver == "byrnes"
    assert any(key.startswith("layer_absorption|") for key in result.extras)
    assert result.audit["capability_assessment"]["resolved_solver"] == "byrnes"


def test_certificate_contains_convergence_cross_solver_and_catalog_hash() -> None:
    settings = AcceptanceSettings(
        convergence=SpectralConvergenceSettings(
            max_refinements=3,
            max_pointwise_deviation=1e-2,
            max_integral_deviation=2e-3,
        )
    )
    certified = certify_simulation(TMMWorkbench(MaterialRegistry()), _task(), settings)
    certificate = certified.certificate
    assert certificate["status"] == "physically_valid"
    assert certificate["spectral_convergence"]["passed"]
    assert certificate["independent_solver_check"]["status"] == "passed"
    assert len(certificate["material_catalog"]["rii_sqlite"]["database_sha256"]) == 64
    assert len(certificate["certificate_id"]) == 64


def test_certificate_rejects_out_of_domain_without_running_solver() -> None:
    task = replace(
        _task(),
        physics=PhysicsRequirements(material_class="anisotropic"),
    )
    certified = certify_simulation(TMMWorkbench(MaterialRegistry()), task)
    assert certified.result is None
    assert not certified.certificate["accepted"]
    assert certified.certificate["failures"][0]["suggested_solver_family"] == "berreman_4x4"


def test_convergence_audit_exposes_underresolved_spectrum() -> None:
    task = SimulationTask(
        stack=StackSpec(
            layers=(LayerSpec(None, 900.0, constant_n=2.35),),
            incident=MediumSpec.air(),
            exit=MediumSpec(constant_n=1.45),
        ),
        spectrum=SpectralGrid(start_nm=400.0, stop_nm=800.0, points=3),
    )
    outcome = audit_spectral_convergence(
        TMMWorkbench(MaterialRegistry()),
        task,
        SpectralConvergenceSettings(
            max_refinements=1,
            max_pointwise_deviation=1e-8,
            max_integral_deviation=1e-9,
        ),
    )
    assert not outcome.passed
    assert outcome.status == "spectral_convergence_failure"
    assert outcome.rounds[0]["max_pointwise_deviation"] > 1e-4
