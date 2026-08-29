from __future__ import annotations

from dataclasses import replace

import pytest

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


@pytest.mark.parametrize(
    ("physics", "expected_code"),
    [
        (PhysicsRequirements(geometry_class="lateral_periodic"), FailureCode.UNSUPPORTED_GEOMETRY),
        (PhysicsRequirements(geometry_class="arbitrary_2d"), FailureCode.UNSUPPORTED_GEOMETRY),
        (PhysicsRequirements(geometry_class="arbitrary_3d"), FailureCode.UNSUPPORTED_GEOMETRY),
        (PhysicsRequirements(material_class="anisotropic"), FailureCode.UNSUPPORTED_MATERIAL_MODEL),
        (PhysicsRequirements(material_class="magneto_optic"), FailureCode.UNSUPPORTED_MATERIAL_MODEL),
        (PhysicsRequirements(material_class="nonlinear"), FailureCode.UNSUPPORTED_MATERIAL_MODEL),
        (PhysicsRequirements(excitation_class="finite_beam"), FailureCode.UNSUPPORTED_EXCITATION),
        (PhysicsRequirements(excitation_class="dipole"), FailureCode.UNSUPPORTED_EXCITATION),
        (PhysicsRequirements(excitation_class="mode_source"), FailureCode.UNSUPPORTED_EXCITATION),
        (PhysicsRequirements(time_domain_required=True), FailureCode.TIME_DOMAIN_REQUIRED),
    ],
)
def test_every_declared_out_of_domain_profile_is_rejected_before_solver(
    physics: PhysicsRequirements,
    expected_code: FailureCode,
) -> None:
    assessment = assess_tmm_capability(replace(_task(), physics=physics))
    assert assessment.supported is False
    assert expected_code in {failure.code for failure in assessment.failures}
    assert assessment.resolved_solver is None


def test_capability_rejects_incoherent_complex_amplitude() -> None:
    task = _task()
    layer = replace(task.stack.layers[0], coherence="incoherent")
    task = replace(task, stack=replace(task.stack, layers=(layer,)), requested_outputs=("R", "amplitudes"))
    assessment = assess_tmm_capability(task)
    assert not assessment.supported
    assert assessment.failures[0].code == FailureCode.UNSUPPORTED_OUTPUT_COMBINATION


def test_capability_rejects_byrnes_amplitudes_instead_of_silently_omitting() -> None:
    task = replace(
        _task(),
        solver="byrnes",
        requested_outputs=("R", "T", "A", "amplitudes"),
    )
    assessment = assess_tmm_capability(task)
    assert not assessment.supported
    assert any(
        item.code == FailureCode.UNSUPPORTED_OUTPUT_COMBINATION
        and item.context.get("requested_output") == "amplitudes"
        for item in assessment.failures
    )


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
    assert certificate["evidence_coverage"]["capability_domain"] == "verified"
    assert certificate["evidence_coverage"]["numerical_convergence"] == "verified"
    assert certificate["evidence_coverage"]["independent_solver"] == "verified"
    assert certificate["evidence_coverage"]["experimental_fit"] == "not_evaluated"
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
    assert (
        certified.certificate["uncertainty_budget"]["applicability_gaps"][0]["limitation"]
        == "anisotropy_not_modeled"
    )
    assert (
        certified.certificate["evidence_coverage"]["uncertainty_quantified"]
        == "not_evaluated"
    )


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


def test_certificate_rejects_backend_that_omits_requested_output(monkeypatch) -> None:
    task = replace(_task(), requested_outputs=("R", "T", "A", "amplitudes"))
    workbench = TMMWorkbench(MaterialRegistry())
    broken = workbench.simulate(task)
    for values in broken.channels.values():
        values.pop("r", None)
        values.pop("t", None)
    monkeypatch.setattr(workbench, "simulate", lambda _task: broken)
    certified = certify_simulation(
        workbench,
        task,
        AcceptanceSettings(
            require_spectral_convergence=False,
            require_independent_solver=False,
        ),
    )
    assert certified.result is broken
    assert certified.certificate["accepted"] is False
    failure = certified.certificate["failures"][0]
    assert failure["code"] == "requested_output_missing"
    assert failure["context"]["missing_outputs"] == ["amplitudes"]
