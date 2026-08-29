"""Tests for the typed EvidenceCoverage ledger."""

from __future__ import annotations

import json

from tmm_engine.protocol import PhysicsCertificate
from tmm_engine.protocol.evidence import EvidenceCoverage, EvidenceStatus, from_certificate


def test_evidence_status_enum() -> None:
    assert EvidenceStatus.VERIFIED.value == "verified"
    assert EvidenceStatus.NOT_EVALUATED.value == "not_evaluated"
    assert EvidenceStatus.UNAVAILABLE.value == "unavailable"
    assert EvidenceStatus.FAILED.value == "failed"


def test_default_coverage_all_not_evaluated() -> None:
    coverage = EvidenceCoverage()
    assert coverage.capability_domain == EvidenceStatus.NOT_EVALUATED
    assert coverage.numerical_convergence == EvidenceStatus.NOT_EVALUATED
    assert coverage.independent_solver == EvidenceStatus.NOT_EVALUATED
    assert coverage.experimental_fit == EvidenceStatus.NOT_EVALUATED


def test_from_certificate_accepted_physics() -> None:
    certificate = {
        "accepted": True,
        "task_sha256": "a" * 64,
        "veritmm_version": "1.0.0",
        "physics_audit": {"passivity_check_passed": True},
        "spectral_convergence": {"status": "passed", "passed": True},
        "material_provenance_sha256": "b" * 64,
    }
    coverage = from_certificate(certificate)
    assert coverage.capability_domain == EvidenceStatus.VERIFIED
    assert coverage.passivity == EvidenceStatus.VERIFIED
    assert coverage.numerical_convergence == EvidenceStatus.VERIFIED
    assert coverage.material_provenance == EvidenceStatus.VERIFIED
    assert coverage.reproducibility == EvidenceStatus.VERIFIED


def test_from_certificate_cross_solver_verified() -> None:
    certificate = {
        "accepted": True,
        "cross_solver_check": {"available": True, "agreement": True},
    }
    assert from_certificate(certificate).independent_solver == EvidenceStatus.VERIFIED


def test_from_certificate_cross_solver_unavailable() -> None:
    certificate = {
        "accepted": True,
        "cross_solver_check": {"available": False},
    }
    assert from_certificate(certificate).independent_solver == EvidenceStatus.UNAVAILABLE


def test_from_certificate_high_precision_not_triggered() -> None:
    certificate = {
        "accepted": True,
        "high_precision_referee": {"triggered": False, "status": "not_triggered"},
    }
    assert from_certificate(certificate).high_precision_referee == EvidenceStatus.NOT_EVALUATED


def test_from_certificate_high_precision_verified() -> None:
    certificate = {
        "accepted": True,
        "high_precision_referee": {"triggered": True, "passed": True},
    }
    assert from_certificate(certificate).high_precision_referee == EvidenceStatus.VERIFIED


def test_from_certificate_high_precision_unavailable() -> None:
    certificate = {
        "accepted": True,
        "high_precision_referee": {"triggered": True, "status": "unavailable"},
    }
    assert from_certificate(certificate).high_precision_referee == EvidenceStatus.UNAVAILABLE


def test_experimental_fit_defaults_to_not_evaluated() -> None:
    assert from_certificate({"accepted": True}).experimental_fit == EvidenceStatus.NOT_EVALUATED


def test_coverage_is_json_serializable_and_round_trips() -> None:
    coverage = EvidenceCoverage(
        capability_domain=EvidenceStatus.VERIFIED,
        independent_solver=EvidenceStatus.VERIFIED,
    )
    payload = coverage.model_dump(mode="json")
    assert payload["capability_domain"] == "verified"
    assert payload["independent_solver"] == "verified"
    assert payload["experimental_fit"] == "not_evaluated"
    assert json.loads(json.dumps(payload)) == payload
    assert EvidenceCoverage.model_validate(payload) == coverage


def test_old_certificate_is_additive_compatible() -> None:
    old_certificate = {
        "accepted": True,
        "limitations": [],
        "task_hash": "abc123",
    }
    typed = PhysicsCertificate.model_validate(old_certificate)
    assert typed.evidence_coverage is None
    assert from_certificate(old_certificate).capability_domain == EvidenceStatus.VERIFIED
