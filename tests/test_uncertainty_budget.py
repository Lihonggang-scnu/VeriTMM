"""Tests for the typed UncertaintyBudget ledger."""

from __future__ import annotations

import json

from tmm_engine.protocol.evidence import EvidenceStatus
from tmm_engine.protocol.evidence import from_certificate as coverage_from_certificate
from tmm_engine.protocol.uncertainty_budget import (
    ApplicabilityGap,
    UncertaintyBudget,
    UncertaintyComponent,
    UncertaintyType,
    applicability_gaps_from_certificate,
    from_sensitivity_result,
)


def test_empty_budget() -> None:
    budget = UncertaintyBudget()
    assert not budget.numerical_components
    assert not budget.applicability_gaps
    assert budget.combined_standard_uncertainty is None
    assert not budget.has_quantitative_components()


def test_add_numerical_component() -> None:
    budget = UncertaintyBudget()
    budget.numerical_components.append(
        UncertaintyComponent(
            source="spectral_grid_discretization",
            uncertainty_type=UncertaintyType.TYPE_B,
            value=0.001,
            unit="dimensionless",
            distribution="uniform",
        )
    )
    assert budget.has_quantitative_components()
    assert budget.numerical_components[0].source == "spectral_grid_discretization"


def test_applicability_gap_is_categorical_not_numerical() -> None:
    gap = ApplicabilityGap(limitation="anisotropy_not_modeled", severity="major")
    assert gap.limitation == "anisotropy_not_modeled"
    assert gap.severity == "major"
    assert not hasattr(gap, "value")


def test_from_legacy_sensitivity_result() -> None:
    budget = from_sensitivity_result(
        {"thickness_sensitivity": [0.05, 0.12, 0.03], "sample_count": 1000}
    )
    assert len(budget.parameter_components) == 3
    assert budget.parameter_components[0].source == "thickness_layer_0"
    assert budget.parameter_components[0].sensitivity_coefficient == 0.05
    assert len(budget.sampling_components) == 1
    assert budget.sampling_components[0].degrees_of_freedom == 999


def test_from_current_sensitivity_parameters() -> None:
    budget = from_sensitivity_result(
        {
            "parameters": [
                {
                    "layer_index": 2,
                    "autodiff_derivative_per_nm": 0.25,
                }
            ]
        }
    )
    assert budget.parameter_components[0].source == "thickness_layer_2"
    assert budget.parameter_components[0].sensitivity_coefficient == 0.25


def test_from_tolerance_uncertainties_and_samples() -> None:
    budget = from_sensitivity_result(
        {
            "uncertainties": [
                {"layer_index": 0, "distribution": "uniform", "half_width_nm": 2.0}
            ],
            "sample_count": 64,
        }
    )
    assert budget.parameter_components[0].value == 2.0
    assert budget.parameter_components[0].unit == "nm"
    assert budget.parameter_components[0].distribution == "uniform"
    assert budget.sampling_components[0].degrees_of_freedom == 63


def test_covariance_is_optional() -> None:
    budget = UncertaintyBudget(
        parameter_components=[
            UncertaintyComponent(
                source="thickness_0",
                uncertainty_type=UncertaintyType.TYPE_B,
                value=1.0,
            ),
            UncertaintyComponent(
                source="thickness_1",
                uncertainty_type=UncertaintyType.TYPE_B,
                value=1.5,
            ),
        ]
    )
    assert budget.covariance_matrix is None


def test_combined_uncertainty_is_explicit() -> None:
    budget = UncertaintyBudget(combined_standard_uncertainty=0.25)
    assert budget.combined_standard_uncertainty == 0.25


def test_json_serialization() -> None:
    budget = UncertaintyBudget(
        numerical_components=[
            UncertaintyComponent(
                source="solver_tolerance",
                uncertainty_type=UncertaintyType.TYPE_B,
                value=1e-9,
            )
        ],
        applicability_gaps=[
            ApplicabilityGap(limitation="surface_roughness", severity="moderate")
        ],
    )
    payload = budget.model_dump(mode="json")
    assert payload["numerical_components"][0]["source"] == "solver_tolerance"
    assert payload["applicability_gaps"][0]["limitation"] == "surface_roughness"
    assert json.loads(json.dumps(payload)) == payload


def test_unsupported_physics_becomes_an_applicability_gap() -> None:
    certificate = {
        "capability_assessment": {
            "supported": False,
            "failures": [
                {
                    "code": "unsupported_material_model",
                    "context": {"material_class": "anisotropic"},
                }
            ],
        }
    }
    gaps = applicability_gaps_from_certificate(certificate)
    assert len(gaps) == 1
    assert gaps[0].limitation == "anisotropy_not_modeled"
    assert gaps[0].severity == "blocking"


def test_budget_marks_evidence_coverage_only_for_quantitative_components() -> None:
    budget = from_sensitivity_result({"sample_count": 10})
    certificate = {"accepted": True, "uncertainty_budget": budget.model_dump(mode="json")}
    coverage = coverage_from_certificate(certificate)
    assert coverage.uncertainty_quantified == EvidenceStatus.VERIFIED

    gap_only = UncertaintyBudget(
        applicability_gaps=[ApplicabilityGap(limitation="anisotropy", severity="blocking")]
    )
    coverage = coverage_from_certificate(
        {"accepted": False, "uncertainty_budget": gap_only.model_dump(mode="json")}
    )
    assert coverage.uncertainty_quantified == EvidenceStatus.NOT_EVALUATED
