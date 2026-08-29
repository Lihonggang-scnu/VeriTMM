"""Typed uncertainty budget following GUM/NIST measurement principles.

Each component is tracked separately.  Applicability gaps are categorical:
they describe physics that is not modeled and must never be collapsed into a
misleading numerical standard uncertainty.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class UncertaintyType(str, Enum):
    """Type-A statistical evaluation versus Type-B other information."""

    TYPE_A = "type_a"
    TYPE_B = "type_b"


class UncertaintyComponent(BaseModel):
    """One identified numerical uncertainty source."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(description="Human-readable uncertainty source.")
    uncertainty_type: UncertaintyType
    value: float | None = Field(
        default=None,
        description="Standard uncertainty, expressed as a one-sigma equivalent when applicable.",
    )
    unit: str | None = Field(default=None, description="Physical unit, if applicable.")
    distribution: str = Field(
        default="normal",
        description="normal, uniform, triangular, or unknown",
    )
    sensitivity_coefficient: float | None = Field(
        default=None,
        description="Sensitivity coefficient dy/dx when available.",
    )
    degrees_of_freedom: int | None = Field(
        default=None,
        description="Effective degrees of freedom for a Type-A estimate.",
    )
    notes: str | None = None


class ApplicabilityGap(BaseModel):
    """Categorical limitation for physics outside the modeled domain."""

    model_config = ConfigDict(extra="forbid")

    limitation: str = Field(description="What is not modeled.")
    severity: str = Field(description="minor, moderate, major, or blocking")
    mitigation: str | None = Field(default=None, description="Possible user action.")


class UncertaintyBudget(BaseModel):
    """Auditable ledger of numerical uncertainty sources and model gaps."""

    model_config = ConfigDict(extra="forbid")

    numerical_components: list[UncertaintyComponent] = Field(
        default_factory=list,
        description="Grid discretization, solver tolerance, and finite precision.",
    )
    material_components: list[UncertaintyComponent] = Field(
        default_factory=list,
        description="n,k dataset uncertainty and interpolation error.",
    )
    parameter_components: list[UncertaintyComponent] = Field(
        default_factory=list,
        description="Thickness, angle, and other design-parameter uncertainty.",
    )
    sampling_components: list[UncertaintyComponent] = Field(
        default_factory=list,
        description="Finite-sample uncertainty in Monte Carlo statistics.",
    )
    applicability_gaps: list[ApplicabilityGap] = Field(
        default_factory=list,
        description="Categorical physics limitations, not numerical uncertainties.",
    )
    combined_standard_uncertainty: float | None = Field(
        default=None,
        description="Optional RSS combination when the propagation assumptions are explicit.",
    )
    covariance_matrix: dict[str, Any] | None = Field(
        default=None,
        description="Optional correlated-parameter covariance representation.",
    )

    def has_quantitative_components(self) -> bool:
        """Return whether the budget contains quantitative uncertainty evidence."""

        return bool(
            self.numerical_components
            or self.material_components
            or self.parameter_components
            or self.sampling_components
        )


def _component_from_parameter(
    index: int,
    value: Any,
    *,
    notes: str,
) -> UncertaintyComponent:
    if isinstance(value, Mapping):
        layer_index = value.get("layer_index", index)
        coefficient = value.get("sensitivity_coefficient")
        if coefficient is None:
            coefficient = value.get("autodiff_derivative_per_nm")
        if coefficient is None:
            coefficient = value.get("finite_difference_derivative_per_nm")
        return UncertaintyComponent(
            source=f"thickness_layer_{layer_index}",
            uncertainty_type=UncertaintyType.TYPE_B,
            value=value.get("uncertainty") or value.get("sigma_nm"),
            unit="nm" if value.get("sigma_nm") is not None else None,
            distribution=str(value.get("distribution", "unknown")),
            sensitivity_coefficient=coefficient,
            notes=notes,
        )
    return UncertaintyComponent(
        source=f"thickness_layer_{index}",
        uncertainty_type=UncertaintyType.TYPE_B,
        sensitivity_coefficient=float(value),
        notes=notes,
    )


def from_sensitivity_result(sens: Mapping[str, Any]) -> UncertaintyBudget:
    """Construct a budget from sensitivity, tolerance, or legacy artifacts."""

    if not isinstance(sens, Mapping):
        raise TypeError("sensitivity result must be a mapping")
    budget = UncertaintyBudget()

    legacy_sensitivity = sens.get("thickness_sensitivity")
    if isinstance(legacy_sensitivity, list):
        budget.parameter_components.extend(
            _component_from_parameter(
                index,
                value,
                notes="Finite-difference and autodiff sensitivity source.",
            )
            for index, value in enumerate(legacy_sensitivity)
        )

    parameters = sens.get("parameters")
    if isinstance(parameters, list):
        budget.parameter_components.extend(
            _component_from_parameter(
                index,
                value,
                notes="Sensitivity result with independent derivative audit.",
            )
            for index, value in enumerate(parameters)
            if isinstance(value, Mapping)
        )

    uncertainties = sens.get("uncertainties")
    if isinstance(uncertainties, list):
        for item in uncertainties:
            if not isinstance(item, Mapping):
                continue
            layer_index = item.get("layer_index", len(budget.parameter_components))
            value = item.get("sigma_nm")
            if value is None:
                value = item.get("half_width_nm")
            budget.parameter_components.append(
                UncertaintyComponent(
                    source=f"thickness_layer_{layer_index}",
                    uncertainty_type=UncertaintyType.TYPE_B,
                    value=value,
                    unit="nm" if value is not None else None,
                    distribution=str(item.get("distribution", "unknown")),
                    notes="Declared tolerance distribution.",
                )
            )

    sample_count = sens.get("sample_count")
    if sample_count is None:
        sample_count = sens.get("requested_sample_count")
    if sample_count is None and isinstance(sens.get("samples"), list):
        sample_count = len(sens["samples"])
    if isinstance(sample_count, int) and not isinstance(sample_count, bool) and sample_count > 0:
        budget.sampling_components.append(
            UncertaintyComponent(
                source="monte_carlo_finite_sample",
                uncertainty_type=UncertaintyType.TYPE_A,
                degrees_of_freedom=max(0, sample_count - 1),
                notes="Finite-sample statistical uncertainty.",
            )
        )

    covariance = sens.get("covariance_matrix")
    if isinstance(covariance, Mapping):
        budget.covariance_matrix = dict(covariance)
    combined = sens.get("combined_standard_uncertainty")
    if isinstance(combined, (int, float)) and not isinstance(combined, bool):
        budget.combined_standard_uncertainty = float(combined)
    return budget


def applicability_gaps_from_certificate(
    certificate: Mapping[str, Any],
) -> list[ApplicabilityGap]:
    """Map unsupported capability failures to categorical model limitations."""

    assessment = certificate.get("capability_assessment")
    failures = assessment.get("failures", []) if isinstance(assessment, Mapping) else []
    gaps: list[ApplicabilityGap] = []
    seen: set[str] = set()
    for failure in failures:
        if not isinstance(failure, Mapping):
            continue
        code = str(failure.get("code", ""))
        context = failure.get("context")
        context = context if isinstance(context, Mapping) else {}
        if code == "unsupported_material_model":
            material_class = str(context.get("material_class", "anisotropic_materials"))
            limitation = (
                "anisotropy_not_modeled"
                if material_class == "anisotropic"
                else f"{material_class}_material_model_not_modeled"
            )
            gap = ApplicabilityGap(
                limitation=limitation,
                severity="blocking",
                mitigation="Use a supported passive isotropic scalar-nk material model or another solver family.",
            )
        elif code == "unsupported_geometry":
            gap = ApplicabilityGap(
                limitation="non_planar_or_lateral_geometry_not_modeled",
                severity="blocking",
                mitigation="Use a solver that supports the requested geometry.",
            )
        elif code == "unsupported_excitation":
            gap = ApplicabilityGap(
                limitation="non_plane_wave_excitation_not_modeled",
                severity="blocking",
                mitigation="Use a solver that supports the requested source model.",
            )
        elif code == "time_domain_required":
            gap = ApplicabilityGap(
                limitation="time_domain_response_not_modeled",
                severity="blocking",
                mitigation="Use a time-domain solver for the requested response.",
            )
        else:
            continue
        if gap.limitation not in seen:
            seen.add(gap.limitation)
            gaps.append(gap)
    return gaps


__all__ = [
    "ApplicabilityGap",
    "UncertaintyBudget",
    "UncertaintyComponent",
    "UncertaintyType",
    "applicability_gaps_from_certificate",
    "from_sensitivity_result",
]
