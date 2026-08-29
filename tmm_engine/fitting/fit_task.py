"""Experimental fitting contracts for measured optical data."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MeasurementType(str, Enum):
    """Supported scalar reflectance, transmittance, and ellipsometry data."""

    REFLECTANCE = "R"
    TRANSMITTANCE = "T"
    ELLIPSOMETRY_PSI = "Ψ"
    ELLIPSOMETRY_DELTA = "Δ"


class MeasuredDataPoint(BaseModel):
    """One experimental observation."""

    model_config = ConfigDict(extra="forbid")

    wavelength_nm: float = Field(gt=0)
    angle_deg: float = Field(default=0.0, ge=0.0, lt=90.0)
    polarization: Literal["s", "p", "unpolarized"] = "unpolarized"
    measurement_type: MeasurementType
    value: float
    uncertainty: float | None = Field(default=None, gt=0.0)
    weight: float = Field(default=1.0, gt=0.0)


class FitParameter(BaseModel):
    """One bounded parameter to fit; v1 maps parameters to layer thickness."""

    model_config = ConfigDict(extra="forbid")

    name: str
    layer_index: int | None = Field(default=None, ge=0)
    bounds: tuple[float, float]
    initial_guess: float | None = None

    @model_validator(mode="after")
    def _validate_bounds(self) -> "FitParameter":
        low, high = self.bounds
        if low >= high:
            raise ValueError("fit parameter bounds must be strictly increasing")
        if self.initial_guess is not None and not low <= self.initial_guess <= high:
            raise ValueError("initial_guess must lie inside fit parameter bounds")
        if not self.name.strip():
            raise ValueError("fit parameter name must be non-empty")
        return self


class FitTask(BaseModel):
    """Inverse problem contract: measured data plus bounded fit parameters."""

    model_config = ConfigDict(extra="forbid")

    structure: dict[str, Any] = Field(description="Fixed structure and material definition.")
    measurements: list[MeasuredDataPoint] = Field(min_length=1)
    fit_parameters: list[FitParameter] = Field(min_length=1)
    method: Literal["least_squares", "bounded_lm"] = "least_squares"
    max_iterations: int = Field(default=100, ge=1)
    tolerance: float = Field(default=1e-6, gt=0.0)

    @model_validator(mode="after")
    def _validate_parameter_names(self) -> "FitTask":
        names = [item.name for item in self.fit_parameters]
        if len(names) != len(set(names)):
            raise ValueError("fit parameter names must be unique")
        return self


class IdentifiabilityReport(BaseModel):
    """Fit quality plus local parameter-identifiability evidence."""

    model_config = ConfigDict(extra="forbid")

    rmse: float
    chi_squared: float | None = None
    degrees_of_freedom: int
    jacobian_condition_number: float
    singular_values: list[float]
    effective_rank: int
    parameter_correlation_matrix: list[list[float]]
    identifiability_status: Literal[
        "well_determined", "weakly_identifiable", "non_identifiable"
    ]
    alternative_fits: list[dict[str, Any]] | None = None


class FitResult(BaseModel):
    """Result of a fit; fit quality is not a physics certificate."""

    model_config = ConfigDict(extra="forbid")

    task: FitTask
    converged: bool
    iterations: int
    best_fit_parameters: dict[str, float]
    residuals: list[float]
    # Jacobian of the weighted residual vector at the fitted point.  This is
    # optional so historical FitResult JSON remains readable; measurement
    # planning can reconstruct it from the forward model when absent.
    jacobian: list[list[float]] | None = None
    identifiability: IdentifiabilityReport
    fit_certificate: dict[str, Any]


__all__ = [
    "FitParameter",
    "FitResult",
    "FitTask",
    "IdentifiabilityReport",
    "MeasuredDataPoint",
    "MeasurementType",
]
