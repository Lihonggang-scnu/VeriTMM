"""Tests for the tightest_margin field in physics acceptance certificates."""

from __future__ import annotations

import pytest

from tmm_engine.acceptance import AcceptanceSettings, _compute_tightest_margin


class TestComputeTightestMargin:
    def test_returns_none_when_no_checks_available(self) -> None:
        settings = AcceptanceSettings()
        result = _compute_tightest_margin({}, {"status": "not_requested"}, settings)
        assert result is None

    def test_returns_none_for_unavailable_cross_solver(self) -> None:
        audit = {"energy_conservation_max_abs_error": 1e-9}
        settings = AcceptanceSettings()
        result = _compute_tightest_margin(audit, {"status": "unavailable"}, settings)
        # Only energy check — should return energy
        assert result is not None
        assert result["check"] == "energy_conservation"

    def test_energy_conservation_check(self) -> None:
        audit = {"energy_conservation_max_abs_error": 5e-8}
        settings = AcceptanceSettings(energy_tolerance=1e-7)
        result = _compute_tightest_margin(audit, {"status": "not_requested"}, settings)
        assert result is not None
        assert result["check"] == "energy_conservation"
        assert result["observed_value"] == pytest.approx(5e-8)
        assert result["acceptance_limit"] == pytest.approx(1e-7)
        assert result["distance_to_limit"] == pytest.approx(5e-8)
        assert result["normalized_margin"] == pytest.approx(0.5)

    def test_tightest_check_is_cross_solver_when_closer(self) -> None:
        # energy: observed=1e-9, limit=1e-7 → margin=99e-9, normalized=0.99
        # cross:  observed=9.5e-8, limit=1e-7 → margin=0.5e-8, normalized=0.05
        audit = {"energy_conservation_max_abs_error": 1e-9}
        cross = {"status": "passed", "maximum_absolute_difference": 9.5e-8}
        settings = AcceptanceSettings(energy_tolerance=1e-7, cross_solver_tolerance=1e-7)
        result = _compute_tightest_margin(audit, cross, settings)
        assert result is not None
        assert result["check"] == "cross_solver_agreement"
        assert result["observed_value"] == pytest.approx(9.5e-8)

    def test_tightest_check_is_energy_when_closer(self) -> None:
        # energy: actual=9.9e-8, threshold=1e-7 → rel=0.01
        # cross:  actual=1e-9,   threshold=1e-7 → rel=0.99
        audit = {"energy_conservation_max_abs_error": 9.9e-8}
        cross = {"status": "passed", "maximum_absolute_difference": 1e-9}
        settings = AcceptanceSettings(energy_tolerance=1e-7, cross_solver_tolerance=1e-7)
        result = _compute_tightest_margin(audit, cross, settings)
        assert result is not None
        assert result["check"] == "energy_conservation"

    def test_result_has_all_fields(self) -> None:
        audit = {"energy_conservation_max_abs_error": 2e-8}
        settings = AcceptanceSettings(energy_tolerance=1e-7)
        result = _compute_tightest_margin(audit, {"status": "not_requested"}, settings)
        assert result is not None
        for key in ("check", "observed_value", "acceptance_limit", "distance_to_limit", "normalized_margin"):
            assert key in result

    def test_nonfinite_energy_is_ignored(self) -> None:
        audit = {"energy_conservation_max_abs_error": float("inf")}
        settings = AcceptanceSettings()
        result = _compute_tightest_margin(audit, {"status": "not_requested"}, settings)
        assert result is None
