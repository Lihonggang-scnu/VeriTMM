from __future__ import annotations

from tmm_engine.capabilities import (
    FailureCode,
    FailureRecord,
    enrich_failure_actions,
)


def test_recoverable_failures_have_navigation_without_silent_scientific_patch() -> None:
    cases = [
        FailureCode.UNSUPPORTED_GEOMETRY,
        FailureCode.UNSUPPORTED_MATERIAL_MODEL,
        FailureCode.UNSUPPORTED_EXCITATION,
        FailureCode.TIME_DOMAIN_REQUIRED,
        FailureCode.UNSUPPORTED_OUTPUT_COMBINATION,
        FailureCode.REQUESTED_OUTPUT_MISSING,
        FailureCode.MATERIAL_NOT_FOUND,
        FailureCode.MATERIAL_AMBIGUITY,
        FailureCode.MATERIAL_RANGE_ERROR,
        FailureCode.OPTIONAL_DEPENDENCY_MISSING,
        FailureCode.SPECTRAL_CONVERGENCE_FAILURE,
        FailureCode.SOLVER_DISAGREEMENT,
        FailureCode.OPTIMIZER_FAILURE,
        FailureCode.BUDGET_EXHAUSTED,
    ]
    for code in cases:
        enriched = enrich_failure_actions(FailureRecord(code, "test", True))
        assert enriched.actions, code
        for action in enriched.actions:
            if action.safety != "safe":
                assert not action.patch


def test_nonrecoverable_failure_is_not_decorated_with_fake_action() -> None:
    failure = FailureRecord(FailureCode.PASSIVITY_VIOLATION, "bad", False)
    assert enrich_failure_actions(failure).actions == ()
