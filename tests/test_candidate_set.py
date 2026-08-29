from __future__ import annotations

import json

import numpy as np

from tmm_engine.research import CandidateSet, DiversityMetric, spectral_distance_matrix


def test_pareto_filter_removes_dominated_candidates() -> None:
    candidates = [
        {"name": "balanced", "objectives": {"throughput": 1.0, "loss": 1.0}},
        {"name": "dominated", "objectives": {"throughput": 0.5, "loss": 0.5}},
        {"name": "throughput", "objectives": {"throughput": 2.0, "loss": 0.5}},
    ]

    result = CandidateSet.pareto_filter(
        candidates,
        {"throughput": "maximize", "loss": "maximize"},
    )

    assert [item["name"] for item in result.candidates] == ["balanced", "throughput"]
    assert result.pareto_front_indices == [0, 2]
    assert result.provenance.source_method == "pareto_archive"


def test_deduplication_merges_candidates_within_tolerance() -> None:
    candidates = [
        {"normalized_design": [0.0, 0.0]},
        {"normalized_design": [0.0005, 0.0004]},
        {"normalized_design": [0.1, 0.0]},
    ]

    result = CandidateSet.deduplicate(candidates, tolerance=1e-3)

    assert len(result.candidates) == 2
    assert result.provenance.deduplicated is True
    assert result.provenance.deduplication_tolerance == 1e-3


def test_distance_matrix_is_symmetric_with_zero_diagonal() -> None:
    candidates = [
        {"normalized_design": [0.0, 0.0], "spectrum": [0.0, 1.0]},
        {"normalized_design": [0.5, 0.0], "spectrum": [0.5, 0.5]},
        {"normalized_design": [1.0, 0.0], "spectrum": [1.0, 0.0]},
    ]

    spectral = spectral_distance_matrix(candidates)
    structural = CandidateSet.structural_distance_matrix(candidates)

    np.testing.assert_allclose(spectral, spectral.T)
    np.testing.assert_allclose(structural, structural.T)
    np.testing.assert_allclose(np.diag(spectral), 0.0)
    np.testing.assert_allclose(np.diag(structural), 0.0)
    assert spectral[0, 1] == spectral[1, 0]


def test_diversity_metrics_and_provenance_are_json_serializable() -> None:
    candidate_set = CandidateSet(
        candidates=[{"normalized_design": [0.25]}],
        diversity_metrics=[
            DiversityMetric(metric="structural_distance", value=0.25, threshold=1e-3)
        ],
    )

    payload = candidate_set.model_dump(mode="json")
    restored = CandidateSet.model_validate(json.loads(json.dumps(payload)))

    assert restored == candidate_set
    assert restored.provenance.source_method == "explicit"
