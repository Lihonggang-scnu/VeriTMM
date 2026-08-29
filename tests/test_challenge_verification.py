"""Tests for active challenge verification."""

from __future__ import annotations

import json
from pathlib import Path

from tmm_engine.cli import main as cli_main
from tmm_engine.verifier.challenge import (
    ChallengeObjective,
    ChallengeSpec,
    generate_challenge_candidate,
    run_challenge_search,
)


def test_challenge_spec_defaults() -> None:
    spec = ChallengeSpec(seed=42)
    assert spec.budget == 100
    assert spec.objective == ChallengeObjective.MIN_MARGIN


def test_generate_candidate_is_deterministic() -> None:
    spec = ChallengeSpec(seed=123)
    assert generate_challenge_candidate(spec, 0) == generate_challenge_candidate(spec, 0)


def test_generate_candidate_changes_with_iteration() -> None:
    spec = ChallengeSpec(seed=123)
    assert generate_challenge_candidate(spec, 0) != generate_challenge_candidate(spec, 1)


def test_challenge_search_respects_budget() -> None:
    result = run_challenge_search(ChallengeSpec(seed=42, budget=5))
    assert result.candidates_evaluated == 5
    assert result.trajectory_sha256


def test_challenge_search_trajectory_is_replayable() -> None:
    spec = ChallengeSpec(seed=42, budget=3)
    first = run_challenge_search(spec)
    second = run_challenge_search(spec)
    assert first.trajectory_sha256 == second.trajectory_sha256
    assert first.worst_candidate == second.worst_candidate


def test_challenge_search_finds_a_margin_within_budget() -> None:
    result = run_challenge_search(ChallengeSpec(seed=42, budget=10))
    assert result.candidates_evaluated == 10
    assert result.worst_margin is not None
    assert result.certificate is not None
    assert result.worst_candidate is not None


def test_challenge_cli_writes_machine_result(tmp_path: Path, capsys) -> None:
    output = tmp_path / "challenge_result.json"
    assert (
        cli_main(
            [
                "challenge",
                "--seed",
                "7",
                "--budget",
                "2",
                "--objective",
                "min_margin",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["candidates_evaluated"] == 2
    assert payload["certificate"] is not None
    assert payload["canonical_task_path"]
    assert Path(payload["canonical_task_path"]).is_file()
    assert json.loads(capsys.readouterr().out)["candidates_evaluated"] == 2
