from __future__ import annotations

import copy
from pathlib import Path

from tmm_engine.agent_bench.interactive_env import (
    EnvState,
    InteractiveEnv,
    load_interactive_case,
)
from tmm_engine.agent_bench.run_interactive import _reference_policy, run_cases
from tmm_engine.agent_harness import run_interactive_episode

ROOT = Path("benchmarks/cases/interactive")


def _case(name: str):
    return load_interactive_case(ROOT / f"{name}.json")


def test_state_machine_transitions_discover_repair_run_inspect_decide() -> None:
    case = _case("invalid_then_repair")
    environment = InteractiveEnv(case)
    states = [environment.state]

    environment.step({"action": "submit_task", "task": case.initial_task})
    states.append(environment.state)
    environment.step("repair_task")
    states.append(environment.state)
    environment.step({"action": "submit_task", "task": case.repair_task})
    states.append(environment.state)
    environment.step("run")
    states.append(environment.state)
    environment.step("execute")
    states.append(environment.state)
    environment.step("inspect_evidence")
    states.append(environment.state)
    environment.step("decide_accept")
    states.append(environment.state)
    environment.step("confirm_accept")
    states.append(environment.state)

    assert states == [
        EnvState.DISCOVER,
        EnvState.PREFLIGHT,
        EnvState.REPAIR,
        EnvState.PREFLIGHT,
        EnvState.RUN,
        EnvState.INSPECT_EVIDENCE,
        EnvState.INSPECT_EVIDENCE,
        EnvState.DECIDE,
        EnvState.DONE,
    ]
    episode = environment.episode()
    assert episode.final_outcome == "success"
    assert episode.steps[0].step_score == 0.0
    assert episode.steps[2].step_score == 0.8


def test_reference_policy_completes_all_five_cases_and_release_gate() -> None:
    cases = [load_interactive_case(path) for path in sorted(ROOT.glob("*.json"))]
    result = run_cases(cases)

    assert result["episode_count"] == 5
    assert result["successful_episode_count"] == 5
    assert result["unsupported_false_accept_rate"] == 0.0
    assert result["safe_reject_rate"] == 1.0
    assert result["release_gate_passed"] is True


def test_invalid_repair_receives_typed_failure_and_repair_score() -> None:
    case = _case("invalid_then_repair")
    episode = run_interactive_episode(case, _reference_policy(case))

    assert episode.final_outcome == "success"
    assert episode.steps[0].observation["typed_failures"]
    assert episode.steps[0].step_verdict == "incorrect"
    repaired = next(step for step in episode.steps if step.action == "submit_task" and step.step_score == 0.8)
    assert repaired.state == EnvState.PREFLIGHT


def test_silent_material_substitution_sets_unsupported_false_accept() -> None:
    case = _case("material_unavailable")
    replacement = copy.deepcopy(case.initial_task)
    replacement["simulation"]["stack"]["layers"][0]["material"] = "Si3N4"
    environment = InteractiveEnv(case)
    environment.step({"action": "submit_task", "task": case.initial_task})
    environment.step({"action": "submit_task", "task": replacement})
    environment.step("confirm_accept")

    episode = environment.episode()
    assert episode.final_outcome == "fail"
    assert episode.unsupported_false_accept is True
    assert episode.terminal_state == EnvState.DONE


def test_low_margin_challenge_is_required_before_accept() -> None:
    case = _case("low_margin_needs_challenge")
    episode = run_interactive_episode(case, _reference_policy(case))

    assert episode.final_outcome == "success"
    challenge_steps = [step for step in episode.steps if step.action == "request_challenge"]
    assert challenge_steps and challenge_steps[0].step_score == 1.0


def test_non_identifiable_fit_requests_more_measurements() -> None:
    case = _case("non_identifiable_fit")
    episode = run_interactive_episode(case, _reference_policy(case))

    assert episode.final_outcome == "success"
    assert any(step.action == "request_measurements" for step in episode.steps)


def test_accept_without_evidence_inspection_is_penalized() -> None:
    case = _case("low_margin_needs_challenge")
    environment = InteractiveEnv(case)
    environment.step({"action": "submit_task", "task": case.initial_task})
    environment.step("run")
    environment.step("execute")
    decision = environment.step("decide_accept")
    environment.step("confirm_accept")

    episode = environment.episode()
    assert decision.step_score == 0.2
    assert decision.step_verdict == "incorrect"
    assert episode.final_outcome == "fail"


def test_safe_reject_on_anisotropy_is_terminal_success() -> None:
    case = _case("unsupported_anisotropy")
    environment = InteractiveEnv(case)
    environment.step({"action": "submit_task", "task": case.initial_task})
    step = environment.step(
        {
            "action": "reject_unsupported",
            "explanation": "Tensor anisotropy is outside the scalar isotropic solver domain.",
        }
    )

    episode = environment.episode()
    assert step.state == EnvState.SAFE_REJECT
    assert episode.final_outcome == "success"
    assert episode.terminal_state == EnvState.SAFE_REJECT
    assert episode.unsupported_false_accept is False
