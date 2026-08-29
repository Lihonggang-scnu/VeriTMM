"""Run the deterministic reference policy over interactive AgentBench cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..agent_harness import run_interactive_episode
from .interactive_env import InteractiveCase, InteractiveEnvEpisode, load_interactive_case


def _reference_policy(case: InteractiveCase):
    actions = list(case.reference_actions)
    index = 0

    def policy(_request: dict[str, Any]) -> str | dict[str, Any]:
        nonlocal index
        if index >= len(actions):
            return "confirm_reject"
        action = actions[index]
        index += 1
        return action

    return policy


def run_cases(cases: list[InteractiveCase], *, max_steps: int = 20) -> dict[str, Any]:
    episodes: list[InteractiveEnvEpisode] = [
        run_interactive_episode(case, _reference_policy(case), max_steps=max_steps)
        for case in sorted(cases, key=lambda item: item.case_id)
    ]
    false_accepts = sum(int(item.unsupported_false_accept) for item in episodes)
    unsupported = sum(
        int(bool(item.ground_truth.get("requires_unsupported_physics"))) for item in episodes
    )
    safe_rejects = sum(
        int(item.terminal_state.value == "safe_reject" and item.final_outcome == "success")
        for item in episodes
        if bool(item.ground_truth.get("expected_decision") == "reject")
    )
    return {
        "schema_version": "veritmm-interactive-bench-v1",
        "episode_count": len(episodes),
        "successful_episode_count": sum(item.final_outcome == "success" for item in episodes),
        "unsupported_false_accept_count": false_accepts,
        "unsupported_case_count": unsupported,
        "unsupported_false_accept_rate": float(false_accepts / max(1, len(episodes))),
        "safe_reject_count": safe_rejects,
        "safe_reject_denominator": sum(
            int(item.ground_truth.get("expected_decision") == "reject") for item in episodes
        ),
        "safe_reject_rate": float(safe_rejects / max(1, sum(
            int(item.ground_truth.get("expected_decision") == "reject") for item in episodes
        ))),
        "release_gate_passed": false_accepts == 0,
        "episodes": [item.model_dump(mode="json") for item in episodes],
    }


def _load_cases(case: str | None, cases_dir: str | None) -> list[InteractiveCase]:
    if case:
        return [load_interactive_case(case)]
    root = Path(cases_dir or "benchmarks/cases/interactive")
    return [load_interactive_case(path) for path in sorted(root.glob("*.json"))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default=None)
    parser.add_argument("--cases-dir", default=None)
    parser.add_argument("--output", type=Path, default=Path("interactive_bench_results.json"))
    parser.add_argument("--max-steps", type=int, default=20)
    args = parser.parse_args()
    cases = _load_cases(args.case, args.cases_dir)
    if not cases:
        raise ValueError("no interactive cases found")
    result = run_cases(cases, max_steps=args.max_steps)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "episodes"}, sort_keys=True))
    return 0 if result["release_gate_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
