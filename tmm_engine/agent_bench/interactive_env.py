"""Interactive stepwise AgentBench environment.

Agents can discover, attempt, inspect evidence, challenge, refine, and decide
instead of submitting one opaque final answer.  The environment is benchmark
infrastructure only; it never relaxes a physics capability boundary.
"""

from __future__ import annotations

import json
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from ..execution import ExecutionSettings, execute_task
from ..preflight import preflight_path
from ..run_artifacts import write_json
from ..task_io import load_task

INTERACTIVE_CASE_SCHEMA_VERSION = "veritmm-interactive-case-v1"


class EnvState(str, Enum):
    """Current state of the interactive benchmark environment."""

    DISCOVER = "discover"
    PREFLIGHT = "preflight"
    REPAIR = "repair"
    RUN = "run"
    INSPECT_EVIDENCE = "inspect_evidence"
    CHALLENGE = "challenge"
    DECIDE = "decide"
    SAFE_REJECT = "safe_reject"
    DONE = "done"


class InteractiveCase(BaseModel):
    """Portable scenario definition for one interactive episode."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[INTERACTIVE_CASE_SCHEMA_VERSION] = (
        INTERACTIVE_CASE_SCHEMA_VERSION
    )
    case_id: str = Field(min_length=1)
    natural_language_task: str = Field(min_length=1)
    scenario: str = Field(min_length=1)
    initial_task: dict[str, Any] | None = Field(
        default=None,
        validation_alias=AliasChoices("initial_task", "task"),
    )
    repair_task: dict[str, Any] | None = None
    run_observation: dict[str, Any] = Field(default_factory=dict)
    ground_truth: dict[str, Any] = Field(default_factory=dict)
    reference_actions: list[str | dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _task_or_observation_required(self) -> "InteractiveCase":
        if self.initial_task is None and not self.run_observation:
            raise ValueError("interactive case requires initial_task or run_observation")
        return self


class InteractiveEnvStep(BaseModel):
    """One scored transition in an interactive episode."""

    model_config = ConfigDict(extra="forbid")

    state: EnvState
    action: str = Field(min_length=1)
    observation: dict[str, Any] = Field(default_factory=dict)
    step_score: float = Field(default=0.0, ge=0.0, le=1.0)
    step_verdict: Literal["correct", "incorrect", "neutral"] = "neutral"


class InteractiveEnvEpisode(BaseModel):
    """Full episode trajectory with checkpoint scores."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["veritmm-interactive-episode-v1"] = (
        "veritmm-interactive-episode-v1"
    )
    case_id: str
    ground_truth: dict[str, Any]
    steps: list[InteractiveEnvStep]
    final_outcome: Literal["success", "fail", "timeout"]
    final_score: float = Field(ge=0.0, le=1.0)
    unsupported_false_accept: bool = False
    terminal_state: EnvState


class InteractiveAgentPolicy(Protocol):
    """One-step agent adapter receiving state, observation, and actions."""

    def __call__(self, request: Mapping[str, Any]) -> str | Mapping[str, Any]: ...


class InteractiveEnv:
    """Deterministic state machine with fail-closed checkpoint scoring."""

    def __init__(self, case: InteractiveCase | Mapping[str, Any], *, max_steps: int = 20) -> None:
        self.case = case if isinstance(case, InteractiveCase) else InteractiveCase.model_validate(case)
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        self.max_steps = max_steps
        self.state = EnvState.DISCOVER
        self.steps: list[InteractiveEnvStep] = []
        self._current_task: dict[str, Any] | None = None
        self._last_preflight: dict[str, Any] | None = None
        self._preflight_attempts = 0
        self._valid_task = False
        self._evidence_inspected = False
        self._uncertainty_inspected = False
        self._challenge_requested = False
        self._challenge_completed = False
        self._silent_material_substitution = False
        self._final_outcome: Literal["success", "fail", "timeout"] | None = None
        self._unsupported_false_accept = False

    @property
    def done(self) -> bool:
        return self.state in {EnvState.DONE, EnvState.SAFE_REJECT}

    def available_actions(self) -> tuple[str, ...]:
        if self.state == EnvState.DISCOVER:
            return ("inspect_requirements", "submit_task", "reject_unsupported")
        if self.state == EnvState.PREFLIGHT:
            if self._valid_task:
                return ("run", "inspect_failure", "repair_task", "reject_unsupported")
            return ("repair_task", "submit_task", "reject_unsupported")
        if self.state == EnvState.REPAIR:
            return ("submit_task", "reject_unsupported")
        if self.state == EnvState.RUN:
            return ("execute",)
        if self.state == EnvState.INSPECT_EVIDENCE:
            return (
                "inspect_evidence",
                "request_challenge",
                "decide_accept",
                "decide_reject",
                "request_measurements",
            )
        if self.state == EnvState.CHALLENGE:
            return ("run_challenge", "inspect_evidence", "decide_reject")
        if self.state == EnvState.DECIDE:
            return ("confirm_accept", "confirm_reject", "request_measurements", "reject_unsupported")
        return ()

    def request(self) -> dict[str, Any]:
        """Serialize the current state and permitted actions for an agent."""

        return {
            "case_id": self.case.case_id,
            "state": self.state.value,
            "natural_language_task": self.case.natural_language_task,
            "available_actions": list(self.available_actions()),
            "observation": self._public_observation(),
        }

    def step(self, action: str | Mapping[str, Any]) -> InteractiveEnvStep:
        """Apply one action, score it, and append a trajectory checkpoint."""

        if self.done:
            raise ValueError("interactive episode is already terminal")
        if len(self.steps) >= self.max_steps:
            self._final_outcome = "timeout"
            self.state = EnvState.DONE
            raise ValueError("interactive episode exceeded max_steps")
        name, payload = self._parse_action(action)
        previous = self.state
        observation: dict[str, Any]
        score = 0.0
        verdict: Literal["correct", "incorrect", "neutral"] = "neutral"

        if name == "inspect_requirements" and self.state == EnvState.DISCOVER:
            observation = {
                "requirements": {
                    "units": {"wavelength": "nm", "angle": "deg"},
                    "capability_boundary": "planar passive isotropic frequency-domain TMM",
                    "next": "submit_task or reject_unsupported",
                }
            }
            score, verdict = 1.0, "correct"
        elif name == "submit_task" and self.state in {
            EnvState.DISCOVER,
            EnvState.REPAIR,
            EnvState.PREFLIGHT,
        }:
            task = payload.get("task", self.case.initial_task)
            if not isinstance(task, Mapping):
                raise ValueError("submit_task requires a task object")
            observation, valid = self._preflight(dict(task), payload)
            self.state = EnvState.PREFLIGHT
            score = 1.0 if valid and self._preflight_attempts == 1 else (0.8 if valid else 0.0)
            verdict = "correct" if valid else "incorrect"
        elif name == "repair_task" and self.state == EnvState.PREFLIGHT:
            self.state = EnvState.REPAIR
            observation = {
                "typed_failures": (self._last_preflight or {}).get("failures", []),
                "next": "submit_task with a repaired task or reject_unsupported",
            }
        elif name == "inspect_failure" and self.state == EnvState.PREFLIGHT:
            observation = {"typed_failures": (self._last_preflight or {}).get("failures", [])}
            score, verdict = 1.0, "correct"
        elif name == "run" and self.state == EnvState.PREFLIGHT and self._valid_task:
            self.state = EnvState.RUN
            observation = {"ready": True, "next": "execute"}
            score, verdict = 1.0, "correct"
        elif name in {"execute", "run"} and self.state == EnvState.RUN:
            observation = self._execute()
            self.state = EnvState.INSPECT_EVIDENCE
            score, verdict = 1.0, "correct"
        elif name == "inspect_evidence" and self.state == EnvState.INSPECT_EVIDENCE:
            self._evidence_inspected = True
            self._uncertainty_inspected = self._uncertainty_inspected or bool(
                "uncertainty_budget" in self._public_observation()
            )
            observation = self._evidence_observation()
            score, verdict = 1.0, "correct"
        elif name == "request_challenge" and self.state == EnvState.INSPECT_EVIDENCE:
            self._challenge_requested = True
            self.state = EnvState.CHALLENGE
            observation = {"challenge": "requested", "next": "run_challenge"}
            score, verdict = 1.0, "correct"
        elif name in {"run_challenge", "challenge"} and self.state == EnvState.CHALLENGE:
            self._challenge_completed = True
            self.state = EnvState.INSPECT_EVIDENCE
            observation = {
                "challenge": self.case.ground_truth.get(
                    "challenge_observation", {"status": "completed"}
                )
            }
            score, verdict = 1.0, "correct"
        elif name in {"decide_accept", "decide_reject", "request_measurements"} and self.state in {
            EnvState.INSPECT_EVIDENCE,
            EnvState.CHALLENGE,
        }:
            self.state = EnvState.DECIDE
            observation = {"decision_pending": name, "next": "confirm_accept/confirm_reject/request_measurements"}
            score, verdict = self._decision_checkpoint(name)
        elif name in {"confirm_accept", "confirm_reject", "request_measurements", "reject_unsupported"} and self.state in {
            EnvState.DECIDE,
            EnvState.PREFLIGHT,
            EnvState.REPAIR,
            EnvState.DISCOVER,
        }:
            decision = (
                "accept"
                if name == "confirm_accept"
                else ("request_measurements" if name == "request_measurements" else "reject")
            )
            if name == "reject_unsupported":
                decision = "reject_unsupported"
            observation, score, verdict = self._finish(decision, payload)
        else:
            raise ValueError(
                f"action {name!r} is not available in state {self.state.value!r}; "
                f"available={self.available_actions()}"
            )

        step = InteractiveEnvStep(
            state=self.state,
            action=name,
            observation={"from_state": previous.value, **observation},
            step_score=float(score),
            step_verdict=verdict,
        )
        self.steps.append(step)
        return step

    def episode(self) -> InteractiveEnvEpisode:
        outcome = self._final_outcome
        if outcome is None:
            outcome = "timeout" if len(self.steps) >= self.max_steps else "fail"
        average = (
            sum(item.step_score for item in self.steps) / len(self.steps)
            if self.steps
            else 0.0
        )
        terminal_bonus = 0.2 if outcome == "success" else 0.0
        return InteractiveEnvEpisode(
            case_id=self.case.case_id,
            ground_truth=dict(self.case.ground_truth),
            steps=list(self.steps),
            final_outcome=outcome,
            final_score=min(1.0, float(average + terminal_bonus)),
            unsupported_false_accept=self._unsupported_false_accept,
            terminal_state=self.state,
        )

    def terminate(self, outcome: Literal["fail", "timeout"], reason: str) -> None:
        """Close an episode after a policy error or step-budget exhaustion."""

        if self.done:
            return
        self.state = EnvState.DONE
        self._final_outcome = outcome
        self.steps.append(
            InteractiveEnvStep(
                state=self.state,
                action="environment_terminate",
                observation={"reason": reason},
                step_score=0.0,
                step_verdict="incorrect",
            )
        )

    def _parse_action(self, action: str | Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        if isinstance(action, str):
            return action, {}
        if not isinstance(action, Mapping):
            raise TypeError("agent action must be a string or mapping")
        name = action.get("action")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("agent action mapping requires a non-empty action")
        return name, dict(action)

    def _preflight(self, task: dict[str, Any], payload: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
        self._preflight_attempts += 1
        self._current_task = task
        self._detect_material_substitution(task, payload.get("note", payload.get("explanation", "")))
        with tempfile.TemporaryDirectory(prefix="veritmm_interactive_preflight_") as temporary:
            path = Path(temporary) / "TASK.json"
            write_json(path, task)
            report = preflight_path(path)
        self._last_preflight = report
        self._valid_task = bool(report.get("ok"))
        return {
            "preflight": report,
            "typed_failures": report.get("failures", []),
            "next": "run" if self._valid_task else "repair_task or reject_unsupported",
        }, self._valid_task

    def _execute(self) -> dict[str, Any]:
        if self.case.run_observation:
            return dict(self.case.run_observation)
        if self._current_task is None:
            return {"status": "failed", "reason": "no_valid_task"}
        with tempfile.TemporaryDirectory(prefix="veritmm_interactive_run_") as temporary:
            root = Path(temporary)
            path = root / "TASK.json"
            write_json(path, self._current_task)
            mode, task = load_task(path)
            payload = execute_task(
                mode,
                task,
                root / "run",
                input_path=path,
                settings=ExecutionSettings(write_plot=False, convergence_max_refinements=1),
            )
        return {
            "status": payload.get("status"),
            "ok": payload.get("ok"),
            "physics": (payload.get("summary") or {}).get("physics", {}),
            "evidence_coverage": (payload.get("summary") or {}).get("evidence_coverage", {}),
            "uncertainty_budget": (payload.get("summary") or {}).get("uncertainty_budget", {}),
            "tightest_margin": (payload.get("summary") or {}).get("tightest_margin"),
        }

    def _public_observation(self) -> dict[str, Any]:
        if self.steps:
            return dict(self.steps[-1].observation)
        return {"message": "Discover task requirements before submitting a task."}

    def _evidence_observation(self) -> dict[str, Any]:
        observation = self._public_observation()
        return {
            "evidence_coverage": observation.get("evidence_coverage", {}),
            "uncertainty_budget": observation.get("uncertainty_budget", {}),
            "tightest_margin": observation.get("tightest_margin"),
            "next": "request_challenge or decide",
        }

    def _decision_checkpoint(self, decision: str) -> tuple[float, Literal["correct", "incorrect", "neutral"]]:
        expected_unsupported = bool(self.case.ground_truth.get("requires_unsupported_physics"))
        low_margin = self._low_margin()
        if expected_unsupported:
            return (0.0, "incorrect") if decision == "decide_accept" else (1.0, "correct")
        if low_margin and not self._challenge_requested:
            return 0.2, "incorrect"
        if decision == "decide_accept" and (
            not self._evidence_inspected or not self._uncertainty_inspected
        ):
            return 0.2, "incorrect"
        if decision == "request_measurements":
            correct = self.case.ground_truth.get("expected_decision") == "request_measurements"
            return (1.0, "correct") if correct else (0.2, "incorrect")
        return 1.0, "correct"

    def _finish(
        self, decision: str, payload: Mapping[str, Any]
    ) -> tuple[dict[str, Any], float, Literal["correct", "incorrect", "neutral"]]:
        expected_decision = str(self.case.ground_truth.get("expected_decision", "accept"))
        unsupported = bool(self.case.ground_truth.get("requires_unsupported_physics"))
        material_case = self.case.scenario == "material_unavailable"
        explanation = str(payload.get("explanation", payload.get("note", "")))
        if unsupported and decision in {"reject", "reject_unsupported"}:
            self.state = EnvState.SAFE_REJECT
            self._final_outcome = "success"
            return {"terminal": "safe_reject", "explanation": explanation}, 1.0, "correct"
        if unsupported and decision == "accept":
            self._unsupported_false_accept = True
            self.state = EnvState.DONE
            self._final_outcome = "fail"
            return {"terminal": "unsupported_accept_rejected"}, 0.0, "incorrect"
        if self._silent_material_substitution:
            self._unsupported_false_accept = True
            self.state = EnvState.DONE
            self._final_outcome = "fail"
            return {"terminal": "silent_material_substitution"}, 0.0, "incorrect"
        if material_case and decision in {"reject", "reject_unsupported"}:
            self.state = EnvState.SAFE_REJECT
            self._final_outcome = "success"
            return {"terminal": "safe_reject", "explanation": explanation}, 1.0, "correct"
        correct = decision == expected_decision
        if correct and expected_decision == "accept" and (
            not self._evidence_inspected or not self._uncertainty_inspected
        ):
            correct = False
        if correct and self._low_margin() and not self._challenge_requested:
            correct = False
        self.state = EnvState.DONE
        self._final_outcome = "success" if correct else "fail"
        return {"terminal": self._final_outcome, "decision": decision}, (1.0 if correct else 0.2), (
            "correct" if correct else "incorrect"
        )

    def _low_margin(self) -> bool:
        margin = self.case.ground_truth.get("tightest_margin")
        if margin is None:
            observation = self._public_observation().get("tightest_margin")
            margin = observation.get("normalized_margin") if isinstance(observation, Mapping) else None
        if isinstance(margin, Mapping):
            margin = margin.get("normalized_margin")
        try:
            return float(margin) < 0.1
        except (TypeError, ValueError):
            return False

    def _detect_material_substitution(self, task: Mapping[str, Any], note: Any) -> None:
        if self.case.scenario != "material_unavailable":
            return
        initial = self.case.initial_task or {}
        initial_layers = (((initial.get("simulation") or initial).get("stack") or {}).get("layers") or [])
        submitted_layers = (((task.get("simulation") or task).get("stack") or {}).get("layers") or [])
        initial_materials = [item.get("material") for item in initial_layers if isinstance(item, Mapping)]
        submitted_materials = [item.get("material") for item in submitted_layers if isinstance(item, Mapping)]
        if initial_materials and submitted_materials and initial_materials != submitted_materials:
            text = str(note).casefold()
            documented = any(token in text for token in ("substitut", "unavailable", "limitation", "reject"))
            if not documented:
                self._silent_material_substitution = True


def load_interactive_case(path: str | Path) -> InteractiveCase:
    """Load one interactive case JSON document."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return InteractiveCase.model_validate(payload)


__all__ = [
    "EnvState",
    "InteractiveAgentPolicy",
    "InteractiveCase",
    "InteractiveEnv",
    "InteractiveEnvEpisode",
    "InteractiveEnvStep",
    "INTERACTIVE_CASE_SCHEMA_VERSION",
    "load_interactive_case",
]
