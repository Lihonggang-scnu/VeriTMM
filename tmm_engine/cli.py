"""Stable command-line protocol for human and autonomous callers."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Sequence

from .archive.schema_registry import ARCHIVE_SCHEMA_VERSION
from .capabilities import failure_from_exception
from .execution import ExecutionSettings
from .experiment_store import ExperimentStore, compare_runs, default_store_root
from .managed_execution import execute_managed_task
from .preflight import preflight_path
from .protocol.responses import (
    DEFAULT_RESPONSE_DETAIL,
    is_projected_response,
    project_response,
    validate_projected_response,
)
from .run_artifacts import (
    ResponseDetailUnavailableError,
    build_result_summary,
    file_sha256,
    prepare_output_directory,
    stable_payload_sha256,
    write_json,
    write_run_result,
)
from .schemas import dataclass_to_dict
from .task_io import load_task

ROOT = Path(__file__).resolve().parents[1]


def _emit(
    payload: Any,
    *,
    detail: str = DEFAULT_RESPONSE_DETAIL,
    project: bool = True,
) -> None:
    """Emit exactly one JSON object to stdout.

    All machine-facing commands use the shared response projection.  Schema
    documents opt out because their arrays are part of the contract itself.
    """

    if project and is_projected_response(payload):
        validate_projected_response(payload, detail=detail)
        rendered = payload
    elif project:
        rendered = project_response(payload, detail=detail)
    else:
        rendered = payload

    print(
        json.dumps(
            rendered,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=str,
        )
    )


class _MachineArgumentParser(argparse.ArgumentParser):
    """Return parser errors as one JSON object instead of mixed help text."""

    def error(self, message: str) -> None:
        failure = failure_from_exception(ValueError(message))
        _emit(
            {
                "ok": False,
                "operation": "argument_parsing",
                "status": "failed",
                "error": failure.to_dict(),
            }
        )
        raise SystemExit(2)


def _build_parser() -> argparse.ArgumentParser:
    parser = _MachineArgumentParser(
        prog="veritmm",
        description="Agent-ready, verifier-first transfer-matrix execution tool.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_detail(
        command: argparse.ArgumentParser,
        *,
        default: str = DEFAULT_RESPONSE_DETAIL,
    ) -> None:
        command.add_argument(
            "--detail",
            choices=("compact", "standard", "full"),
            default=default,
            help=f"Response profile for machine-facing output (default: {default})",
        )

    describe = subparsers.add_parser("describe", help="Describe supported physics and protocol")
    describe.add_argument("--json", action="store_true", help="Emit compact machine JSON")
    add_detail(describe)

    schema = subparsers.add_parser("schema", help="Export a public JSON Schema")
    schema.add_argument(
        "kind",
        choices=(
            "simulation",
            "optimization",
            "sweep",
            "sensitivity",
            "tolerance",
            "preflight",
            "failure",
            "run_result",
            "response",
        ),
    )
    add_detail(schema)

    examples = subparsers.add_parser("examples", help="List bundled task examples")
    add_detail(examples)

    preflight = subparsers.add_parser("preflight", help="Validate without running a spectrum")
    preflight.add_argument("task")
    preflight.add_argument("--json", action="store_true", help="Emit compact machine JSON")
    add_detail(preflight)

    run = subparsers.add_parser("run", help="Execute and certify one task")
    run.add_argument("task")
    run.add_argument("--output-dir", required=True)
    run.add_argument("--json", action="store_true", help="Emit compact machine JSON")
    add_detail(run)
    run.add_argument("--device", default="cpu")
    run.add_argument("--physics-python", default=None)
    run.add_argument("--skip-certificate", action="store_true")
    run.add_argument("--convergence-max-refinements", type=int, default=6)
    run.add_argument("--convergence-pointwise-tolerance", type=float, default=5e-3)
    run.add_argument("--convergence-integral-tolerance", type=float, default=1e-3)
    run.add_argument("--no-plot", action="store_true")
    run.add_argument("--child-timeout-seconds", type=float, default=3600.0)
    run.add_argument("--portfolio-max-candidates", type=int, default=6)
    run.add_argument("--store-dir", default=None)
    run.add_argument("--no-store", action="store_true")
    run.add_argument("--experiment-id", default=None)
    run.add_argument("--parent-run-id", default=None)
    run.add_argument("--tag", action="append", default=[])
    run.add_argument("--hypothesis", default=None)
    run.add_argument("--change-reason", default=None)
    run.add_argument("--user-metadata-json", default="{}")
    run.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--resume", action="store_true")

    challenge = subparsers.add_parser(
        "challenge", help="Run deterministic active challenge verification"
    )
    challenge.add_argument("--seed", type=int, default=42)
    challenge.add_argument("--budget", type=int, default=100)
    challenge.add_argument(
        "--objective",
        choices=(
            "min_margin",
            "max_solver_disagreement",
            "max_convergence_residual",
            "metamorphic_violation",
        ),
        default="min_margin",
    )
    challenge.add_argument("--output", required=True)
    challenge.add_argument("--json", action="store_true")
    add_detail(challenge)

    fit = subparsers.add_parser(
        "fit", help="Fit experimental measurements and report identifiability"
    )
    fit.add_argument("fit_task_json")
    fit.add_argument("--output", required=True)
    fit.add_argument("--json", action="store_true")
    add_detail(fit)

    plan_measurement = subparsers.add_parser(
        "plan-measurement",
        help="Choose next measurements from local Fisher information at a fitted point",
    )
    plan_measurement.add_argument("fit_result_json")
    plan_measurement.add_argument("--candidates", required=True)
    plan_measurement.add_argument(
        "--criterion", choices=("d_optimal", "a_optimal"), default="d_optimal"
    )
    plan_measurement.add_argument("--n", dest="n_select", type=int, default=1)
    plan_measurement.add_argument("--output", required=True)
    plan_measurement.add_argument("--json", action="store_true")
    add_detail(plan_measurement)

    history = subparsers.add_parser("history", help="List persisted experiment runs")
    history.add_argument("--store-dir", default=None)
    history.add_argument("--experiment", default=None)
    history.add_argument("--limit", type=int, default=100)
    history.add_argument("--json", action="store_true")
    add_detail(history)

    for name, help_text in (
        ("inspect", "Inspect one persisted run"),
        ("lineage", "Show ancestors and children for a run"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("run_id")
        command.add_argument("--store-dir", default=None)
        command.add_argument("--json", action="store_true")
        add_detail(command)

    compare = subparsers.add_parser("compare", help="Compare two persisted runs")
    compare.add_argument("run_a")
    compare.add_argument("run_b")
    compare.add_argument("--store-dir", default=None)
    compare.add_argument("--json", action="store_true")
    add_detail(compare)

    benchmark = subparsers.add_parser(
        "benchmark", help="Run the deterministic offline AgentBench suite"
    )
    benchmark.add_argument("--offline", action="store_true", required=True)
    benchmark.add_argument("--cases-dir", default=None)
    benchmark.add_argument("--output", default="BENCHMARK_RESULT.json")
    benchmark.add_argument("--work-dir", default=None)
    benchmark.add_argument("--json", action="store_true")
    add_detail(benchmark)

    agent_benchmark = subparsers.add_parser(
        "agent-benchmark", help="Score framework-neutral agent A/B trajectories"
    )
    agent_benchmark.add_argument("--trajectories", required=True)
    agent_benchmark.add_argument("--cases-dir", default=None)
    agent_benchmark.add_argument("--output", default=None)
    agent_benchmark.add_argument("--json", action="store_true")
    add_detail(agent_benchmark)
    return parser


def _store_from_args(args: argparse.Namespace) -> ExperimentStore | None:
    if bool(getattr(args, "no_store", False)):
        return None
    root = getattr(args, "store_dir", None)
    return ExperimentStore(root or default_store_root())


def _execution_settings_from_args(args: argparse.Namespace) -> ExecutionSettings:
    return ExecutionSettings(
        device=args.device,
        skip_certificate=args.skip_certificate,
        convergence_max_refinements=args.convergence_max_refinements,
        convergence_pointwise_tolerance=args.convergence_pointwise_tolerance,
        convergence_integral_tolerance=args.convergence_integral_tolerance,
        write_plot=not args.no_plot,
        portfolio_max_candidates=args.portfolio_max_candidates,
    )


def _user_metadata(args: argparse.Namespace) -> dict[str, Any]:
    payload = json.loads(str(getattr(args, "user_metadata_json", "{}")))
    if not isinstance(payload, dict):
        raise ValueError("--user-metadata-json must encode a JSON object")
    return payload


def _record_failed_load(
    args: argparse.Namespace, payload: dict[str, Any], store: ExperimentStore | None
) -> None:
    if store is None:
        return
    archived = store.archive_artifacts(args.output_dir, str(payload["run_id"]))
    store.record_envelope(
        payload,
        artifact_root=archived,
        experiment_id=args.experiment_id,
        parent_run_id=args.parent_run_id,
        tags=args.tag,
        hypothesis=args.hypothesis,
        change_reason=args.change_reason,
        user_metadata=_user_metadata(args),
    )


def _describe() -> dict[str, Any]:
    from .protocol.capabilities import describe_capabilities

    manifest = describe_capabilities()
    return manifest.model_dump(mode="json") if hasattr(manifest, "model_dump") else dict(manifest)


def _schema(kind: str) -> dict[str, Any]:
    from .protocol.schema_export import export_schema

    return export_schema(kind)


def _examples() -> dict[str, Any]:
    packaged = Path(__file__).resolve().parent / "examples"
    source_checkout = ROOT / "examples" / "tmm_tasks"
    root = packaged if packaged.is_dir() else source_checkout
    examples = []
    for path in sorted(root.glob("*.json")):
        try:
            mode, _ = load_task(path)
            examples.append(
                {
                    "name": path.name,
                    "mode": mode,
                    "path": path.as_posix(),
                }
            )
        except Exception as exc:
            examples.append(
                {
                    "name": path.name,
                    "mode": "invalid",
                    "path": path.as_posix(),
                    "error": str(exc),
                }
            )
    return {"ok": all(item["mode"] != "invalid" for item in examples), "examples": examples}


def _failed_load_result(
    task_path: str | Path,
    output_dir: str | Path,
    *,
    detail: str = DEFAULT_RESPONSE_DETAIL,
) -> dict[str, Any]:
    report = preflight_path(task_path)
    root = prepare_output_directory(output_dir)
    run_id = f"run_{uuid.uuid4().hex}"
    write_json(root / "PREFLIGHT_REPORT.json", report)
    summary = build_result_summary(
        mode="unknown",
        forward=None,
        certificate=None,
        warnings=report.get("warnings", []),
        run_id=run_id,
        task_sha256=None,
        run_status="preflight_rejected",
    )
    write_json(root / "RESULT_SUMMARY.json", summary)
    input_sha256 = None
    try:
        input_sha256 = file_sha256(task_path)
    except Exception:
        pass
    manifest = {
        "mode": "unknown",
        "status": "preflight_rejected",
        "input": str(Path(task_path).resolve()),
        "task_sha256": None,
        "task_hash_scope": "normalized_operation_wrapper",
        "archive_schema_version": ARCHIVE_SCHEMA_VERSION,
        "input_sha256": input_sha256,
        "failures": report.get("failures", []),
    }
    write_json(root / "RUN_MANIFEST.json", manifest)
    return write_run_result(
        root,
        operation="run",
        task_sha256=None,
        status="preflight_rejected",
        ok=False,
        summary=summary,
        warnings=report.get("warnings", []),
        failures=report.get("failures", []),
        run_id=run_id,
        input_sha256=input_sha256,
        detail=detail,
    )


def _child_runtime_failure(
    args: argparse.Namespace,
    mode: str,
    task: Any,
    exc: Exception,
    *,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = prepare_output_directory(args.output_dir)
    run_id = f"run_{uuid.uuid4().hex}"
    normalized = {
        "mode": mode,
        "simulation" if mode == "simulate" else "optimization": dataclass_to_dict(task),
    }
    task_sha256 = stable_payload_sha256(normalized)
    failure = failure_from_exception(exc)
    failure_payload = failure.to_dict()
    if context:
        failure_payload["context"] = {**failure_payload.get("context", {}), **context}
    summary = build_result_summary(
        mode=mode,
        forward=None,
        certificate=None,
        run_id=run_id,
        task_sha256=task_sha256,
        run_status="child_runtime_protocol_failure",
    )
    write_json(output / "RESULT_SUMMARY.json", summary)
    write_json(
        output / "RUN_MANIFEST.json",
        {
            "mode": mode,
            "status": "child_runtime_protocol_failure",
            "run_id": run_id,
            "task_sha256": task_sha256,
            "archive_schema_version": ARCHIVE_SCHEMA_VERSION,
            "failures": [failure_payload],
        },
    )
    return write_run_result(
        output,
        operation=mode,
        task_sha256=task_sha256,
        status="child_runtime_protocol_failure",
        ok=False,
        summary=summary,
        failures=[failure_payload],
        run_id=run_id,
        input_sha256=file_sha256(args.task) if Path(args.task).is_file() else None,
        detail=str(getattr(args, "detail", DEFAULT_RESPONSE_DETAIL)),
    )


def _optimization_child_command(args: argparse.Namespace) -> list[str] | None:
    if os.environ.get("VERITMM_PHYSICS_CHILD") == "1":
        return None
    try:
        mode, _ = load_task(args.task)
    except Exception:
        return None
    if mode != "optimize" or importlib.util.find_spec("torch") is not None:
        return None
    from .physics_runtime import discover_physics_python

    executable = discover_physics_python(args.physics_python)
    command = [
        str(executable),
        "-m",
        "tmm_engine.cli",
        "run",
        str(Path(args.task).resolve()),
        "--output-dir",
        str(Path(args.output_dir).resolve()),
        "--device",
        args.device,
        "--convergence-max-refinements",
        str(args.convergence_max_refinements),
        "--convergence-pointwise-tolerance",
        str(args.convergence_pointwise_tolerance),
        "--convergence-integral-tolerance",
        str(args.convergence_integral_tolerance),
        "--json",
        "--detail",
        str(args.detail),
        "--portfolio-max-candidates",
        str(args.portfolio_max_candidates),
    ]
    if args.skip_certificate:
        command.append("--skip-certificate")
    if args.no_plot:
        command.append("--no-plot")
    if args.no_store:
        command.append("--no-store")
    else:
        command.extend(["--store-dir", str(args.store_dir or default_store_root())])
        if args.experiment_id:
            command.extend(["--experiment-id", args.experiment_id])
        if args.parent_run_id:
            command.extend(["--parent-run-id", args.parent_run_id])
        for tag in args.tag:
            command.extend(["--tag", tag])
        if args.hypothesis:
            command.extend(["--hypothesis", args.hypothesis])
        if args.change_reason:
            command.extend(["--change-reason", args.change_reason])
        command.extend(["--user-metadata-json", json.dumps(_user_metadata(args))])
        command.append("--cache" if args.cache else "--no-cache")
        if args.resume:
            command.append("--resume")
    return command


def _run(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    store = _store_from_args(args)
    try:
        mode, task = load_task(args.task)
    except Exception:
        payload = _failed_load_result(args.task, args.output_dir, detail=args.detail)
        _record_failed_load(args, payload, store)
        return 2, payload

    try:
        child_command = _optimization_child_command(args)
    except Exception as exc:
        return 3, _child_runtime_failure(args, mode, task, exc)
    if child_command is not None:
        environment = dict(os.environ)
        environment["VERITMM_PHYSICS_CHILD"] = "1"
        environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
        try:
            completed = subprocess.run(
                child_command,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=float(args.child_timeout_seconds),
                check=False,
            )
        except Exception as exc:
            return 3, _child_runtime_failure(
                args,
                mode,
                task,
                exc,
                context={"child_command": child_command},
            )
        try:
            payload = json.loads(completed.stdout)
            from .protocol import RunResultEnvelope

            payload = RunResultEnvelope.model_validate(payload).model_dump(mode="json")
        except Exception as exc:
            payload = _child_runtime_failure(
                args,
                mode,
                task,
                exc,
                context={
                    "child_returncode": completed.returncode,
                    "child_stderr_tail": completed.stderr[-4000:],
                },
            )
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        return completed.returncode, payload

    payload = execute_managed_task(
        mode,
        task,
        args.output_dir,
        input_path=args.task,
        execution_settings=_execution_settings_from_args(args),
        store=store,
        experiment_id=args.experiment_id,
        parent_run_id=args.parent_run_id,
        tags=args.tag,
        hypothesis=args.hypothesis,
        change_reason=args.change_reason,
        user_metadata=_user_metadata(args),
        cache=bool(args.cache),
        resume=bool(args.resume),
        detail=args.detail,
    )
    return (0 if payload["ok"] else 3), payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "describe":
            _emit(_describe(), detail=args.detail)
            return 0
        if args.command == "schema":
            _emit(_schema(args.kind), project=False)
            return 0
        if args.command == "examples":
            payload = _examples()
            _emit(payload, detail=args.detail)
            return 0 if payload["ok"] else 2
        if args.command == "preflight":
            payload = preflight_path(args.task)
            _emit(payload, detail=args.detail)
            return 0 if payload["ok"] else 2
        if args.command == "run":
            code, payload = _run(args)
            _emit(payload, detail=args.detail)
            return code
        if args.command == "challenge":
            from .verifier.challenge import (
                ChallengeObjective,
                ChallengeSpec,
                run_challenge_search,
            )

            spec = ChallengeSpec(
                seed=args.seed,
                budget=args.budget,
                objective=ChallengeObjective(args.objective),
            )
            result = run_challenge_search(spec)
            if result.minimized_candidate is not None:
                output_path = Path(args.output)
                canonical_path = output_path.with_name(
                    f"{output_path.stem}_canonical_task.json"
                )
                write_json(canonical_path, result.minimized_candidate)
                result.canonical_task_path = str(canonical_path)
            payload = result.model_dump(mode="json")
            write_json(args.output, payload)
            _emit(payload, detail=args.detail, project=False)
            return 0
        if args.command == "fit":
            from .fitting.fit_task import FitTask
            from .fitting.optimizer import fit_task as execute_fit

            task_data = json.loads(Path(args.fit_task_json).read_text(encoding="utf-8"))
            task = FitTask.model_validate(task_data)
            result = execute_fit(task)
            payload = result.model_dump(mode="json")
            write_json(args.output, payload)
            _emit(payload, detail=args.detail, project=False)
            return 0 if result.converged else 3
        if args.command == "plan-measurement":
            from .fitting.fit_task import FitResult
            from .fitting.measurement_plan import (
                MeasurementAction,
                MeasurementPlanTask,
                build_measurement_plan,
            )

            fit_payload = json.loads(
                Path(args.fit_result_json).read_text(encoding="utf-8")
            )
            fit_result = FitResult.model_validate(fit_payload)
            candidate_payload = json.loads(
                Path(args.candidates).read_text(encoding="utf-8")
            )
            if isinstance(candidate_payload, dict):
                candidate_payload = candidate_payload.get(
                    "candidates", candidate_payload.get("candidate_pool")
                )
            if not isinstance(candidate_payload, list):
                raise ValueError(
                    "candidate pool JSON must be a list or an object with candidates"
                )
            candidates = [
                MeasurementAction.model_validate(item) for item in candidate_payload
            ]
            plan_task = MeasurementPlanTask(
                fit_result=fit_result,
                candidates=candidates,
                criterion=args.criterion,
                n_select=args.n_select,
            )
            result = build_measurement_plan(plan_task)
            payload = result.model_dump(mode="json")
            write_json(args.output, payload)
            _emit(payload, detail=args.detail, project=False)
            return 0
        if args.command == "history":
            store = ExperimentStore(args.store_dir or default_store_root())
            runs = store.list_runs(experiment_id=args.experiment, limit=args.limit)
            _emit(
                {
                    "schema_version": "veritmm-history-v1",
                    "ok": True,
                    "store_root": str(store.root),
                    "runs": [record.to_dict() for record in runs],
                },
                detail=args.detail,
            )
            return 0
        if args.command == "inspect":
            store = ExperimentStore(args.store_dir or default_store_root())
            _emit(store.inspect(args.run_id, detail=args.detail), detail=args.detail)
            return 0
        if args.command == "lineage":
            store = ExperimentStore(args.store_dir or default_store_root())
            _emit(
                {
                    "ok": True,
                    **store.get_lineage(args.run_id, detail=args.detail),
                },
                detail=args.detail,
            )
            return 0
        if args.command == "compare":
            store = ExperimentStore(args.store_dir or default_store_root())
            _emit({"ok": True, **compare_runs(store, args.run_a, args.run_b)}, detail=args.detail)
            return 0
        if args.command == "benchmark":
            from .agent_bench import (
                default_benchmark_cases_dir,
                load_benchmark_cases,
                run_offline_benchmark,
            )

            case_root = args.cases_dir or default_benchmark_cases_dir()
            payload = run_offline_benchmark(
                load_benchmark_cases(case_root),
                output_path=args.output,
                work_dir=args.work_dir,
                detail=args.detail,
            )
            _emit(payload, detail=args.detail)
            return 0 if payload["release_gate_passed"] else 3
        if args.command == "agent-benchmark":
            from .agent_harness import load_trajectories, score_trajectories

            cases = None
            if args.cases_dir:
                from .agent_bench import load_benchmark_cases

                cases = {
                    item.case_id: item for item in load_benchmark_cases(args.cases_dir)
                }
            payload = score_trajectories(
                load_trajectories(args.trajectories),
                cases=cases,
                output_path=args.output,
                detail=args.detail,
            )
            _emit(payload, detail=args.detail)
            return 0
    except ResponseDetailUnavailableError as exc:
        _emit(
            exc.to_response(operation=str(args.command)),
            detail=str(getattr(args, "detail", DEFAULT_RESPONSE_DETAIL)),
        )
        return 2
    except Exception as exc:
        failure = failure_from_exception(exc)
        _emit(
            {
                "ok": False,
                "operation": str(args.command),
                "status": "failed",
                "error": failure.to_dict(),
            },
            detail=str(getattr(args, "detail", DEFAULT_RESPONSE_DETAIL)),
        )
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
