from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INVALID = ROOT / "tests" / "fixtures" / "agent_invalid"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "tmm_engine.cli", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_describe_schema_and_preflight_stdout_are_single_json_objects() -> None:
    commands = [
        ("describe", "--json"),
        ("schema", "simulation"),
        ("preflight", str(ROOT / "examples" / "tmm_tasks" / "periodic_dbr_simulation.json"), "--json"),
    ]
    for command in commands:
        result = _run(*command)
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert isinstance(payload, dict)
        assert result.stdout.count("\n") <= 1


def test_successful_run_stdout_is_one_parseable_json_object(tmp_path: Path) -> None:
    output = tmp_path / "successful_run"
    result = _run(
        "run",
        str(ROOT / "examples" / "tmm_tasks" / "periodic_dbr_simulation.json"),
        "--output-dir",
        str(output),
        "--json",
        "--no-plot",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert result.stdout.count("\n") <= 1
    on_disk = json.loads((output / "RUN_RESULT.json").read_text(encoding="utf-8"))
    assert on_disk["summary"]["response"] == payload["summary"]["response"]
    assert (output / "RESPONSE_CONTEXT.json").is_file()


def test_invalid_run_has_parseable_stdout_no_traceback_and_result_file(tmp_path: Path) -> None:
    output = tmp_path / "invalid_run"
    result = _run(
        "run",
        str(INVALID / "negative_thickness.json"),
        "--output-dir",
        str(output),
        "--json",
    )
    assert result.returncode != 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert "Traceback" not in result.stdout
    assert "Traceback" not in result.stderr
    on_disk = json.loads((output / "RUN_RESULT.json").read_text(encoding="utf-8"))
    assert on_disk["run_id"] == payload["run_id"]


def test_failed_preflight_and_argument_parser_emit_json_without_traceback() -> None:
    failed_preflight = _run(
        "preflight",
        str(INVALID / "unsupported_geometry.json"),
        "--json",
    )
    assert failed_preflight.returncode != 0
    assert json.loads(failed_preflight.stdout)["ok"] is False
    assert "Traceback" not in failed_preflight.stderr

    parser_failure = _run("schema", "not-a-schema")
    assert parser_failure.returncode != 0
    parser_payload = json.loads(parser_failure.stdout)
    assert parser_payload["ok"] is False
    assert parser_payload["operation"] == "argument_parsing"
    assert "Traceback" not in parser_failure.stderr
