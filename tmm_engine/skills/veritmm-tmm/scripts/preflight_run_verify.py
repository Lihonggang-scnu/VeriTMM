"""Minimal VeriTMM skill workflow: preflight -> run -> verify-run (CLI fallback).

The preferred surface is the veritmm MCP tools (preflight / run / verify_run).
This script is the CLI fallback for hosts without the MCP connection and
chains the same three steps against one task file.

Usage:
    python preflight_run_verify.py example_task.json OUTPUT_DIR
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from shutil import which


def _cli_base() -> list[str]:
    if which("veritmm"):
        return ["veritmm"]
    return [sys.executable, "-m", "tmm_engine.cli"]


def _run_json(args: list[str]) -> dict:
    completed = subprocess.run(
        [*_cli_base(), *args, "--json"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=600,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    task_path, output_dir = Path(argv[1]), Path(argv[2])

    preflight = _run_json(["preflight", str(task_path)])
    if not preflight.get("ok"):
        print("PREFLIGHT REJECTED:")
        print(json.dumps(preflight.get("failures", preflight), indent=2))
        return 2

    run = _run_json(
        ["run", str(task_path), "--output-dir", str(output_dir), "--no-cache"]
    )
    print("run:", run.get("status"), "| certificate:", run.get("certificate_id"))

    verify = _run_json(["verify-run", str(output_dir)])
    print(
        "verify-run:",
        verify["integrity_status"],
        verify["certification_status"],
        verify["replay_status"],
        verify["authenticity_status"],
        "->",
        verify["overall_status"],
    )
    return 0 if verify["overall_status"] == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
