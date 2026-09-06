"""Backward-compatible wrapper around the stable ``veritmm run`` command.

The v0.1 script spelling remains available for existing automation, but there
is only one execution implementation: :mod:`tmm_engine.cli` delegates to
:mod:`tmm_engine.execution` and always writes the v0.2 result envelope.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tmm_engine.cli import main as veritmm_main  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compatibility wrapper for the VeriTMM agent-facing CLI."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--physics-python", default=None)
    parser.add_argument("--skip-certificate", action="store_true")
    parser.add_argument("--convergence-max-refinements", type=int, default=3)
    parser.add_argument("--convergence-pointwise-tolerance", type=float, default=5e-3)
    parser.add_argument("--convergence-integral-tolerance", type=float, default=1e-3)
    parser.add_argument("--portfolio-max-candidates", type=int, default=6)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--child-timeout-seconds", type=float, default=3600.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    forwarded = [
        "run",
        str(Path(args.input).resolve()),
        "--output-dir",
        str(Path(args.output_dir).resolve()),
        "--device",
        str(args.device),
        "--convergence-max-refinements",
        str(args.convergence_max_refinements),
        "--convergence-pointwise-tolerance",
        str(args.convergence_pointwise_tolerance),
        "--convergence-integral-tolerance",
        str(args.convergence_integral_tolerance),
        "--portfolio-max-candidates",
        str(args.portfolio_max_candidates),
        "--child-timeout-seconds",
        str(args.child_timeout_seconds),
        "--json",
    ]
    if args.physics_python:
        forwarded.extend(["--physics-python", str(args.physics_python)])
    if args.skip_certificate:
        forwarded.append("--skip-certificate")
    if args.no_plot:
        forwarded.append("--no-plot")
    return int(veritmm_main(forwarded))


if __name__ == "__main__":
    raise SystemExit(main())
