from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tmm_engine import MaterialRegistry, TMMWorkbench  # noqa: E402
from tmm_engine.validation_cases import validate_pmc9147317  # noqa: E402

PAPER_FIGURE_URL = (
    "https://cdn.ncbi.nlm.nih.gov/pmc/blobs/ffd5/9147317/cee236241fa1/"
    "sensors-22-03627-g003.jpg"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "tmm_validation" / "PMC9147317"),
    )
    parser.add_argument("--skip-paper-figure", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    report = validate_pmc9147317(
        TMMWorkbench(MaterialRegistry()), output_dir=output_dir
    )
    if not args.skip_paper_figure:
        figure_path = output_dir / "paper_figure_3.jpg"
        if not figure_path.exists():
            try:
                urllib.request.urlretrieve(PAPER_FIGURE_URL, figure_path)
            except Exception as exc:
                (output_dir / "PAPER_FIGURE_DOWNLOAD_ERROR.txt").write_text(
                    "%s: %s\n" % (type(exc).__name__, exc), encoding="utf-8"
                )
    print(json.dumps(report))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
