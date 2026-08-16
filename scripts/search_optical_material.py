"""Search and sample the bundled refractiveindex.info material mirror."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tmm_engine.material_registry import MaterialRegistry  # noqa: E402


def _candidate_dict(candidate: object) -> dict:
    return {
        "provider": candidate.provider,
        "dataset_id": candidate.dataset_id,
        "shelf": candidate.shelf,
        "book": candidate.book,
        "page": candidate.page,
        "filepath": candidate.filepath,
        "range_um": candidate.range_um,
        "has_n": candidate.has_n,
        "has_k": candidate.has_k,
        "points": candidate.points,
        "exact_book": candidate.exact_book,
        "full_coverage": candidate.full_coverage,
        "rank_key": list(candidate.rank_key),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default=None)
    parser.add_argument("--provider", choices=("auto", "local", "rii"), default="rii")
    parser.add_argument("--start-nm", type=float)
    parser.add_argument("--stop-nm", type=float)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--dataset-id")
    parser.add_argument("--sample-points", type=int, default=101)
    parser.add_argument("--output", default=None)
    parser.add_argument("--export-csv", default=None)
    parser.add_argument("--allow-extrapolation", action="store_true")
    args = parser.parse_args()
    if (args.start_nm is None) != (args.stop_nm is None):
        parser.error("--start-nm and --stop-nm must be supplied together")
    range_um = None
    if args.start_nm is not None:
        range_um = (float(args.start_nm) * 1e-3, float(args.stop_nm) * 1e-3)
    provider = None if args.provider == "auto" else args.provider
    registry = MaterialRegistry()
    candidates = registry.search(
        args.query,
        wavelength_range=range_um,
        provider=provider,
        dataset_id=args.dataset_id,
    )
    payload = {
        "query": args.query,
        "requested_range_nm": None
        if range_um is None
        else [float(args.start_nm), float(args.stop_nm)],
        "catalog_status": registry.catalog_status(),
        "candidate_count": len(candidates),
        "candidates": [_candidate_dict(item) for item in candidates[: max(0, args.limit)]],
    }
    if args.dataset_id is not None:
        if range_um is None:
            raise ValueError("sampling a dataset requires --start-nm and --stop-nm")
        wavelengths_um = __import__("numpy").linspace(
            range_um[0], range_um[1], max(2, int(args.sample_points))
        )
        sampled = registry.sample(
            args.query,
            wavelengths_um,
            provider=provider,
            dataset_id=args.dataset_id,
            allow_extrapolation=args.allow_extrapolation,
        )
        payload["selected_dataset"] = sampled.provenance
        payload["sample_warnings"] = sampled.warnings
        if args.export_csv:
            export = Path(args.export_csv).resolve()
            export.parent.mkdir(parents=True, exist_ok=True)
            with export.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["wavelength_um", "n", "k"])
                writer.writerows(zip(sampled.wavelengths_um, sampled.n, sampled.k))
            payload["export_csv"] = str(export)
    encoded = json.dumps(payload, indent=2)
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded)
    return 0 if candidates else 2


if __name__ == "__main__":
    raise SystemExit(main())
