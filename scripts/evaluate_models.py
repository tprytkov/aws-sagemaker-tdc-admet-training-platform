"""CLI for local model-run evaluation, comparison, and model-card generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from admet_platform.evaluation import ComparisonOptions, evaluate_model_runs  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate and compare completed local ADMET model runs.")
    parser.add_argument("--run-dir", action="append", default=[], help="Model-run directory. May be repeated.")
    parser.add_argument("--discover-parent", help="Scan a parent directory for model-run directories.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--near-tie-tolerance", type=float, default=0.01)
    parser.add_argument("--primary-metric")
    parser.add_argument("--include-development-runs", action="store_true")
    parser.add_argument("--endpoint-id")
    parser.add_argument("--registry-schema-version", default="1.0.0")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = evaluate_model_runs(
            args.run_dir,
            args.output_dir,
            discovery_parent=args.discover_parent,
            options=ComparisonOptions(
                near_tie_tolerance=args.near_tie_tolerance,
                primary_metric_override=args.primary_metric,
                include_development_runs=args.include_development_runs,
                registry_schema_version=args.registry_schema_version,
            ),
            explicit_endpoint_id=args.endpoint_id,
        )
    except Exception as exc:  # noqa: BLE001 - CLI should return nonzero with useful error.
        parser.exit(1, f"Model evaluation failed: {exc}\n")
    print(f"Evaluation status: {result.recommendation_status}")
    print(f"Recommended run: {result.recommended_run_id or 'none'}")
    print(f"Artifacts: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
