"""Build leakage-safe coordinated splits for continuous ADMET endpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from admet_platform.data.coordinated_multitask_regression import (  # noqa: E402
    build_coordinated_multitask_regression_splits,
    load_regression_split_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build globally scaffold-grouped multi-task regression splits."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-root")
    parser.add_argument("--output-root")
    parser.add_argument("--seed", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_regression_split_config(args.config)
        result = build_coordinated_multitask_regression_splits(
            config,
            source_root=args.source_root,
            output_root=args.output_root,
            seed=args.seed,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Coordinated regression split build failed: {exc}", file=sys.stderr)
        return 1
    print(f"Coordinated regression output: {result.output_root}")
    print(json.dumps(result.manifest["leakage_audit"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
