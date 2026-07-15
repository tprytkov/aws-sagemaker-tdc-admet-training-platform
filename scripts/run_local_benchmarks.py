"""Run local classical baseline benchmarks for prepared or downloadable TDC datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from admet_platform.benchmarks import run_local_benchmarks  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local ADMET baseline benchmarks.")
    parser.add_argument("--config", action="append", required=True, help="Endpoint config path. Repeatable.")
    parser.add_argument(
        "--feature-type",
        action="append",
        choices=["descriptors", "morgan"],
        help="Feature type to run. Repeatable. Defaults to both.",
    )
    parser.add_argument("--prepared-root", default="outputs/local/full_datasets")
    parser.add_argument("--benchmark-root", default="outputs/local/benchmarks")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--morgan-radius", type=int, default=2)
    parser.add_argument("--morgan-bits", type=int, default=2048)
    parser.add_argument("--max-rows", type=int, help="Development-only per-split row limit.")
    args = parser.parse_args()

    result = run_local_benchmarks(
        config_paths=args.config,
        prepared_root=args.prepared_root,
        benchmark_root=args.benchmark_root,
        feature_types=args.feature_type,
        force_rerun=args.force_rerun,
        random_seed=args.random_seed,
        morgan_radius=args.morgan_radius,
        morgan_bits=args.morgan_bits,
        max_rows=args.max_rows,
    )
    print(f"Wrote benchmark artifacts: {result['benchmark_root']}")
    print(f"Successful runs: {sum(1 for row in result['benchmark_rows'] if row['run_status'] == 'success')}")
    print(f"Failures: {len(result['failures'])}")
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
