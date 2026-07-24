"""Combine five completed single-task validation summaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from admet_platform.models.multitask_regression_chemberta import (  # noqa: E402
    DEFAULT_REGRESSION_ENDPOINTS,
)
from admet_platform.training.single_task_regression_run import (  # noqa: E402
    build_single_task_regression_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a comparison-ready validation summary for five baselines."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Endpoint and run directory as endpoint=path; repeat five times.",
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    runs = {}
    for value in args.run:
        endpoint, separator, path = value.partition("=")
        if not separator or endpoint not in DEFAULT_REGRESSION_ENDPOINTS or not path:
            parser.error(f"Invalid --run value: {value}")
        if endpoint in runs:
            parser.error(f"Duplicate --run endpoint: {endpoint}")
        runs[endpoint] = path
    frame = build_single_task_regression_comparison(
        runs, output_csv=args.output_csv, output_json=args.output_json
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
