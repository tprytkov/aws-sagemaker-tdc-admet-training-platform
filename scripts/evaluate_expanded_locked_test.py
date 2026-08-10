"""Dry-run or execute the frozen ten-endpoint locked-test evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from admet_platform.training.expanded_locked_test_evaluation import (  # noqa: E402
    run_expanded_locked_test_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate or run one frozen ten-endpoint checkpoint on locked test data."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/final_test_evaluation_expanded.yaml",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate frozen metadata without opening or hashing any test CSV.",
    )
    args = parser.parse_args()
    result = run_expanded_locked_test_evaluation(
        evaluation_config=args.config,
        output_dir=args.output_dir,
        device=args.device,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
