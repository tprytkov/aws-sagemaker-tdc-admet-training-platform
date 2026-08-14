"""Run a synthetic, CPU-only Chemprop smoke experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from admet_platform.chemprop.smoke import run_synthetic_smoke  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", required=True, choices=("multitask_regression", "binary_classification")
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--accelerator", choices=("cpu", "cuda", "auto"), default="cpu")
    args = parser.parse_args()
    print(json.dumps(
        run_synthetic_smoke(args.task, args.output_dir, args.seed, args.accelerator), indent=2
    ))


if __name__ == "__main__":
    main()
