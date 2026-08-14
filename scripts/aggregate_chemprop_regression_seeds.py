"""Aggregate five-head regression predictions across completed seed runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from admet_platform.chemprop.config import load_chemprop_config  # noqa: E402
from admet_platform.chemprop.uncertainty import aggregate_endpoint_seed_predictions  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate per-endpoint seed variability; this does not access locked tests."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = load_chemprop_config(args.config)
    if not config.tasks:
        raise ValueError("This command accepts the multitask regression configuration only.")
    if len(args.run_dir) < 2:
        raise ValueError("At least two independently seeded run directories are required.")
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for endpoint in config.tasks:
        frames = {
            index: pd.read_csv(Path(run_dir) / "validation_predictions" / f"{endpoint}.csv")
            for index, run_dir in enumerate(args.run_dir)
        }
        aggregate_endpoint_seed_predictions(frames).to_csv(
            destination / f"{endpoint}.csv", index=False
        )


if __name__ == "__main__":
    main()
