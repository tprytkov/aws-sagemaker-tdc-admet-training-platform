"""Fit and freeze the TRAIN-only GMC-MPNN six-feature GGL scaler."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from admet_platform.gmc_mpnn.scaling import fit_training_ggl_scaler  # noqa: E402


DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "gpu" / "pilot" / "gmc_mpnn_ggl_scaler_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate frozen TRAIN-only raw GGL artifacts, fit sklearn StandardScaler over "
            "pooled training atoms, and write portable scaler statistics."
        )
    )
    parser.add_argument(
        "--training-preprocessing-dir",
        type=Path,
        required=True,
        help="Completed TRAIN-only raw-GGL preprocessing directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="New output directory for scaler.json, scaler.npz, and fit_summary.json.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = fit_training_ggl_scaler(
        args.training_preprocessing_dir,
        args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
