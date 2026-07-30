"""Fit validation-only endpoint calibrators for a selected multi-task checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from admet_platform.training.multitask_calibration import (  # noqa: E402
    run_multitask_calibration,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit validation-only Platt calibrators without training or test access."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda"))
    parser.add_argument("--split", default="validation")
    args = parser.parse_args()
    result = run_multitask_calibration(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        prepared_root=args.prepared_root,
        output_dir=args.output_dir,
        device=args.device,
        source_split=args.split,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
