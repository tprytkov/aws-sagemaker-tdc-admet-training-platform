"""Evaluate fixed selected ChemBERTa checkpoints on coordinated test splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from admet_platform.training.multitask_final_evaluation import (  # noqa: E402
    run_final_test_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate preselected ChemBERTa checkpoints on coordinated test data only."
    )
    parser.add_argument(
        "--config", default="configs/final_test_evaluation.yaml"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    result = run_final_test_evaluation(
        evaluation_config=args.config,
        output_dir=args.output_dir,
        device=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
