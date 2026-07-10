"""Command-line interface for local baseline ADMET model training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from admet_platform.training.baseline import train_baseline_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a local classical ADMET baseline model.")
    parser.add_argument("--input-csv", required=True, help="Path to the prepared input CSV.")
    parser.add_argument("--config", required=True, help="Path to the endpoint YAML config.")
    parser.add_argument("--model-output", required=True, help="Path for the joblib model artifact.")
    parser.add_argument("--metrics-json", required=True, help="Path for the metrics JSON.")
    args = parser.parse_args()

    train_baseline_model(
        input_csv=args.input_csv,
        config_path=args.config,
        model_output_path=args.model_output,
        metrics_json_path=args.metrics_json,
    )

    print(f"Wrote baseline model: {args.model_output}")
    print(f"Wrote metrics JSON: {args.metrics_json}")


if __name__ == "__main__":
    main()
