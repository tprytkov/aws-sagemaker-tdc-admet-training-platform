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
from admet_platform.models import train_local_baseline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a local classical ADMET baseline model.")
    parser.add_argument("--input-csv", help="Legacy path to one prepared input CSV.")
    parser.add_argument("--model-output", help="Legacy path for the joblib model artifact.")
    parser.add_argument("--metrics-json", help="Legacy path for the metrics JSON.")
    parser.add_argument("--train-csv", help="Path to the prepared train CSV.")
    parser.add_argument("--validation-csv", help="Path to the prepared validation CSV.")
    parser.add_argument("--test-csv", help="Path to the prepared test CSV.")
    parser.add_argument("--config", required=True, help="Path to the endpoint YAML config.")
    parser.add_argument("--feature-type", choices=["descriptors", "morgan"], help="Feature type to train on.")
    parser.add_argument("--output-dir", help="Directory for model and evaluation artifacts.")
    parser.add_argument("--morgan-radius", type=int, default=2)
    parser.add_argument("--morgan-bits", type=int, default=2048)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    if args.train_csv or args.validation_csv or args.test_csv or args.output_dir or args.feature_type:
        required_new_args = {
            "--train-csv": args.train_csv,
            "--validation-csv": args.validation_csv,
            "--test-csv": args.test_csv,
            "--feature-type": args.feature_type,
            "--output-dir": args.output_dir,
        }
        missing = [name for name, value in required_new_args.items() if not value]
        if missing:
            parser.error(f"Missing required split-based argument(s): {', '.join(missing)}")

        train_local_baseline(
            train_csv=args.train_csv,
            validation_csv=args.validation_csv,
            test_csv=args.test_csv,
            config_path=args.config,
            feature_type=args.feature_type,
            output_dir=args.output_dir,
            morgan_radius=args.morgan_radius,
            morgan_bits=args.morgan_bits,
            random_seed=args.random_seed,
        )
        print(f"Wrote baseline artifacts: {args.output_dir}")
        return

    if not args.input_csv or not args.model_output or not args.metrics_json:
        parser.error(
            "Use either split-based arguments or legacy --input-csv, --model-output, and --metrics-json."
        )

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
