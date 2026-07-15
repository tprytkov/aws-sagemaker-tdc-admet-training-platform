"""Command-line interface for local ChemBERTa fine-tuning."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from admet_platform.models.chemberta import (  # noqa: E402
    DEFAULT_CHEMBERTA_MODEL,
    ChemBERTaTrainingConfig,
    train_chemberta_model,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune ChemBERTa locally on prepared ADMET splits.")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--validation-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--config", required=True, help="Endpoint YAML config.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default=DEFAULT_CHEMBERTA_MODEL)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--row-limit", type=int, help="Development-only row limit per split.")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-sequence-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--evaluation-batch-size", type=int, default=16)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    training_config = ChemBERTaTrainingConfig(
        model_name=args.model_name,
        max_sequence_length=args.max_sequence_length,
        learning_rate=args.learning_rate,
        training_epochs=args.epochs if args.epochs is not None else 3,
        train_batch_size=args.train_batch_size,
        evaluation_batch_size=args.evaluation_batch_size,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        random_seed=args.random_seed,
        development_row_limit=args.row_limit,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    train_chemberta_model(
        train_csv=args.train_csv,
        validation_csv=args.validation_csv,
        test_csv=args.test_csv,
        config_path=args.config,
        output_dir=args.output_dir,
        training_config=training_config,
    )
    print(f"Wrote ChemBERTa artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
