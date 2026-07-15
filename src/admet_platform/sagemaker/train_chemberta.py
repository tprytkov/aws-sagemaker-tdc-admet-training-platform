"""SageMaker-compatible ChemBERTa training entry point."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from admet_platform.config import load_endpoint_config
from admet_platform.models.artifacts import write_json
from admet_platform.models.chemberta import (
    DEFAULT_CHEMBERTA_MODEL,
    ChemBERTaTrainingConfig,
    train_chemberta_model,
)


REQUIRED_COLUMNS = {"molecule_id", "canonical_smiles", "target", "split"}
OUTPUT_ARTIFACTS = [
    "metrics.json",
    "predictions_validation.csv",
    "predictions_test.csv",
    "training_metadata.json",
    "training_history.json",
    "warnings.json",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SageMaker ChemBERTa training entry point.")
    parser.add_argument("--endpoint-config", required=True)
    parser.add_argument("--model-name", default=DEFAULT_CHEMBERTA_MODEL)
    parser.add_argument("--max-sequence-length", default="128")
    parser.add_argument("--learning-rate", default="2e-5")
    parser.add_argument("--epochs", default="3")
    parser.add_argument("--train-batch-size", default="8")
    parser.add_argument("--evaluation-batch-size", default="16")
    parser.add_argument("--weight-decay", default="0.01")
    parser.add_argument("--early-stopping-patience", default="2")
    parser.add_argument("--random-seed", default="42")
    parser.add_argument("--development-row-limit", default=None)
    parser.add_argument("--local-files-only", default="false")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--train-channel")
    parser.add_argument("--validation-channel")
    parser.add_argument("--test-channel")
    parser.add_argument("--model-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--checkpoint-dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    start = datetime.now(UTC)
    start_time = time.perf_counter()
    output_dir: Path | None = None
    try:
        paths = resolve_sagemaker_paths(args, os.environ)
        output_dir = paths["output_dir"]
        config = load_endpoint_config(args.endpoint_config)
        hyperparameters = parse_hyperparameters(args)
        train_csv = resolve_channel_csv(paths["train_channel"])
        validation_csv = resolve_channel_csv(paths["validation_channel"])
        test_csv = resolve_channel_csv(paths["test_channel"])
        validate_required_columns(train_csv)
        validate_required_columns(validation_csv)
        validate_required_columns(test_csv)

        result = train_chemberta_model(
            train_csv=train_csv,
            validation_csv=validation_csv,
            test_csv=test_csv,
            config_path=args.endpoint_config,
            output_dir=output_dir,
            model_dir=paths["model_dir"],
            training_config=ChemBERTaTrainingConfig(**hyperparameters),
        )
        _copy_model_config_to_model_dir(output_dir, paths["model_dir"])
        manifest = build_run_manifest(
            run_id=str(uuid.uuid4()),
            config=config,
            paths=paths,
            hyperparameters=hyperparameters,
            result=result,
            start=start,
            runtime_seconds=time.perf_counter() - start_time,
            status="completed",
            error=None,
        )
        write_json(output_dir / "run_manifest.json", manifest)
        return 0
    except Exception as exc:  # noqa: BLE001 - entry point must write failure record.
        if output_dir is None:
            try:
                output_dir = resolve_output_dir(args, os.environ)
            except Exception:
                output_dir = None
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            write_json(
                output_dir / "run_manifest.json",
                {
                    "run_id": str(uuid.uuid4()),
                    "status": "failed",
                    "error": {
                        "type": type(exc).__name__,
                        "message": sanitize_message(str(exc)),
                    },
                    "completed_at": datetime.now(UTC).isoformat(),
                },
            )
        print(f"SageMaker ChemBERTa training failed: {sanitize_message(str(exc))}", file=sys.stderr)
        return 1


def parse_hyperparameters(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model_name": args.model_name,
        "max_sequence_length": parse_int(args.max_sequence_length, "max_sequence_length"),
        "learning_rate": parse_float(args.learning_rate, "learning_rate"),
        "training_epochs": parse_int(args.epochs, "epochs"),
        "train_batch_size": parse_int(args.train_batch_size, "train_batch_size"),
        "evaluation_batch_size": parse_int(args.evaluation_batch_size, "evaluation_batch_size"),
        "weight_decay": parse_float(args.weight_decay, "weight_decay"),
        "early_stopping_patience": parse_int(args.early_stopping_patience, "early_stopping_patience"),
        "random_seed": parse_int(args.random_seed, "random_seed"),
        "development_row_limit": parse_optional_int(args.development_row_limit, "development_row_limit"),
        "local_files_only": parse_bool(args.local_files_only, "local_files_only"),
        "cache_dir": parse_optional_str(args.cache_dir),
    }


def parse_bool(value: str | bool | None, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value for {name}: {value!r}.")


def parse_int(value: str | int, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer value for {name}: {value!r}.") from exc


def parse_optional_int(value: str | int | None, name: str) -> int | None:
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        return None
    return parse_int(value, name)


def parse_float(value: str | float, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid float value for {name}: {value!r}.") from exc


def parse_optional_str(value: str | None) -> str | None:
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        return None
    return str(value)


def resolve_sagemaker_paths(args: argparse.Namespace, env: dict[str, str]) -> dict[str, Path]:
    return {
        "train_channel": Path(args.train_channel or require_env(env, "SM_CHANNEL_TRAIN")),
        "validation_channel": Path(
            args.validation_channel
            or env.get("SM_CHANNEL_VALIDATION")
            or env.get("SM_CHANNEL_VALID")
            or missing_channel("SM_CHANNEL_VALIDATION")
        ),
        "test_channel": Path(args.test_channel or require_env(env, "SM_CHANNEL_TEST")),
        "model_dir": Path(args.model_dir or require_env(env, "SM_MODEL_DIR")),
        "output_dir": resolve_output_dir(args, env),
        "checkpoint_dir": Path(args.checkpoint_dir or env.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints")),
    }


def resolve_output_dir(args: argparse.Namespace, env: dict[str, str]) -> Path:
    return Path(args.output_dir or env.get("SM_OUTPUT_DIR", "/opt/ml/output/data"))


def require_env(env: dict[str, str], name: str) -> str:
    value = env.get(name)
    if not value:
        raise ValueError(f"Required SageMaker environment variable is missing: {name}.")
    return value


def missing_channel(name: str) -> str:
    raise ValueError(f"Required SageMaker channel is missing: {name}.")


def resolve_channel_csv(channel_dir: str | Path) -> Path:
    path = Path(channel_dir)
    if not path.exists():
        raise ValueError(f"Required channel directory is missing: {path}.")
    csv_files = sorted(file for file in path.iterdir() if file.is_file() and file.suffix.lower() == ".csv")
    if not csv_files:
        raise ValueError(f"No CSV file found in channel directory: {path}.")
    if len(csv_files) > 1:
        names = ", ".join(file.name for file in csv_files)
        raise ValueError(f"Multiple CSV files found in channel directory {path}: {names}.")
    return csv_files[0]


def validate_required_columns(csv_path: str | Path) -> None:
    import pandas as pd

    columns = set(pd.read_csv(csv_path, nrows=0).columns)
    missing = sorted(REQUIRED_COLUMNS - columns)
    if missing:
        raise ValueError(f"Input CSV {csv_path} is missing required column(s): {', '.join(missing)}.")


def build_run_manifest(
    run_id: str,
    config: Any,
    paths: dict[str, Path],
    hyperparameters: dict[str, Any],
    result: dict[str, Any],
    start: datetime,
    runtime_seconds: float,
    status: str,
    error: dict[str, str] | None,
) -> dict[str, Any]:
    model_files = _relative_files(paths["model_dir"])
    output_files = _relative_files(paths["output_dir"])
    metadata = result.get("training_metadata", {})
    return {
        "run_id": run_id,
        "endpoint_id": config.endpoint_id,
        "task_type": config.task_type,
        "source_dataset": config.tdc_name,
        "input_channels": {
            "train": str(paths["train_channel"]),
            "validation": str(paths["validation_channel"]),
            "test": str(paths["test_channel"]),
        },
        "model_dir": str(paths["model_dir"]),
        "output_data_dir": str(paths["output_dir"]),
        "checkpoint_dir": str(paths["checkpoint_dir"]),
        "pretrained_checkpoint": hyperparameters["model_name"],
        "hyperparameters": hyperparameters,
        "train_count": metadata.get("training_row_count"),
        "validation_count": metadata.get("validation_row_count"),
        "test_count": metadata.get("test_row_count"),
        "model_artifact_files": model_files,
        "output_artifact_files": output_files,
        "started_at": start.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "runtime_seconds": runtime_seconds,
        "status": status,
        "warnings": metadata.get("warnings", []),
        "development_mode": hyperparameters.get("development_row_limit") is not None,
        "error": error,
    }


def sanitize_message(message: str) -> str:
    redacted = message
    for marker in ["AWS_SECRET_ACCESS_KEY", "AWS_ACCESS_KEY_ID", "AWS_SESSION_TOKEN", "HF_TOKEN"]:
        redacted = redacted.replace(marker, "[REDACTED]")
    return redacted


def _relative_files(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())


def _copy_model_config_to_model_dir(output_dir: Path, model_dir: Path) -> None:
    source = output_dir / "model_config.json"
    if source.exists():
        shutil.copy2(source, model_dir / "model_config.json")
