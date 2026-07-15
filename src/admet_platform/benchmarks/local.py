"""Reproducible local benchmark runner for prepared ADMET datasets."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from admet_platform.config import EndpointConfig, load_endpoint_config
from admet_platform.data.prepare import prepare_dataset_artifacts
from admet_platform.data.tdc_loader import load_tdc_split, normalize_tdc_dataframe
from admet_platform.models.artifacts import to_json_safe, write_json


FEATURE_TYPES = ("descriptors", "morgan")
REQUIRED_PREPARED_FILES = (
    "train.csv",
    "valid.csv",
    "test.csv",
    "data_profile.json",
    "split_metadata.json",
    "rejected_rows.csv",
)


def run_local_benchmarks(
    config_paths: list[str | Path],
    prepared_root: str | Path = "outputs/local/full_datasets",
    benchmark_root: str | Path = "outputs/local/benchmarks",
    feature_types: list[str] | None = None,
    force_rerun: bool = False,
    random_seed: int = 42,
    morgan_radius: int = 2,
    morgan_bits: int = 2048,
    max_rows: int | None = None,
    prepare_dataset_func: Callable[..., dict[str, Any]] | None = None,
    train_func: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Prepare datasets, run local baselines, and write aggregate benchmark artifacts."""

    if train_func is None:
        from admet_platform.models.baseline import train_local_baseline

        train_func = train_local_baseline

    selected_features = feature_types or list(FEATURE_TYPES)
    prepared_base = Path(prepared_root)
    benchmark_base = Path(benchmark_root)
    benchmark_base.mkdir(parents=True, exist_ok=True)
    prepare_dataset_func = prepare_dataset_func or _prepare_real_tdc_dataset

    dataset_rows: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started_at = datetime.now(UTC)

    for config_path in config_paths:
        config = load_endpoint_config(config_path)
        prepared_dir = prepared_base / config.endpoint_id
        try:
            if force_rerun or not _prepared_dataset_exists(prepared_dir):
                prepare_dataset_func(
                    config_path=config_path,
                    output_dir=prepared_dir,
                    max_rows=max_rows,
                )
            dataset_summary = summarize_prepared_dataset(prepared_dir, config)
            dataset_summary["development_row_limit"] = max_rows
            dataset_rows.append(dataset_summary)
        except Exception as exc:  # noqa: BLE001 - benchmark should continue.
            failures.append(_failure(config, "prepare", None, exc))
            continue

        for feature_type in selected_features:
            run_output_dir = benchmark_base / config.endpoint_id / feature_type
            start = time.perf_counter()
            try:
                result = train_func(
                    train_csv=prepared_dir / "train.csv",
                    validation_csv=prepared_dir / "valid.csv",
                    test_csv=prepared_dir / "test.csv",
                    config_path=config_path,
                    feature_type=feature_type,
                    output_dir=run_output_dir,
                    morgan_radius=morgan_radius,
                    morgan_bits=morgan_bits,
                    random_seed=random_seed,
                )
                runtime_seconds = time.perf_counter() - start
                benchmark_rows.append(
                    _benchmark_row(
                        config=config,
                        feature_type=feature_type,
                        result=result,
                        output_dir=run_output_dir,
                        runtime_seconds=runtime_seconds,
                        status="success",
                    )
                )
            except Exception as exc:  # noqa: BLE001 - benchmark should continue.
                runtime_seconds = time.perf_counter() - start
                failures.append(_failure(config, "train", feature_type, exc))
                benchmark_rows.append(
                    _failed_benchmark_row(config, feature_type, run_output_dir, runtime_seconds, exc)
                )

    _write_aggregate_artifacts(
        benchmark_base=benchmark_base,
        benchmark_rows=benchmark_rows,
        dataset_rows=dataset_rows,
        failures=failures,
        metadata={
            "created_at": started_at.isoformat(),
            "completed_at": datetime.now(UTC).isoformat(),
            "config_paths": [str(path) for path in config_paths],
            "feature_types": selected_features,
            "random_seed": random_seed,
            "morgan_radius": morgan_radius,
            "morgan_bits": morgan_bits,
            "max_rows": max_rows,
            "development_only_row_limit": max_rows is not None,
            "scientific_safeguards": [
                "single_split_baseline_benchmark",
                "validation_and_test_metrics_recorded_separately",
                "no_hyperparameter_tuning",
                "test_metrics_not_used_for_model_selection",
            ],
        },
    )

    return {
        "benchmark_rows": benchmark_rows,
        "dataset_rows": dataset_rows,
        "failures": failures,
        "benchmark_root": str(benchmark_base),
        "success": not failures,
    }


def summarize_prepared_dataset(prepared_dir: str | Path, config: EndpointConfig) -> dict[str, Any]:
    prepared_path = Path(prepared_dir)
    train = pd.read_csv(prepared_path / "train.csv")
    validation = pd.read_csv(prepared_path / "valid.csv")
    test = pd.read_csv(prepared_path / "test.csv")
    rejected = pd.read_csv(prepared_path / "rejected_rows.csv")
    profile = json.loads((prepared_path / "data_profile.json").read_text(encoding="utf-8"))
    combined = pd.concat([train, validation, test], ignore_index=True)

    canonical_by_split = {
        "train": set(train["canonical_smiles"]),
        "validation": set(validation["canonical_smiles"]),
        "test": set(test["canonical_smiles"]),
    }
    overlap = {
        "train_validation": sorted(canonical_by_split["train"] & canonical_by_split["validation"]),
        "train_test": sorted(canonical_by_split["train"] & canonical_by_split["test"]),
        "validation_test": sorted(canonical_by_split["validation"] & canonical_by_split["test"]),
    }
    overlap_counts = {key: len(value) for key, value in overlap.items()}
    duplicate_count = int(combined["canonical_smiles"].duplicated().sum())
    warnings = []
    if any(overlap_counts.values()):
        warnings.append("canonical_smiles overlap detected across splits")

    row: dict[str, Any] = {
        "endpoint_id": config.endpoint_id,
        "tdc_name": config.tdc_name,
        "task_type": config.task_type,
        "total_source_rows": int(profile["n_rows"]),
        "accepted_rows": int(profile["n_accepted_rows"]),
        "rejected_rows": int(len(rejected)),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "duplicate_canonical_smiles_count": duplicate_count,
        "cross_split_overlap_counts": overlap_counts,
        "cross_split_overlap_examples": {key: value[:10] for key, value in overlap.items()},
        "warnings": warnings,
    }
    if config.task_type == "binary_classification":
        class_counts = combined["target"].astype(int).value_counts().sort_index()
        row["target_distribution"] = {str(label): int(count) for label, count in class_counts.items()}
        total = max(1, int(class_counts.sum()))
        minority_fraction = float(class_counts.min() / total) if len(class_counts) else 0.0
        row["minority_class_fraction"] = minority_fraction
        if minority_fraction < 0.1:
            row["warnings"].append("severe class imbalance detected")
    else:
        targets = pd.to_numeric(combined["target"], errors="coerce")
        row["target_mean"] = float(targets.mean())
        row["target_std"] = float(targets.std())
        row["target_min"] = float(targets.min())
        row["target_max"] = float(targets.max())
    return row


def _prepare_real_tdc_dataset(
    config_path: str | Path,
    output_dir: str | Path,
    max_rows: int | None = None,
) -> dict[str, Any]:
    config = load_endpoint_config(config_path)
    split_data = load_tdc_split(config)
    normalized = [
        normalize_tdc_dataframe(split_df, split_name, config)
        for split_name, split_df in split_data.items()
    ]
    df = pd.concat(normalized, ignore_index=True)
    if max_rows is not None:
        df = df.groupby("split", group_keys=False).head(max_rows).reset_index(drop=True)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    source_csv = output_path / "_source_normalized.csv"
    df.to_csv(source_csv, index=False)
    return prepare_dataset_artifacts(source_csv, config_path, output_path)


def _prepared_dataset_exists(prepared_dir: Path) -> bool:
    return all((prepared_dir / name).exists() for name in REQUIRED_PREPARED_FILES)


def _benchmark_row(
    config: EndpointConfig,
    feature_type: str,
    result: dict[str, Any],
    output_dir: Path,
    runtime_seconds: float,
    status: str,
) -> dict[str, Any]:
    training_metadata = result["training_metadata"]
    metrics = result["metrics"]
    validation_primary, validation_secondary = _primary_metrics(config.task_type, metrics["validation"])
    test_primary, test_secondary = _primary_metrics(config.task_type, metrics["test"])
    return {
        "endpoint_id": config.endpoint_id,
        "task_type": config.task_type,
        "feature_type": feature_type,
        "model_type": training_metadata["model_type"],
        "train_rows": training_metadata["training_row_count"],
        "validation_rows": training_metadata["validation_row_count"],
        "test_rows": training_metadata["test_row_count"],
        "feature_count": training_metadata["feature_count"],
        "primary_validation_metric": validation_primary,
        "secondary_validation_metric": validation_secondary,
        "primary_test_metric": test_primary,
        "secondary_test_metric": test_secondary,
        "output_directory": str(output_dir),
        "warnings": metrics.get("warnings", []),
        "runtime_seconds": float(runtime_seconds),
        "run_status": status,
    }


def _failed_benchmark_row(
    config: EndpointConfig,
    feature_type: str,
    output_dir: Path,
    runtime_seconds: float,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "endpoint_id": config.endpoint_id,
        "task_type": config.task_type,
        "feature_type": feature_type,
        "model_type": None,
        "train_rows": None,
        "validation_rows": None,
        "test_rows": None,
        "feature_count": None,
        "primary_validation_metric": None,
        "secondary_validation_metric": None,
        "primary_test_metric": None,
        "secondary_test_metric": None,
        "output_directory": str(output_dir),
        "warnings": [str(exc)],
        "runtime_seconds": float(runtime_seconds),
        "run_status": "failed",
    }


def _primary_metrics(task_type: str, metrics: dict[str, Any]) -> tuple[Any, Any]:
    if task_type == "binary_classification":
        return metrics.get("roc_auc"), metrics.get("pr_auc")
    return metrics.get("rmse"), metrics.get("mae")


def _failure(config: EndpointConfig, stage: str, feature_type: str | None, exc: Exception) -> dict[str, Any]:
    return {
        "endpoint_id": config.endpoint_id,
        "stage": stage,
        "feature_type": feature_type,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }


def _write_aggregate_artifacts(
    benchmark_base: Path,
    benchmark_rows: list[dict[str, Any]],
    dataset_rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    pd.DataFrame(benchmark_rows).to_csv(benchmark_base / "benchmark_summary.csv", index=False)
    pd.DataFrame(dataset_rows).to_csv(benchmark_base / "dataset_summary.csv", index=False)
    write_json(benchmark_base / "benchmark_summary.json", {"runs": benchmark_rows})
    write_json(benchmark_base / "benchmark_failures.json", {"failures": failures})
    write_json(benchmark_base / "benchmark_metadata.json", metadata)
