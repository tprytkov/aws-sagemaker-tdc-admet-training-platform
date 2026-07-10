"""Local dataset preparation for normalized ADMET CSV files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from admet_platform.config import EndpointConfig, load_endpoint_config
from admet_platform.data.schema import (
    validate_dataset_columns,
    validate_smiles_column,
    validate_split_column,
    validate_target_column,
)


NORMALIZED_COLUMNS = ["molecule_id", "smiles", "target", "split"]


def prepare_dataset(
    input_csv: str | Path,
    config_path: str | Path,
    output_csv: str | Path,
    summary_json: str | Path,
) -> dict[str, Any]:
    """Validate, clean, and summarize a local endpoint CSV."""

    config = load_endpoint_config(config_path)
    raw_df = pd.read_csv(input_csv)

    validate_dataset_columns(raw_df, config)
    validate_split_column(raw_df)

    cleaned_df = _normalize_dataframe(raw_df, config)
    validate_dataset_columns(cleaned_df, config)
    validate_smiles_column(cleaned_df, config)
    validate_target_column(cleaned_df, config)
    validate_split_column(cleaned_df)

    output_path = Path(output_csv)
    summary_path = Path(summary_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    cleaned_df.to_csv(output_path, index=False)
    summary = _build_summary(cleaned_df, config)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    return summary


def _normalize_dataframe(df: pd.DataFrame, config: EndpointConfig) -> pd.DataFrame:
    normalized = df.rename(
        columns={
            config.smiles_column: "smiles",
            config.target_column: "target",
        }
    )
    normalized = normalized[NORMALIZED_COLUMNS].copy()

    for column in ("molecule_id", "smiles", "split"):
        normalized[column] = normalized[column].astype("string").str.strip()

    if config.task_type == "regression":
        normalized["target"] = pd.to_numeric(normalized["target"], errors="coerce")
    elif config.task_type == "binary_classification":
        binary_target = pd.to_numeric(normalized["target"], errors="raise")
        if binary_target.isna().any():
            normalized["target"] = binary_target.astype("Int64")
        else:
            normalized["target"] = binary_target.astype(int)

    return normalized.drop_duplicates(ignore_index=True)


def _build_summary(df: pd.DataFrame, config: EndpointConfig) -> dict[str, Any]:
    split_counts = df["split"].value_counts()
    summary: dict[str, Any] = {
        "endpoint_id": config.endpoint_id,
        "tdc_name": config.tdc_name,
        "task_type": config.task_type,
        "n_rows": int(len(df)),
        "n_train": int(split_counts.get("train", 0)),
        "n_validation": int(split_counts.get("validation", 0)),
        "n_test": int(split_counts.get("test", 0)),
        "n_unique_smiles": int(df["smiles"].nunique()),
    }

    targets = df["target"].dropna()
    if config.task_type == "regression":
        numeric_targets = pd.to_numeric(targets, errors="coerce")
        summary["target_min"] = float(numeric_targets.min())
        summary["target_max"] = float(numeric_targets.max())
    elif config.task_type == "binary_classification":
        class_counts = targets.astype(int).value_counts().sort_index()
        summary["class_counts"] = {str(label): int(count) for label, count in class_counts.items()}

    return summary
