"""Local schema validation for normalized ADMET datasets."""

from __future__ import annotations

from typing import Any

import pandas as pd

from admet_platform.config import EndpointConfig


REQUIRED_BASE_COLUMNS = ("molecule_id",)
VALID_SPLITS = {"train", "validation", "test"}


def validate_dataset_columns(df: pd.DataFrame, config: EndpointConfig) -> None:
    """Validate that the normalized dataset has required structural columns."""

    _validate_non_empty_dataframe(df)
    required_columns = [*REQUIRED_BASE_COLUMNS, config.smiles_column, config.target_column]
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Dataset is missing required column(s): {missing}.")


def validate_smiles_column(df: pd.DataFrame, config: EndpointConfig) -> None:
    """Validate that SMILES values are present and non-empty."""

    _validate_non_empty_dataframe(df)
    if config.smiles_column not in df.columns:
        raise ValueError(f"Dataset is missing SMILES column '{config.smiles_column}'.")

    smiles_values = df[config.smiles_column]
    missing_or_empty = smiles_values.isna() | smiles_values.astype(str).str.strip().eq("")
    if missing_or_empty.any():
        raise ValueError(f"Dataset column '{config.smiles_column}' contains missing or empty SMILES.")


def validate_target_column(df: pd.DataFrame, config: EndpointConfig) -> None:
    """Validate endpoint target values for classification or regression."""

    _validate_non_empty_dataframe(df)
    if config.target_column not in df.columns:
        raise ValueError(f"Dataset is missing target column '{config.target_column}'.")

    targets = df[config.target_column].dropna()
    if config.task_type == "binary_classification":
        invalid_targets = ~targets.isin([0, 1, 0.0, 1.0, "0", "1"])
        if invalid_targets.any():
            raise ValueError(
                f"Binary classification target column '{config.target_column}' must contain only 0 or 1."
            )
        return

    if config.task_type == "regression":
        numeric_targets = pd.to_numeric(targets, errors="coerce")
        if numeric_targets.isna().any():
            raise ValueError(f"Regression target column '{config.target_column}' must be numeric.")
        return

    raise ValueError(f"Unsupported task_type '{config.task_type}'.")


def validate_split_column(df: pd.DataFrame, split_column: str = "split") -> None:
    """Validate train/validation/test split labels."""

    _validate_non_empty_dataframe(df)
    if split_column not in df.columns:
        raise ValueError(f"Dataset is missing split column '{split_column}'.")

    split_values = df[split_column].dropna().astype(str)
    invalid_splits = sorted(set(split_values) - VALID_SPLITS)
    if invalid_splits:
        invalid = ", ".join(invalid_splits)
        allowed = ", ".join(sorted(VALID_SPLITS))
        raise ValueError(f"Split column '{split_column}' has invalid value(s): {invalid}. Allowed: {allowed}.")


def summarize_dataset(df: pd.DataFrame, config: EndpointConfig) -> dict[str, Any]:
    """Validate and summarize a normalized local ADMET dataset."""

    validate_dataset_columns(df, config)
    validate_smiles_column(df, config)
    validate_target_column(df, config)
    validate_split_column(df)

    split_counts = df["split"].value_counts().reindex(["train", "validation", "test"], fill_value=0)
    return {
        "endpoint_id": config.endpoint_id,
        "task_type": config.task_type,
        "row_count": int(len(df)),
        "molecule_count": int(df["molecule_id"].nunique()),
        "split_counts": {split: int(count) for split, count in split_counts.items()},
        "metric_names": list(config.metric_names),
    }


def _validate_non_empty_dataframe(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Dataset must not be empty.")
