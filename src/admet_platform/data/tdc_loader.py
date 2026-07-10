"""Optional TDC dataset loading and normalization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from admet_platform.config import EndpointConfig, load_endpoint_config
from admet_platform.data.prepare import _build_summary
from admet_platform.data.schema import (
    validate_dataset_columns,
    validate_smiles_column,
    validate_split_column,
    validate_target_column,
)


SPLIT_NAME_MAP = {
    "train": "train",
    "valid": "validation",
    "validation": "validation",
    "test": "test",
}


def load_tdc_split(config: EndpointConfig) -> dict[str, pd.DataFrame]:
    """Load a public TDC split for an endpoint config."""

    loader_class = _get_tdc_loader_class(config.task_group)
    dataset = loader_class(name=config.tdc_name)
    return dataset.get_split(method=config.split_strategy)


def normalize_tdc_dataframe(
    df: pd.DataFrame,
    split_name: str,
    config: EndpointConfig,
) -> pd.DataFrame:
    """Normalize one TDC split DataFrame to the project CSV schema."""

    normalized_split = _normalize_split_name(split_name)
    smiles_column = _resolve_column(df, [config.smiles_column, "Drug", "SMILES", "smiles"])
    target_column = _resolve_column(df, [config.target_column, "Y", "target", "Label", "label"])
    molecule_id_column = _resolve_optional_column(
        df,
        ["molecule_id", "Drug_ID", "Drug_IDs", "id", "ID", "Index"],
    )

    normalized = pd.DataFrame(
        {
            "molecule_id": (
                df[molecule_id_column].astype(str)
                if molecule_id_column is not None
                else [
                    f"{config.endpoint_id}_{normalized_split}_{index:06d}"
                    for index in range(len(df))
                ]
            ),
            "smiles": df[smiles_column],
            "target": df[target_column],
            "split": normalized_split,
        }
    )

    for column in ("molecule_id", "smiles", "split"):
        normalized[column] = normalized[column].astype("string").str.strip()

    if config.task_type == "regression":
        normalized["target"] = pd.to_numeric(normalized["target"], errors="coerce")
    elif config.task_type == "binary_classification":
        normalized["target"] = pd.to_numeric(normalized["target"], errors="raise").astype(int)

    return normalized[["molecule_id", "smiles", "target", "split"]]


def download_and_prepare_tdc_dataset(
    config_path: str | Path,
    output_csv: str | Path,
    summary_json: str | Path,
) -> dict[str, Any]:
    """Download, normalize, validate, and summarize a public TDC endpoint dataset."""

    config = load_endpoint_config(config_path)
    split_data = load_tdc_split(config)
    normalized_splits = [
        normalize_tdc_dataframe(split_df, split_name, config)
        for split_name, split_df in split_data.items()
    ]
    cleaned_df = pd.concat(normalized_splits, ignore_index=True).drop_duplicates(ignore_index=True)

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


def _get_tdc_loader_class(task_group: str) -> type:
    try:
        from tdc.single_pred import ADME, Tox
    except ImportError as exc:
        raise RuntimeError(
            "PyTDC is required to download TDC datasets. Install project dependencies with "
            "`pip install -r requirements.txt` or install PyTDC directly."
        ) from exc

    if task_group == "ADME":
        return ADME
    if task_group == "Tox":
        return Tox
    raise ValueError(f"Unsupported TDC task_group '{task_group}'.")


def _normalize_split_name(split_name: str) -> str:
    normalized = SPLIT_NAME_MAP.get(split_name)
    if normalized is None:
        allowed = ", ".join(sorted(SPLIT_NAME_MAP))
        raise ValueError(f"Unsupported TDC split name '{split_name}'. Expected one of: {allowed}.")
    return normalized


def _resolve_column(df: pd.DataFrame, candidates: list[str]) -> str:
    column = _resolve_optional_column(df, candidates)
    if column is None:
        candidate_list = ", ".join(candidates)
        raise ValueError(f"TDC DataFrame is missing expected column. Tried: {candidate_list}.")
    return column


def _resolve_optional_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None
