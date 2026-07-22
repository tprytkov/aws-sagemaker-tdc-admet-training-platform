"""Optional TDC dataset loading and normalization."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from admet_platform.config import EndpointConfig, load_endpoint_config
from admet_platform.data.prepare import _build_raw_summary
from admet_platform.data.schema import (
    validate_dataset_columns,
    validate_smiles_column,
    validate_target_column,
)


SPLIT_NAME_MAP = {
    "train": "train",
    "valid": "validation",
    "validation": "validation",
    "test": "test",
}


def load_tdc_data(config: EndpointConfig) -> pd.DataFrame:
    """Load unsplit public TDC records without invoking TDC scaffold code."""

    loader_class = _get_tdc_loader_class(config.task_group)
    dataset = loader_class(name=config.tdc_name)
    return dataset.get_data()


def load_tdc_split(config: EndpointConfig) -> pd.DataFrame:
    """Backward-compatible name for unsplit TDC loading.

    This intentionally returns raw records and never calls ``dataset.get_split``.
    """

    return load_tdc_data(config)


def normalize_tdc_raw_dataframe(df: pd.DataFrame, config: EndpointConfig) -> pd.DataFrame:
    """Normalize unsplit TDC records to the local preparation input contract."""

    smiles_column = _resolve_column(df, [config.smiles_column, "Drug", "SMILES", "smiles"])
    target_column = _resolve_column(df, [config.target_column, "Y", "target", "Label", "label"])
    molecule_id_column = _resolve_optional_column(
        df, ["molecule_id", "Drug_ID", "Drug_IDs", "id", "ID", "Index"]
    )
    normalized = pd.DataFrame(
        {
            "molecule_id": (
                df[molecule_id_column].astype(str)
                if molecule_id_column is not None
                else [f"{config.endpoint_id}_raw_{index:06d}" for index in range(len(df))]
            ),
            "smiles": df[smiles_column],
            "target": df[target_column],
        }
    )
    for column in ("molecule_id", "smiles"):
        normalized[column] = normalized[column].astype("string").str.strip()
    if config.task_type == "regression":
        normalized["target"] = pd.to_numeric(normalized["target"], errors="coerce")
    else:
        normalized["target"] = pd.to_numeric(normalized["target"], errors="raise").astype(int)
    return normalized[["molecule_id", "smiles", "target"]]


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
    raw_data = load_tdc_data(config)
    cleaned_df = normalize_tdc_raw_dataframe(raw_data, config).drop_duplicates(ignore_index=True)

    validate_dataset_columns(cleaned_df, config)
    validate_smiles_column(cleaned_df, config)
    validate_target_column(cleaned_df, config)

    output_path = Path(output_csv)
    summary_path = Path(summary_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary = _build_raw_summary(cleaned_df, config)
    _atomic_write_csv(cleaned_df, output_path)
    _atomic_write_text(summary_path, json.dumps(summary, indent=2) + "\n")
    return summary


def _atomic_write_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", suffix=".tmp",
        dir=destination.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    os.replace(temporary, destination)


def _atomic_write_text(destination: Path, content: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".tmp", dir=destination.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    os.replace(temporary, destination)


def _get_tdc_loader_class(task_group: str) -> type:
    try:
        from tdc.single_pred import ADME, Tox
    except ImportError as exc:
        raise RuntimeError(
            "PyTDC is required only for TDC dataset downloads. Install the verified download "
            "dependencies with `pip install -r requirements-tdc-download.txt`."
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
