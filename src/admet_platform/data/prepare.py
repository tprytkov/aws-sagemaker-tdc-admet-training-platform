"""Local dataset preparation for normalized ADMET CSV files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem

from admet_platform.config import EndpointConfig, load_endpoint_config
from admet_platform.data.schema import (
    validate_dataset_columns,
    validate_smiles_column,
    validate_split_column,
    validate_target_column,
)


NORMALIZED_COLUMNS = ["molecule_id", "smiles", "target", "split"]
ARTIFACT_COLUMNS = ["molecule_id", "smiles", "canonical_smiles", "target", "split"]
REJECTED_ROW_COLUMNS = ["molecule_id", "smiles", "target", "split", "rejection_reason"]


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


def prepare_dataset_artifacts(
    input_csv: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Prepare split CSVs and metadata artifacts for a local ADMET dataset."""

    config = load_endpoint_config(config_path)
    raw_df = pd.read_csv(input_csv)

    validate_dataset_columns(raw_df, config)
    validate_split_column(raw_df)

    normalized_df = _normalize_dataframe(raw_df, config)
    accepted_df, rejected_df = _canonicalize_and_filter_smiles(normalized_df)

    if not accepted_df.empty:
        validate_dataset_columns(accepted_df, config)
        validate_smiles_column(accepted_df, config)
        validate_target_column(accepted_df, config)
        validate_split_column(accepted_df)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    _write_split_csvs(accepted_df, output_path)
    rejected_df.to_csv(output_path / "rejected_rows.csv", index=False)

    profile = _build_data_profile(accepted_df, rejected_df, config)
    split_metadata = _build_split_metadata(accepted_df, config)

    (output_path / "data_profile.json").write_text(
        json.dumps(profile, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_path / "split_metadata.json").write_text(
        json.dumps(split_metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    return profile


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


def _canonicalize_and_filter_smiles(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    accepted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []

    for row in df.to_dict(orient="records"):
        smiles = str(row["smiles"]).strip() if pd.notna(row["smiles"]) else ""
        rejection_reason = ""
        canonical_smiles = ""

        if not smiles:
            rejection_reason = "missing_smiles"
        else:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                rejection_reason = "invalid_smiles"
            else:
                canonical_smiles = Chem.MolToSmiles(molecule, canonical=True)

        if rejection_reason:
            rejected_rows.append(
                {
                    "molecule_id": row.get("molecule_id", ""),
                    "smiles": row.get("smiles", ""),
                    "target": row.get("target", ""),
                    "split": row.get("split", ""),
                    "rejection_reason": rejection_reason,
                }
            )
        else:
            accepted_rows.append(
                {
                    "molecule_id": row["molecule_id"],
                    "smiles": row["smiles"],
                    "canonical_smiles": canonical_smiles,
                    "target": row["target"],
                    "split": row["split"],
                }
            )

    accepted_df = pd.DataFrame(accepted_rows, columns=ARTIFACT_COLUMNS).drop_duplicates(ignore_index=True)
    rejected_df = pd.DataFrame(rejected_rows, columns=REJECTED_ROW_COLUMNS)
    return accepted_df, rejected_df


def _write_split_csvs(df: pd.DataFrame, output_dir: Path) -> None:
    split_to_file = {
        "train": "train.csv",
        "validation": "valid.csv",
        "test": "test.csv",
    }
    for split_name, file_name in split_to_file.items():
        split_df = df[df["split"] == split_name] if not df.empty else pd.DataFrame(columns=ARTIFACT_COLUMNS)
        split_df.to_csv(output_dir / file_name, index=False)


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


def _build_data_profile(
    accepted_df: pd.DataFrame,
    rejected_df: pd.DataFrame,
    config: EndpointConfig,
) -> dict[str, Any]:
    split_counts = accepted_df["split"].value_counts() if not accepted_df.empty else pd.Series(dtype=int)
    profile: dict[str, Any] = {
        "endpoint_id": config.endpoint_id,
        "tdc_name": config.tdc_name,
        "task_group": config.task_group,
        "task_type": config.task_type,
        "source_dataset": config.tdc_name,
        "n_rows": int(len(accepted_df) + len(rejected_df)),
        "n_accepted_rows": int(len(accepted_df)),
        "n_rejected_rows": int(len(rejected_df)),
        "n_train": int(split_counts.get("train", 0)),
        "n_validation": int(split_counts.get("validation", 0)),
        "n_test": int(split_counts.get("test", 0)),
        "n_unique_smiles": int(accepted_df["smiles"].nunique()) if not accepted_df.empty else 0,
        "n_unique_canonical_smiles": (
            int(accepted_df["canonical_smiles"].nunique()) if not accepted_df.empty else 0
        ),
    }

    targets = accepted_df["target"].dropna() if not accepted_df.empty else pd.Series(dtype=float)
    if config.task_type == "regression":
        numeric_targets = pd.to_numeric(targets, errors="coerce")
        profile["target_min"] = float(numeric_targets.min()) if not numeric_targets.empty else None
        profile["target_max"] = float(numeric_targets.max()) if not numeric_targets.empty else None
    elif config.task_type == "binary_classification":
        class_counts = targets.astype(int).value_counts().sort_index()
        profile["class_counts"] = {str(label): int(count) for label, count in class_counts.items()}

    return profile


def _build_split_metadata(df: pd.DataFrame, config: EndpointConfig) -> dict[str, Any]:
    split_counts = df["split"].value_counts() if not df.empty else pd.Series(dtype=int)
    return {
        "endpoint_id": config.endpoint_id,
        "tdc_name": config.tdc_name,
        "source_dataset": config.tdc_name,
        "task_group": config.task_group,
        "task_type": config.task_type,
        "split_strategy": config.split_strategy,
        "split_counts": {
            "train": int(split_counts.get("train", 0)),
            "validation": int(split_counts.get("validation", 0)),
            "test": int(split_counts.get("test", 0)),
        },
        "artifacts": {
            "train": "train.csv",
            "validation": "valid.csv",
            "test": "test.csv",
            "data_profile": "data_profile.json",
            "split_metadata": "split_metadata.json",
            "rejected_rows": "rejected_rows.csv",
        },
    }
