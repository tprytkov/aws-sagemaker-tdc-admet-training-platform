"""Local dataset preparation for normalized ADMET CSV files."""

from __future__ import annotations

import json
import os
import tempfile
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
from admet_platform.data.scaffolds import safe_murcko_scaffold


NORMALIZED_COLUMNS = ["molecule_id", "smiles", "target", "split"]
ARTIFACT_COLUMNS = ["molecule_id", "smiles", "canonical_smiles", "target", "split"]
REJECTED_ROW_COLUMNS = ["molecule_id", "smiles", "target", "split", "rejection_reason"]
PROBLEM_COLUMNS = [
    "source_row", "molecule_id", "original_smiles", "endpoint", "failure_stage",
    "exception_category", "error_message",
]
DEFAULT_SPLIT_FRACTIONS = {"train": 0.8, "validation": 0.1, "test": 0.1}


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
    has_split = "split" in raw_df.columns
    if has_split:
        validate_split_column(raw_df)

    cleaned_df = _normalize_dataframe(raw_df, config, require_split=has_split)
    if not has_split:
        accepted, rejected, _ = _canonicalize_and_filter_smiles(cleaned_df, config)
        cleaned_df, scaffold_rejected, _ = _assign_deterministic_scaffold_splits(accepted, config)
        rejected = pd.concat([rejected, scaffold_rejected], ignore_index=True)
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
    has_split = "split" in raw_df.columns
    if has_split:
        validate_split_column(raw_df)

    normalized_df = _normalize_dataframe(raw_df, config, require_split=has_split)
    accepted_df, rejected_df, problem_df = _canonicalize_and_filter_smiles(normalized_df, config)
    if not has_split:
        accepted_df, scaffold_rejected, scaffold_problems = _assign_deterministic_scaffold_splits(
            accepted_df, config
        )
        rejected_df = pd.concat([rejected_df, scaffold_rejected], ignore_index=True)
        problem_df = pd.concat([problem_df, scaffold_problems], ignore_index=True)

    if not accepted_df.empty:
        validate_dataset_columns(accepted_df, config)
        validate_smiles_column(accepted_df, config)
        validate_target_column(accepted_df, config)
        validate_split_column(accepted_df)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    _write_split_csvs(accepted_df, output_path)
    _atomic_write_csv(rejected_df, output_path / "rejected_rows.csv")
    _atomic_write_csv(problem_df, output_path / "problematic_molecules.csv")

    profile = _build_data_profile(accepted_df, rejected_df, config)
    split_metadata = _build_split_metadata(accepted_df, config)

    _atomic_write_text(output_path / "data_profile.json", json.dumps(profile, indent=2) + "\n")
    _atomic_write_text(output_path / "split_metadata.json", json.dumps(split_metadata, indent=2) + "\n")

    return profile


def _normalize_dataframe(
    df: pd.DataFrame, config: EndpointConfig, *, require_split: bool = True
) -> pd.DataFrame:
    normalized = df.rename(
        columns={
            config.smiles_column: "smiles",
            config.target_column: "target",
        }
    )
    columns = NORMALIZED_COLUMNS if require_split else NORMALIZED_COLUMNS[:-1]
    normalized = normalized[columns].copy()

    string_columns = ("molecule_id", "smiles", "split") if require_split else ("molecule_id", "smiles")
    for column in string_columns:
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


def _canonicalize_and_filter_smiles(
    df: pd.DataFrame, config: EndpointConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    accepted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    problem_rows: list[dict[str, Any]] = []

    for source_row, row in enumerate(df.to_dict(orient="records")):
        smiles = str(row["smiles"]).strip() if pd.notna(row["smiles"]) else ""
        rejection_reason = ""
        canonical_smiles = ""

        error: Exception | None = None
        if not smiles:
            rejection_reason = "missing_smiles"
        else:
            try:
                molecule = Chem.MolFromSmiles(smiles)
                if molecule is None:
                    rejection_reason = "invalid_smiles"
                else:
                    canonical_smiles = Chem.MolToSmiles(molecule, canonical=True)
            except Exception as exc:  # RDKit exception classes vary by build.
                error = exc
                rejection_reason = "smiles_processing_failed"

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
            problem_rows.append(_problem_record(
                source_row, row, config, "smiles_parsing",
                error or ValueError(rejection_reason),
            ))
        else:
            accepted_rows.append(
                {
                    "molecule_id": row["molecule_id"],
                    "smiles": row["smiles"],
                    "canonical_smiles": canonical_smiles,
                    "target": row["target"],
                    "split": row.get("split", ""),
                }
            )

    accepted_df = pd.DataFrame(accepted_rows, columns=ARTIFACT_COLUMNS).drop_duplicates(ignore_index=True)
    rejected_df = pd.DataFrame(rejected_rows, columns=REJECTED_ROW_COLUMNS)
    problem_df = pd.DataFrame(problem_rows, columns=PROBLEM_COLUMNS)
    return accepted_df, rejected_df, problem_df


def _assign_deterministic_scaffold_splits(
    frame: pd.DataFrame, config: EndpointConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assign complete scaffold groups deterministically to 80/10/10 splits."""

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    problems: list[dict[str, Any]] = []
    for source_row, row in frame.iterrows():
        try:
            result = safe_murcko_scaffold(str(row["canonical_smiles"]))
            scaffold_key = result.scaffold or f"ACYCLIC::{row['canonical_smiles']}"
            record = row.to_dict()
            record["_scaffold_key"] = scaffold_key
            accepted.append(record)
        except Exception as exc:  # Preserve the molecule; reject only from preparation output.
            rejected.append({
                "molecule_id": row.get("molecule_id", ""), "smiles": row.get("smiles", ""),
                "target": row.get("target", ""), "split": "", "rejection_reason": "scaffold_generation_failed",
            })
            problems.append(_problem_record(
                int(source_row), row.to_dict(), config, "scaffold_assignment", exc
            ))
    accepted_df = pd.DataFrame(accepted)
    if not accepted_df.empty:
        groups = [
            (key, sorted(group.index.tolist()))
            for key, group in accepted_df.groupby("_scaffold_key", sort=True)
        ]
        groups.sort(key=lambda item: (-len(item[1]), item[0]))
        targets = {name: DEFAULT_SPLIT_FRACTIONS[name] * len(accepted_df) for name in DEFAULT_SPLIT_FRACTIONS}
        counts = {name: 0 for name in DEFAULT_SPLIT_FRACTIONS}
        assignments: dict[int, str] = {}
        for _, indices in groups:
            split = max(
                DEFAULT_SPLIT_FRACTIONS,
                key=lambda name: (targets[name] - counts[name], -list(DEFAULT_SPLIT_FRACTIONS).index(name)),
            )
            for index in indices:
                assignments[index] = split
            counts[split] += len(indices)
        accepted_df["split"] = [assignments[index] for index in accepted_df.index]
        accepted_df = accepted_df[ARTIFACT_COLUMNS].reset_index(drop=True)
    else:
        accepted_df = pd.DataFrame(columns=ARTIFACT_COLUMNS)
    return (
        accepted_df,
        pd.DataFrame(rejected, columns=REJECTED_ROW_COLUMNS),
        pd.DataFrame(problems, columns=PROBLEM_COLUMNS),
    )


def _problem_record(
    source_row: int, row: dict[str, Any], config: EndpointConfig,
    failure_stage: str, exc: Exception,
) -> dict[str, Any]:
    return {
        "source_row": int(source_row), "molecule_id": row.get("molecule_id", ""),
        "original_smiles": row.get("smiles", ""), "endpoint": config.endpoint_id,
        "failure_stage": failure_stage, "exception_category": type(exc).__name__,
        "error_message": str(exc),
    }


def _write_split_csvs(df: pd.DataFrame, output_dir: Path) -> None:
    split_to_file = {
        "train": "train.csv",
        "validation": "valid.csv",
        "test": "test.csv",
    }
    for split_name, file_name in split_to_file.items():
        split_df = df[df["split"] == split_name] if not df.empty else pd.DataFrame(columns=ARTIFACT_COLUMNS)
        _atomic_write_csv(split_df, output_dir / file_name)


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


def _build_raw_summary(df: pd.DataFrame, config: EndpointConfig) -> dict[str, Any]:
    """Summarize unsplit normalized records saved by the TDC downloader."""
    summary: dict[str, Any] = {
        "endpoint_id": config.endpoint_id, "tdc_name": config.tdc_name,
        "task_type": config.task_type, "n_rows": int(len(df)),
        "n_accepted_rows": int(len(df)), "n_rejected_rows": 0,
        "n_unique_smiles": int(df["smiles"].nunique()), "split_status": "unsplit",
    }
    targets = df["target"].dropna()
    if config.task_type == "binary_classification":
        counts = targets.astype(int).value_counts().sort_index()
        summary["class_counts"] = {str(label): int(count) for label, count in counts.items()}
    else:
        numeric = pd.to_numeric(targets, errors="coerce")
        summary["target_min"] = float(numeric.min())
        summary["target_max"] = float(numeric.max())
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
            "problematic_molecules": "problematic_molecules.csv",
        },
    }
