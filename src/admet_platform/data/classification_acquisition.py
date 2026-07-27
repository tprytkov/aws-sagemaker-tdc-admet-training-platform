"""Audited, unsplit acquisition for candidate binary TDC endpoints."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem

from admet_platform.config import EndpointConfig, load_endpoint_config
from admet_platform.data.tdc_loader import load_tdc_data


VERIFIED_PYTDC_VERSION = "0.3.9"
NORMALIZED_COLUMNS = ["molecule_id", "smiles", "target"]
DETAIL_COLUMNS = [
    "source_row",
    "molecule_id",
    "smiles",
    "canonical_smiles",
    "target",
    "reason",
]


def acquire_and_audit_binary_tdc_dataset(
    config_path: str | Path,
    output_directory: str | Path,
    *,
    expected_pytdc_version: str = VERIFIED_PYTDC_VERSION,
) -> dict[str, Any]:
    """Load an unsplit endpoint and retain every row while writing audit artifacts."""

    config = load_endpoint_config(config_path)
    _validate_candidate_config(config)
    pytdc_version = importlib.metadata.version("PyTDC")
    if pytdc_version != expected_pytdc_version:
        raise RuntimeError(
            f"Expected PyTDC=={expected_pytdc_version}, found PyTDC=={pytdc_version}."
        )
    registry_name = _verify_adme_registry_name(config.tdc_name)

    raw = load_tdc_data(config)
    normalized = _normalize_without_dropping(raw, config)
    details = _canonicalize_and_classify(normalized)

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    normalized_path = destination / "normalized.csv"
    normalized.to_csv(normalized_path, index=False, lineterminator="\n")

    invalid_smiles = details.loc[details["_invalid_smiles"]]
    missing_labels = details.loc[details["_missing_label"]]
    invalid_labels = details.loc[details["_invalid_label"]]
    valid = details.loc[
        ~(details["_invalid_smiles"] | details["_missing_label"] | details["_invalid_label"])
    ]

    conflict_keys = (
        valid.groupby("canonical_smiles", sort=True)["target"]
        .nunique()
        .loc[lambda values: values > 1]
        .index
    )
    conflicts = valid.loc[valid["canonical_smiles"].isin(conflict_keys)].copy()
    conflicts["reason"] = "conflicting_labels_for_canonical_structure"

    duplicate_mask = valid.duplicated(["canonical_smiles", "target"], keep=False)
    duplicates = valid.loc[duplicate_mask].copy()
    duplicates["reason"] = "repeated_canonical_structure_and_label"

    _write_detail_csv(destination / "quarantined_invalid_smiles.csv", invalid_smiles)
    _write_detail_csv(destination / "quarantined_missing_labels.csv", missing_labels)
    _write_detail_csv(destination / "quarantined_invalid_labels.csv", invalid_labels)
    _write_detail_csv(destination / "quarantined_conflicting_labels.csv", conflicts)
    _write_detail_csv(destination / "exact_duplicates.csv", duplicates)

    valid_labels = pd.to_numeric(
        normalized["target"], errors="coerce"
    ).loc[lambda values: values.isin([0, 1])]
    class_counts = {str(label): int((valid_labels == label).sum()) for label in (0, 1)}
    denominator = len(valid_labels)
    class_fractions = {
        label: (count / denominator if denominator else 0.0)
        for label, count in class_counts.items()
    }
    duplicate_groups = (
        valid.groupby(["canonical_smiles", "target"], sort=True)
        .size()
        .loc[lambda sizes: sizes > 1]
    )

    audit: dict[str, Any] = {
        "schema_version": 1,
        "endpoint_id": config.endpoint_id,
        "tdc_name": config.tdc_name,
        "tdc_registry_name": registry_name,
        "task_group": config.task_group,
        "task_type": config.task_type,
        "pytdc_version": pytdc_version,
        "label_semantics": config.problem_description,
        "raw_columns": [str(column) for column in raw.columns],
        "raw_row_count": int(len(raw)),
        "normalized_row_count": int(len(normalized)),
        "unique_raw_label_values": _json_label_values(raw, config),
        "class_counts": class_counts,
        "class_fractions": class_fractions,
        "missing_label_count": int(len(missing_labels)),
        "invalid_label_count": int(len(invalid_labels)),
        "invalid_smiles_count": int(len(invalid_smiles)),
        "exact_duplicate_count": int(duplicate_groups.sub(1).sum()),
        "exact_duplicate_group_count": int(len(duplicate_groups)),
        "conflicting_label_duplicate_count": int(len(conflict_keys)),
        "conflicting_label_duplicate_row_count": int(len(conflicts)),
        "normalized_csv_sha256": _sha256(normalized_path),
        "split_status": "unsplit",
        "rows_discarded": 0,
        "artifacts": {
            "normalized_csv": normalized_path.name,
            "invalid_smiles_quarantine": "quarantined_invalid_smiles.csv",
            "missing_labels_quarantine": "quarantined_missing_labels.csv",
            "invalid_labels_quarantine": "quarantined_invalid_labels.csv",
            "conflicting_labels_quarantine": "quarantined_conflicting_labels.csv",
            "exact_duplicates": "exact_duplicates.csv",
        },
    }
    (destination / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit


def _validate_candidate_config(config: EndpointConfig) -> None:
    if config.task_group != "ADME":
        raise ValueError("Candidate classification acquisition supports only TDC ADME endpoints.")
    if config.task_type != "binary_classification":
        raise ValueError("Candidate classification acquisition requires binary classification.")


def _verify_adme_registry_name(tdc_name: str) -> str:
    from tdc.metadata import adme_dataset_names

    requested_registry_name = tdc_name.lower()
    if requested_registry_name not in adme_dataset_names:
        raise ValueError(
            f"TDC dataset '{tdc_name}' is not registered in the PyTDC ADME task group."
        )
    return requested_registry_name


def _normalize_without_dropping(raw: pd.DataFrame, config: EndpointConfig) -> pd.DataFrame:
    smiles_column = _resolve_column(raw, [config.smiles_column, "Drug", "SMILES", "smiles"])
    target_column = _resolve_column(raw, [config.target_column, "Y", "target", "Label", "label"])
    identifier_column = _resolve_optional_column(
        raw, ["molecule_id", "Drug_ID", "Drug_IDs", "id", "ID", "Index"]
    )
    normalized = pd.DataFrame(
        {
            "molecule_id": (
                raw[identifier_column].astype("string")
                if identifier_column is not None
                else [
                    f"{config.endpoint_id}_raw_{index:06d}"
                    for index in range(len(raw))
                ]
            ),
            "smiles": raw[smiles_column].astype("string"),
            "target": pd.to_numeric(raw[target_column], errors="coerce"),
        }
    )
    normalized["molecule_id"] = normalized["molecule_id"].str.strip()
    normalized["smiles"] = normalized["smiles"].str.strip()
    return normalized[NORMALIZED_COLUMNS]


def _canonicalize_and_classify(normalized: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_row, row in normalized.iterrows():
        smiles = row["smiles"]
        target = row["target"]
        canonical = ""
        missing_label = bool(pd.isna(target))
        invalid_label = not missing_label and float(target) not in {0.0, 1.0}
        invalid_smiles = bool(pd.isna(smiles) or not str(smiles).strip())
        if pd.isna(smiles) or not str(smiles).strip():
            invalid_smiles = True
        else:
            molecule = Chem.MolFromSmiles(str(smiles))
            if molecule is None:
                invalid_smiles = True
            else:
                canonical = Chem.MolToSmiles(molecule, canonical=True)
        reasons = []
        if invalid_smiles:
            reasons.append("invalid_smiles")
        if missing_label:
            reasons.append("missing_label")
        if invalid_label:
            reasons.append("invalid_binary_label")
        rows.append(
            {
                "source_row": int(source_row),
                "molecule_id": row["molecule_id"],
                "smiles": smiles,
                "canonical_smiles": canonical,
                "target": target,
                "reason": ";".join(reasons),
                "_invalid_smiles": invalid_smiles,
                "_missing_label": missing_label,
                "_invalid_label": invalid_label,
            }
        )
    return pd.DataFrame(rows)


def _json_label_values(raw: pd.DataFrame, config: EndpointConfig) -> list[Any]:
    target_column = _resolve_column(raw, [config.target_column, "Y", "target", "Label", "label"])
    values: list[Any] = []
    for value in raw[target_column].drop_duplicates().tolist():
        if pd.isna(value):
            values.append(None)
        elif hasattr(value, "item"):
            values.append(value.item())
        else:
            values.append(value)
    return sorted(values, key=lambda value: (value is None, str(value)))


def _write_detail_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.reindex(columns=DETAIL_COLUMNS).to_csv(path, index=False, lineterminator="\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_column(frame: pd.DataFrame, candidates: list[str]) -> str:
    column = _resolve_optional_column(frame, candidates)
    if column is None:
        raise ValueError(f"TDC DataFrame is missing expected columns: {', '.join(candidates)}.")
    return column


def _resolve_optional_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((column for column in candidates if column in frame.columns), None)


__all__ = [
    "VERIFIED_PYTDC_VERSION",
    "acquire_and_audit_binary_tdc_dataset",
]
