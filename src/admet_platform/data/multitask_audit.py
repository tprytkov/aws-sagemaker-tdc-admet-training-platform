"""Exact-structure and scaffold leakage auditing for multi-task ADMET data."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold

from admet_platform.data.multitask import EndpointDatasetSplits, MultiTaskConfig


AUDIT_ARTIFACTS = {
    "summary": "audit_summary.json",
    "datasets": "dataset_summary.csv",
    "exact_overlaps": "exact_smiles_overlaps.csv",
    "scaffold_overlaps": "scaffold_overlaps.csv",
    "duplicates": "duplicates.csv",
    "conflicts": "conflicting_labels.csv",
    "invalid": "invalid_molecules.csv",
    "violations": "leakage_violations.csv",
}


class MultiTaskAuditError(ValueError):
    """Raised when configured leakage-safe training requirements are violated."""


@dataclass(frozen=True)
class AuditResult:
    """Machine-readable audit result plus tabular details."""

    summary: dict[str, Any]
    tables: Mapping[str, pd.DataFrame]


def audit_multitask_splits(
    config: MultiTaskConfig,
    datasets: Mapping[str, EndpointDatasetSplits],
) -> AuditResult:
    """Audit prepared endpoint splits without modifying official split files."""

    records, invalid_rows, dataset_rows = _normalize_records(datasets)
    valid = pd.DataFrame(records)
    invalid = pd.DataFrame(
        invalid_rows,
        columns=["task_name", "endpoint_id", "split", "row_index", "molecule_id", "canonical_smiles", "reason"],
    )
    dataset_summary = pd.DataFrame(dataset_rows)
    duplicates = _duplicate_rows(valid)
    conflicts = _conflicting_labels(valid)
    exact_overlaps = _overlap_rows(valid, key="normalized_canonical_smiles", value_name="canonical_smiles")
    scaffold_overlaps = _overlap_rows(valid, key="scaffold_key", value_name="scaffold")
    violations = _build_violations(config, exact_overlaps, scaffold_overlaps, duplicates, conflicts, invalid)

    summary = {
        "schema_version": config.schema_version,
        "run_name": config.run_name,
        "split_track": config.split_track,
        "status": "failed" if not violations.empty else "passed",
        "leakage_safe_for_training": bool(violations.empty),
        "task_names": list(config.tasks),
        "endpoint_ids": [task.endpoint_id for task in config.tasks.values()],
        "counts": {
            "input_rows": int(dataset_summary["row_count"].sum()) if not dataset_summary.empty else 0,
            "invalid_molecules": int(len(invalid)),
            "duplicate_groups": int(len(duplicates)),
            "conflicting_label_groups": int(len(conflicts)),
            "exact_overlap_groups": int(len(exact_overlaps)),
            "scaffold_overlap_groups": int(len(scaffold_overlaps)),
            "blocking_violations": int(len(violations)),
        },
        "rules": {
            "enforce_exact_smiles_exclusion": config.audit.enforce_exact_smiles_exclusion,
            "enforce_scaffold_exclusion": config.audit.enforce_scaffold_exclusion,
            "fail_on_invalid_molecules": config.audit.fail_on_invalid_molecules,
            "fail_on_conflicting_labels": config.audit.fail_on_conflicting_labels,
            "fail_on_duplicates": config.audit.fail_on_duplicates,
        },
        "rdkit_version": getattr(rdBase, "rdkitVersion", None),
        "notes": [
            "Official endpoint split files were audited but not modified.",
            "Acyclic molecules use their canonical SMILES as the scaffold key so all acyclic molecules are not grouped together.",
            "Coordinated multi-task outputs must be written separately from official TDC benchmark splits.",
        ],
        "artifacts": dict(AUDIT_ARTIFACTS),
    }
    return AuditResult(
        summary=summary,
        tables={
            "datasets": dataset_summary,
            "exact_overlaps": exact_overlaps,
            "scaffold_overlaps": scaffold_overlaps,
            "duplicates": duplicates,
            "conflicts": conflicts,
            "invalid": invalid,
            "violations": violations,
        },
    )


def write_audit_outputs(result: AuditResult, output_dir: str | Path) -> dict[str, str]:
    """Write stable JSON/CSV audit artifacts and return their paths."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {name: destination / file_name for name, file_name in AUDIT_ARTIFACTS.items()}
    paths["summary"].write_text(json.dumps(result.summary, indent=2) + "\n", encoding="utf-8")
    for table_name, frame in result.tables.items():
        frame.to_csv(paths[table_name], index=False)
    return {name: str(path) for name, path in paths.items()}


def require_leakage_safe(result: AuditResult) -> None:
    """Fail clearly after artifacts have been generated when rules are violated."""

    if not result.summary["leakage_safe_for_training"]:
        count = result.summary["counts"]["blocking_violations"]
        raise MultiTaskAuditError(
            f"Multi-task leakage audit failed with {count} blocking violation(s). "
            "Review leakage_violations.csv before training."
        )


def _normalize_records(
    datasets: Mapping[str, EndpointDatasetSplits],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for task_name, endpoint_splits in datasets.items():
        for split, frame in endpoint_splits.by_name().items():
            file_path = endpoint_splits.paths[split]
            summaries.append(
                {
                    "task_name": task_name,
                    "endpoint_id": endpoint_splits.endpoint.endpoint_id,
                    "split": split,
                    "row_count": int(len(frame)),
                    "file_sha256": _sha256(file_path),
                }
            )
            for row_index, row in frame.iterrows():
                supplied = "" if pd.isna(row["canonical_smiles"]) else str(row["canonical_smiles"]).strip()
                molecule = Chem.MolFromSmiles(supplied) if supplied else None
                if molecule is None:
                    invalid.append(
                        {
                            "task_name": task_name,
                            "endpoint_id": endpoint_splits.endpoint.endpoint_id,
                            "split": split,
                            "row_index": int(row_index),
                            "molecule_id": row.get("molecule_id", ""),
                            "canonical_smiles": supplied,
                            "reason": "invalid_or_missing_canonical_smiles",
                        }
                    )
                    continue
                canonical = Chem.MolToSmiles(molecule, canonical=True)
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
                scaffold_key = scaffold if scaffold else f"ACYCLIC::{canonical}"
                records.append(
                    {
                        "task_name": task_name,
                        "endpoint_id": endpoint_splits.endpoint.endpoint_id,
                        "split": split,
                        "row_index": int(row_index),
                        "molecule_id": row.get("molecule_id", ""),
                        "canonical_smiles": supplied,
                        "normalized_canonical_smiles": canonical,
                        "scaffold_key": scaffold_key,
                        "scaffold": scaffold,
                        "target": int(row["target"]),
                    }
                )
    return records, invalid, summaries


def _duplicate_rows(valid: pd.DataFrame) -> pd.DataFrame:
    columns = ["task_name", "endpoint_id", "split", "canonical_smiles", "count", "molecule_ids"]
    if valid.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    grouped = valid.groupby(["task_name", "endpoint_id", "split", "normalized_canonical_smiles"], sort=True)
    for (task, endpoint, split, canonical), group in grouped:
        if len(group) > 1:
            rows.append(
                {
                    "task_name": task,
                    "endpoint_id": endpoint,
                    "split": split,
                    "canonical_smiles": canonical,
                    "count": int(len(group)),
                    "molecule_ids": "|".join(sorted(map(str, group["molecule_id"]))),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _conflicting_labels(valid: pd.DataFrame) -> pd.DataFrame:
    columns = ["task_name", "endpoint_id", "canonical_smiles", "labels", "splits", "row_count"]
    if valid.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for (task, endpoint, canonical), group in valid.groupby(
        ["task_name", "endpoint_id", "normalized_canonical_smiles"], sort=True
    ):
        labels = sorted(set(group["target"]))
        if len(labels) > 1:
            rows.append(
                {
                    "task_name": task,
                    "endpoint_id": endpoint,
                    "canonical_smiles": canonical,
                    "labels": "|".join(map(str, labels)),
                    "splits": "|".join(sorted(set(group["split"]))),
                    "row_count": int(len(group)),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _overlap_rows(valid: pd.DataFrame, key: str, value_name: str) -> pd.DataFrame:
    columns = [
        value_name,
        "left_task",
        "left_endpoint",
        "left_split",
        "right_task",
        "right_endpoint",
        "right_split",
        "cross_task",
        "train_vs_heldout",
    ]
    if valid.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for value, group in valid.groupby(key, sort=True):
        locations = sorted(set(zip(group["task_name"], group["endpoint_id"], group["split"])))
        for left_index, left in enumerate(locations):
            for right in locations[left_index + 1 :]:
                if left == right:
                    continue
                train_vs_heldout = _train_vs_heldout(left[2], right[2])
                rows.append(
                    {
                        value_name: value,
                        "left_task": left[0],
                        "left_endpoint": left[1],
                        "left_split": left[2],
                        "right_task": right[0],
                        "right_endpoint": right[1],
                        "right_split": right[2],
                        "cross_task": bool(left[0] != right[0]),
                        "train_vs_heldout": train_vs_heldout,
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def _train_vs_heldout(left_split: str, right_split: str) -> bool:
    heldout = {"validation", "test"}
    return (left_split == "train" and right_split in heldout) or (
        right_split == "train" and left_split in heldout
    )


def _build_violations(
    config: MultiTaskConfig,
    exact: pd.DataFrame,
    scaffolds: pd.DataFrame,
    duplicates: pd.DataFrame,
    conflicts: pd.DataFrame,
    invalid: pd.DataFrame,
) -> pd.DataFrame:
    columns = ["violation_type", "count", "message"]
    rows: list[dict[str, Any]] = []
    exact_blocking = exact[exact["train_vs_heldout"]] if not exact.empty else exact
    scaffold_blocking = scaffolds[scaffolds["train_vs_heldout"]] if not scaffolds.empty else scaffolds
    checks: Iterable[tuple[bool, str, pd.DataFrame, str]] = (
        (
            config.audit.enforce_exact_smiles_exclusion,
            "exact_smiles_train_heldout_overlap",
            exact_blocking,
            "Canonical SMILES occurs in a training split and a validation/test split.",
        ),
        (
            config.audit.enforce_scaffold_exclusion,
            "scaffold_train_heldout_overlap",
            scaffold_blocking,
            "Murcko scaffold occurs in a training split and a validation/test split.",
        ),
        (
            config.audit.fail_on_duplicates,
            "duplicate_molecules",
            duplicates,
            "Duplicate canonical molecules occur within an endpoint split.",
        ),
        (
            config.audit.fail_on_conflicting_labels,
            "conflicting_labels",
            conflicts,
            "The same endpoint molecule has conflicting binary labels.",
        ),
        (
            config.audit.fail_on_invalid_molecules,
            "invalid_molecules",
            invalid,
            "Invalid or missing canonical SMILES were found.",
        ),
    )
    for enabled, violation_type, frame, message in checks:
        if enabled and not frame.empty:
            rows.append({"violation_type": violation_type, "count": int(len(frame)), "message": message})
    return pd.DataFrame(rows, columns=columns)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

