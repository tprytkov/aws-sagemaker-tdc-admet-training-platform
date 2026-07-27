"""Coordinated unsplit-source preparation for the ten-endpoint classification track."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from rdkit import Chem, rdBase

from admet_platform.config import _load_yaml_mapping
from admet_platform.data.coordinated_multitask import (
    CONFLICT_COLUMNS,
    DEDUPLICATION_COLUMNS,
    SCAFFOLD_ASSIGNMENT_COLUMNS,
    _assign_global_scaffolds,
    _quarantine_conflicts_and_collapse_duplicates,
    _validate_coordinated_records,
    _validate_split_fractions,
    _write_csv,
    _write_split_files,
)
from admet_platform.data.multitask import (
    REQUIRED_SPLITS,
    MultiTaskConfig,
    load_endpoint_datasets,
)
from admet_platform.data.multitask_audit import (
    AuditResult,
    audit_multitask_splits,
    require_leakage_safe,
    write_audit_outputs,
)
from admet_platform.data.scaffolds import safe_murcko_scaffold


EXPECTED_ENDPOINTS = {
    "hia_hou": "HIA_Hou",
    "pgp_broccatelli": "Pgp_Broccatelli",
    "bbb_martins": "BBB_Martins",
    "cyp1a2_veith": "CYP1A2_Veith",
    "cyp2c19_veith": "CYP2C19_Veith",
    "cyp2c9_veith": "CYP2C9_Veith",
    "cyp2d6_veith": "CYP2D6_Veith",
    "cyp3a4_veith": "CYP3A4_Veith",
    "herg_karim": "hERG_Karim",
    "ames": "AMES",
}
DEFAULT_SPLIT_FRACTIONS = {"train": 0.8, "validation": 0.1, "test": 0.1}
SOURCE_METHODS = {"normalized_unsplit_csv", "concatenate_prepared_splits"}


@dataclass(frozen=True)
class ExpandedSplitResult:
    """Generated ten-endpoint split metadata and blocking audit result."""

    output_root: Path
    manifest: Mapping[str, Any]
    audit: AuditResult


def build_expanded_classification_splits(
    config: MultiTaskConfig,
    source_config_path: str | Path,
    output_root: str | Path | None = None,
    *,
    seed: int = 42,
    split_fractions: Mapping[str, float] | None = None,
) -> ExpandedSplitResult:
    """Build the separate ten-endpoint globally coordinated split track."""

    _validate_expanded_config(config)
    fractions = _validate_split_fractions(split_fractions or DEFAULT_SPLIT_FRACTIONS)
    source_config, source_config_hash = _load_source_config(source_config_path, config)
    destination = Path(output_root).resolve() if output_root else config.prepared_root
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(
            f"Expanded coordinated output root already contains files: {destination}. "
            "Use a new or empty output root."
        )

    records, invalid, source_metadata = _load_and_normalize_sources(config, source_config)
    if invalid:
        sample = invalid[0]
        raise ValueError(
            f"Cannot coordinate splits with {len(invalid)} invalid source molecule(s); "
            f"first failure is {sample['endpoint_id']}/{sample['molecule_id']}: "
            f"{sample['reason']}"
        )

    retained, conflicts, deduplications = _quarantine_conflicts_and_collapse_duplicates(records)
    assignments, scaffold_assignments = _assign_global_scaffolds(
        retained,
        tuple(config.tasks),
        fractions,
        seed,
    )
    retained = retained.copy()
    retained["split"] = retained["scaffold_key"].map(assignments)
    _validate_coordinated_records(retained, tuple(config.tasks))

    destination.mkdir(parents=True, exist_ok=True)
    output_hashes = _write_split_files(retained, config, destination)
    _write_csv(conflicts, destination / "quarantined_conflicts.csv", CONFLICT_COLUMNS)
    _write_csv(
        deduplications,
        destination / "deduplication_provenance.csv",
        DEDUPLICATION_COLUMNS,
    )
    _write_csv(
        scaffold_assignments,
        destination / "global_scaffold_assignments.csv",
        SCAFFOLD_ASSIGNMENT_COLUMNS,
    )

    output_datasets = load_endpoint_datasets(config, prepared_root=destination)
    audit = audit_multitask_splits(config, output_datasets)
    write_audit_outputs(audit, destination / "audit")
    require_leakage_safe(audit)

    manifest = _build_manifest(
        config=config,
        source_config_hash=source_config_hash,
        source_metadata=source_metadata,
        records=records,
        retained=retained,
        conflicts=conflicts,
        deduplications=deduplications,
        scaffold_assignments=scaffold_assignments,
        fractions=fractions,
        seed=seed,
        output_hashes=output_hashes,
        audit=audit,
    )
    manifest_path = destination / "coordinated_split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_endpoint_summary_csv(manifest, destination / "endpoint_split_summary.csv")
    _write_human_report(manifest, destination / "coordinated_split_report.md")
    return ExpandedSplitResult(destination, manifest, audit)


def _validate_expanded_config(config: MultiTaskConfig) -> None:
    configured = {task: endpoint.tdc_name for task, endpoint in config.tasks.items()}
    if configured != EXPECTED_ENDPOINTS:
        raise ValueError(
            "Expanded classification config must define exactly the frozen ten endpoints."
        )
    if config.split_track != "coordinated_multitask":
        raise ValueError("Expanded classification requires split_track: coordinated_multitask.")
    if any(endpoint.task_type != "binary_classification" for endpoint in config.tasks.values()):
        raise ValueError("Every expanded classification endpoint must be binary classification.")


def _load_source_config(
    path: str | Path,
    config: MultiTaskConfig,
) -> tuple[dict[str, dict[str, Any]], str]:
    source_path = Path(path).resolve()
    raw = _load_yaml_mapping(
        source_path.read_text(encoding="utf-8"),
        source=str(source_path),
    )
    if raw.get("schema_version") != "1.0.0" or not isinstance(raw.get("endpoints"), dict):
        raise ValueError("Expanded source config requires schema_version 1.0.0 and endpoints.")
    endpoints = raw["endpoints"]
    if set(endpoints) != set(config.tasks):
        missing = sorted(set(config.tasks) - set(endpoints))
        extra = sorted(set(endpoints) - set(config.tasks))
        raise ValueError(f"Expanded source endpoints mismatch; missing={missing}, extra={extra}.")

    resolved: dict[str, dict[str, Any]] = {}
    for task_name, value in endpoints.items():
        if not isinstance(value, dict):
            raise ValueError(f"Source entry {task_name} must be a mapping.")
        method = value.get("method")
        paths = value.get("paths")
        if method not in SOURCE_METHODS:
            raise ValueError(f"Source entry {task_name} has unsupported method '{method}'.")
        if not isinstance(paths, list) or not paths or not all(isinstance(item, str) for item in paths):
            raise ValueError(f"Source entry {task_name} requires a non-empty paths list.")
        if method == "normalized_unsplit_csv" and len(paths) != 1:
            raise ValueError(f"Source entry {task_name} normalized CSV requires exactly one path.")
        if method == "concatenate_prepared_splits" and len(paths) != len(REQUIRED_SPLITS):
            raise ValueError(
                f"Source entry {task_name} prepared reconstruction requires three paths."
            )
        resolved[task_name] = {
            "method": method,
            "configured_paths": list(paths),
            "paths": [(source_path.parent / item).resolve() for item in paths],
        }
    return resolved, _sha256(source_path)


def _load_and_normalize_sources(
    config: MultiTaskConfig,
    sources: Mapping[str, Mapping[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    source_order = 0
    for task_name, endpoint in config.tasks.items():
        source = sources[task_name]
        frames = []
        file_metadata = []
        for configured_path, path in zip(
            source["configured_paths"], source["paths"], strict=True
        ):
            if not path.is_file():
                raise FileNotFoundError(f"Missing expanded source file: {configured_path}")
            frame = pd.read_csv(path)
            frames.append(frame)
            file_metadata.append(
                {
                    "path": configured_path,
                    "sha256": _sha256(path),
                    "row_count": int(len(frame)),
                    "columns": list(frame.columns),
                }
            )
        combined = pd.concat(frames, ignore_index=True)
        required = {"molecule_id", "target"}
        if not required.issubset(combined.columns):
            raise ValueError(
                f"Source {task_name} is missing columns: {sorted(required - set(combined.columns))}."
            )
        smiles_column = "smiles" if "smiles" in combined.columns else "canonical_smiles"
        if smiles_column not in combined.columns:
            raise ValueError(f"Source {task_name} has neither smiles nor canonical_smiles.")
        numeric_targets = pd.to_numeric(combined["target"], errors="coerce")
        if numeric_targets.isna().any() or not numeric_targets.isin([0, 1]).all():
            raise ValueError(f"Source {task_name} contains missing or non-binary targets.")

        for row_index, row in combined.iterrows():
            supplied = str(row[smiles_column]).strip()
            molecule = Chem.MolFromSmiles(supplied)
            molecule_id = str(row["molecule_id"])
            if molecule is None:
                invalid.append(
                    {
                        "task_name": task_name,
                        "endpoint_id": endpoint.endpoint_id,
                        "row_index": int(row_index),
                        "molecule_id": molecule_id,
                        "reason": "invalid_smiles",
                    }
                )
                source_order += 1
                continue
            canonical = Chem.MolToSmiles(molecule, canonical=True)
            try:
                scaffold = safe_murcko_scaffold(molecule).scaffold
            except (RuntimeError, ValueError) as exc:
                invalid.append(
                    {
                        "task_name": task_name,
                        "endpoint_id": endpoint.endpoint_id,
                        "row_index": int(row_index),
                        "molecule_id": molecule_id,
                        "reason": f"scaffold_generation_failed:{type(exc).__name__}:{exc}",
                    }
                )
                source_order += 1
                continue
            rows.append(
                {
                    "task_name": task_name,
                    "endpoint_id": endpoint.endpoint_id,
                    "molecule_id": molecule_id,
                    "smiles": supplied,
                    "canonical_smiles": canonical,
                    "target": int(numeric_targets.iloc[row_index]),
                    "source_split": _source_partition(
                        source["method"], row_index, frames
                    ),
                    "source_row_index": int(row_index),
                    "source_order": source_order,
                    "scaffold": scaffold,
                    "scaffold_key": scaffold if scaffold else f"ACYCLIC::{canonical}",
                }
            )
            source_order += 1
        metadata[task_name] = {
            "endpoint_id": endpoint.endpoint_id,
            "tdc_name": endpoint.tdc_name,
            "method": source["method"],
            "files": file_metadata,
            "combined_row_count": int(len(combined)),
            "old_split_assignment_removed": source["method"]
            == "concatenate_prepared_splits",
        }
    return pd.DataFrame(rows), invalid, metadata


def _source_partition(method: str, row_index: int, frames: list[pd.DataFrame]) -> str:
    if method == "normalized_unsplit_csv":
        return "unsplit"
    cursor = 0
    for split, frame in zip(REQUIRED_SPLITS, frames, strict=True):
        cursor += len(frame)
        if row_index < cursor:
            return split
    raise AssertionError("Reconstructed source row was outside its input partitions.")


def _build_manifest(
    *,
    config: MultiTaskConfig,
    source_config_hash: str,
    source_metadata: Mapping[str, Any],
    records: pd.DataFrame,
    retained: pd.DataFrame,
    conflicts: pd.DataFrame,
    deduplications: pd.DataFrame,
    scaffold_assignments: pd.DataFrame,
    fractions: Mapping[str, float],
    seed: int,
    output_hashes: Mapping[str, str],
    audit: AuditResult,
) -> dict[str, Any]:
    endpoints: dict[str, Any] = {}
    for task_name, endpoint in config.tasks.items():
        input_rows = records[records["task_name"] == task_name]
        output_rows = retained[retained["task_name"] == task_name]
        endpoint_dedup = deduplications[deduplications["endpoint_id"] == endpoint.endpoint_id]
        endpoint_conflicts = conflicts[conflicts["endpoint_id"] == endpoint.endpoint_id]
        split_summaries = {}
        for split in REQUIRED_SPLITS:
            frame = output_rows[output_rows["split"] == split]
            counts = frame["target"].value_counts()
            split_summaries[split] = {
                "row_count": int(len(frame)),
                "class_counts": {
                    str(label): int(counts.get(label, 0)) for label in (0, 1)
                },
                "class_fractions": {
                    str(label): float(counts.get(label, 0) / len(frame))
                    for label in (0, 1)
                },
                "unique_canonical_molecules": int(frame["canonical_smiles"].nunique()),
                "unique_scaffolds": int(frame["scaffold_key"].nunique()),
                "output_csv_sha256": output_hashes[f"{task_name}/{split}"],
            }
        endpoints[task_name] = {
            **source_metadata[task_name],
            "rows_before_deduplication": int(len(input_rows)),
            "rows_after_deduplication": int(len(output_rows)),
            "duplicate_groups_collapsed": int(len(endpoint_dedup)),
            "duplicate_rows_removed": int(endpoint_dedup["removed_row_count"].sum()),
            "conflicting_label_groups": int(len(endpoint_conflicts)),
            "conflicting_label_rows_quarantined": int(
                endpoint_conflicts["row_count"].sum()
            ),
            "splits": split_summaries,
        }

    molecule_groups = retained.groupby("canonical_smiles")
    scaffold_groups = retained.groupby("scaffold_key")
    shared_molecules = molecule_groups["task_name"].nunique()
    shared_scaffolds = scaffold_groups["task_name"].nunique()
    exact_leakage = int((molecule_groups["split"].nunique() > 1).sum())
    scaffold_leakage = int((scaffold_groups["split"].nunique() > 1).sum())

    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "split_track": "expanded_ten_endpoint_coordinated_multitask",
        "configuration_sha256": _sha256(config.source_path),
        "source_configuration_sha256": source_config_hash,
        "random_seed": seed,
        "split_ratios": dict(fractions),
        "rdkit_version": getattr(rdBase, "rdkitVersion", None),
        "total_rows_before_deduplication": int(len(records)),
        "total_rows_after_deduplication": int(len(retained)),
        "unique_canonical_molecules": int(retained["canonical_smiles"].nunique()),
        "unique_scaffolds": int(retained["scaffold_key"].nunique()),
        "exact_molecule_overlap_count_between_splits": exact_leakage,
        "scaffold_overlap_count_between_splits": scaffold_leakage,
        "cross_endpoint_shared_molecule_count": int((shared_molecules > 1).sum()),
        "cross_endpoint_shared_scaffold_count": int((shared_scaffolds > 1).sum()),
        "global_split_rows": {
            split: int((retained["split"] == split).sum()) for split in REQUIRED_SPLITS
        },
        "global_scaffold_groups": {
            "total": int(len(scaffold_assignments)),
            "by_split": {
                split: int((scaffold_assignments["split"] == split).sum())
                for split in REQUIRED_SPLITS
            },
        },
        "deduplication": {
            "groups_collapsed": int(len(deduplications)),
            "rows_removed": int(deduplications["removed_row_count"].sum()),
        },
        "conflicts": {
            "groups_quarantined": int(len(conflicts)),
            "rows_quarantined": int(conflicts["row_count"].sum()),
        },
        "endpoints": endpoints,
        "audit_summary": audit.summary,
        "artifacts": {
            "deduplication": "deduplication_provenance.csv",
            "conflicts": "quarantined_conflicts.csv",
            "scaffold_assignments": "global_scaffold_assignments.csv",
            "endpoint_split_summary": "endpoint_split_summary.csv",
            "audit_directory": "audit",
        },
    }
    identity_payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    manifest["split_manifest_id"] = hashlib.sha256(identity_payload.encode("utf-8")).hexdigest()
    return manifest


def _write_endpoint_summary_csv(manifest: Mapping[str, Any], path: Path) -> None:
    rows = []
    for task_name, endpoint in manifest["endpoints"].items():
        for split, summary in endpoint["splits"].items():
            rows.append(
                {
                    "task_name": task_name,
                    "endpoint_id": endpoint["endpoint_id"],
                    "tdc_name": endpoint["tdc_name"],
                    "split": split,
                    **summary,
                    "class_0_count": summary["class_counts"]["0"],
                    "class_1_count": summary["class_counts"]["1"],
                    "class_0_fraction": summary["class_fractions"]["0"],
                    "class_1_fraction": summary["class_fractions"]["1"],
                    "duplicate_rows_removed": endpoint["duplicate_rows_removed"],
                    "conflicting_label_groups": endpoint["conflicting_label_groups"],
                }
            )
    frame = pd.DataFrame(rows).drop(columns=["class_counts", "class_fractions"])
    frame.to_csv(path, index=False, lineterminator="\n")


def _write_human_report(manifest: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Expanded Classification Coordinated Split Report",
        "",
        f"- Manifest ID: `{manifest['split_manifest_id']}`",
        f"- Seed: {manifest['random_seed']}",
        f"- Rows before/after deduplication: "
        f"{manifest['total_rows_before_deduplication']}/"
        f"{manifest['total_rows_after_deduplication']}",
        f"- Exact-molecule leakage groups: "
        f"{manifest['exact_molecule_overlap_count_between_splits']}",
        f"- Scaffold leakage groups: {manifest['scaffold_overlap_count_between_splits']}",
        "",
        "| Endpoint | Before | After | Train | Validation | Test | Duplicates removed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task_name, endpoint in manifest["endpoints"].items():
        lines.append(
            f"| {task_name} | {endpoint['rows_before_deduplication']} | "
            f"{endpoint['rows_after_deduplication']} | "
            f"{endpoint['splits']['train']['row_count']} | "
            f"{endpoint['splits']['validation']['row_count']} | "
            f"{endpoint['splits']['test']['row_count']} | "
            f"{endpoint['duplicate_rows_removed']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "EXPECTED_ENDPOINTS",
    "ExpandedSplitResult",
    "build_expanded_classification_splits",
]
