"""Deterministic leakage-safe coordinated splits for multi-task ADMET data."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from rdkit import Chem, rdBase

from admet_platform.data.multitask import (
    REQUIRED_SPLITS,
    EndpointDatasetSplits,
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


DEFAULT_SPLIT_FRACTIONS = {"train": 0.8, "validation": 0.1, "test": 0.1}
SPLIT_FILE_NAMES = {"train": "train.csv", "validation": "valid.csv", "test": "test.csv"}
CONFLICT_COLUMNS = (
    "endpoint_id",
    "canonical_smiles",
    "labels",
    "source_splits",
    "molecule_ids",
    "row_count",
)
DEDUPLICATION_COLUMNS = (
    "endpoint_id",
    "canonical_smiles",
    "target",
    "kept_molecule_id",
    "kept_source_split",
    "removed_molecule_ids",
    "removed_source_splits",
    "input_row_count",
    "removed_row_count",
)
SCAFFOLD_ASSIGNMENT_COLUMNS = (
    "scaffold_key",
    "split",
    "row_count",
    "endpoints",
)


@dataclass(frozen=True)
class CoordinatedSplitResult:
    """Generated coordinated split metadata and its blocking audit result."""

    output_root: Path
    manifest: Mapping[str, Any]
    audit: AuditResult


def build_coordinated_multitask_splits(
    config: MultiTaskConfig,
    source_root: str | Path,
    output_root: str | Path | None = None,
    *,
    seed: int = 42,
    split_fractions: Mapping[str, float] | None = None,
) -> CoordinatedSplitResult:
    """Build a separate globally scaffold-grouped multi-task split track."""

    source = Path(source_root).resolve()
    destination = Path(output_root).resolve() if output_root is not None else config.prepared_root
    if (
        source == destination
        or destination.is_relative_to(source)
        or source.is_relative_to(destination)
    ):
        raise ValueError("Coordinated output and prepared source roots must be separate directories.")
    if config.split_track != "coordinated_multitask":
        raise ValueError("Coordinated splitting requires split_track: coordinated_multitask.")
    fractions = _validate_split_fractions(split_fractions or DEFAULT_SPLIT_FRACTIONS)
    datasets = load_endpoint_datasets(config, prepared_root=source)
    records, invalid = _normalize_source_records(datasets)
    if invalid:
        sample = invalid[0]
        raise ValueError(
            f"Cannot coordinate splits with {len(invalid)} invalid source molecule(s); "
            f"first failure is {sample['endpoint_id']}/{sample['molecule_id']}: {sample['reason']}"
        )

    retained, conflicts, deduplications = _quarantine_conflicts_and_collapse_duplicates(records)
    assignments, scaffold_assignments = _assign_global_scaffolds(
        retained, tuple(config.tasks), fractions, seed
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

    coordinated_datasets = load_endpoint_datasets(config, prepared_root=destination)
    audit = audit_multitask_splits(config, coordinated_datasets)
    write_audit_outputs(audit, destination / "audit")
    require_leakage_safe(audit)
    manifest = _build_manifest(
        config=config,
        datasets=datasets,
        retained=retained,
        conflicts=conflicts,
        deduplications=deduplications,
        scaffold_assignments=scaffold_assignments,
        fractions=fractions,
        seed=seed,
        output_hashes=output_hashes,
        audit=audit,
    )
    (destination / "coordinated_split_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return CoordinatedSplitResult(output_root=destination, manifest=manifest, audit=audit)


def _normalize_source_records(
    datasets: Mapping[str, EndpointDatasetSplits],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    source_order = 0
    for task_name, endpoint_splits in datasets.items():
        for source_split in REQUIRED_SPLITS:
            frame = endpoint_splits.by_name()[source_split]
            for row_index, row in frame.iterrows():
                supplied = str(row["canonical_smiles"]).strip()
                molecule = Chem.MolFromSmiles(supplied)
                if molecule is None:
                    invalid.append(
                        {
                            "task_name": task_name,
                            "endpoint_id": endpoint_splits.endpoint.endpoint_id,
                            "source_split": source_split,
                            "row_index": int(row_index),
                            "molecule_id": str(row.get("molecule_id", "")),
                            "canonical_smiles": supplied,
                            "reason": "invalid_canonical_smiles",
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
                            "endpoint_id": endpoint_splits.endpoint.endpoint_id,
                            "source_split": source_split,
                            "row_index": int(row_index),
                            "molecule_id": str(row.get("molecule_id", "")),
                            "canonical_smiles": canonical,
                            "reason": f"scaffold_generation_failed:{type(exc).__name__}:{exc}",
                        }
                    )
                    source_order += 1
                    continue
                rows.append(
                    {
                        "task_name": task_name,
                        "endpoint_id": endpoint_splits.endpoint.endpoint_id,
                        "molecule_id": str(row.get("molecule_id", "")),
                        "smiles": str(row.get("smiles", supplied)),
                        "canonical_smiles": canonical,
                        "target": int(row["target"]),
                        "source_split": source_split,
                        "source_row_index": int(row_index),
                        "source_order": source_order,
                        "scaffold": scaffold,
                        "scaffold_key": scaffold if scaffold else f"ACYCLIC::{canonical}",
                    }
                )
                source_order += 1
    return pd.DataFrame(rows), invalid


def _quarantine_conflicts_and_collapse_duplicates(
    records: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    retained_rows: list[pd.Series] = []
    conflict_rows: list[dict[str, Any]] = []
    deduplication_rows: list[dict[str, Any]] = []
    grouped = records.groupby(["endpoint_id", "canonical_smiles"], sort=True)
    for (endpoint_id, canonical), group in grouped:
        ordered = group.sort_values("source_order", kind="stable")
        labels = sorted(set(int(value) for value in ordered["target"]))
        if len(labels) > 1:
            conflict_rows.append(
                {
                    "endpoint_id": endpoint_id,
                    "canonical_smiles": canonical,
                    "labels": "|".join(map(str, labels)),
                    "source_splits": "|".join(sorted(set(ordered["source_split"]))),
                    "molecule_ids": "|".join(map(str, ordered["molecule_id"])),
                    "row_count": int(len(ordered)),
                }
            )
            continue
        representative = ordered.iloc[0]
        retained_rows.append(representative)
        if len(ordered) > 1:
            removed = ordered.iloc[1:]
            deduplication_rows.append(
                {
                    "endpoint_id": endpoint_id,
                    "canonical_smiles": canonical,
                    "target": labels[0],
                    "kept_molecule_id": str(representative["molecule_id"]),
                    "kept_source_split": str(representative["source_split"]),
                    "removed_molecule_ids": "|".join(map(str, removed["molecule_id"])),
                    "removed_source_splits": "|".join(map(str, removed["source_split"])),
                    "input_row_count": int(len(ordered)),
                    "removed_row_count": int(len(removed)),
                }
            )
    retained = pd.DataFrame(retained_rows).sort_values("source_order", kind="stable").reset_index(drop=True)
    return (
        retained,
        pd.DataFrame(conflict_rows, columns=CONFLICT_COLUMNS),
        pd.DataFrame(deduplication_rows, columns=DEDUPLICATION_COLUMNS),
    )


def _assign_global_scaffolds(
    records: pd.DataFrame,
    task_names: tuple[str, ...],
    fractions: Mapping[str, float],
    seed: int,
) -> tuple[dict[str, str], pd.DataFrame]:
    groups = _scaffold_groups(records)
    assignments: dict[str, str] = {}
    counts = _empty_assignment_counts()

    for split in ("validation", "test", "train"):
        required = {(task, label) for task in task_names for label in (0, 1)}
        while required:
            candidates = [group for group in groups if group["scaffold_key"] not in assignments]
            available = {
                requirement: [group for group in candidates if group["class_counts"].get(requirement, 0)]
                for requirement in required
            }
            impossible = [requirement for requirement, options in available.items() if not options]
            if impossible:
                formatted = ", ".join(f"{task}/class-{label}" for task, label in sorted(impossible))
                raise ValueError(
                    f"Cannot place both classes in split '{split}'; no unassigned scaffold remains for {formatted}."
                )
            rarest = min(
                required,
                key=lambda requirement: (
                    len(available[requirement]),
                    _seeded_rank(seed, split, requirement[0], str(requirement[1])),
                ),
            )
            chosen = min(
                available[rarest],
                key=lambda group: (
                    -sum(bool(group["class_counts"].get(item, 0)) for item in required),
                    group["row_count"],
                    _seeded_rank(seed, split, group["scaffold_key"]),
                ),
            )
            _record_assignment(chosen, split, assignments, counts)
            required = {
                requirement
                for requirement in required
                if counts["classes"][(split, *requirement)] == 0
            }

    targets = _assignment_targets(records, task_names, fractions)
    remaining = [group for group in groups if group["scaffold_key"] not in assignments]
    remaining.sort(
        key=lambda group: (
            -group["row_count"],
            _seeded_rank(seed, "group-order", group["scaffold_key"]),
        )
    )
    for group in remaining:
        split = min(
            REQUIRED_SPLITS,
            key=lambda candidate: (
                _assignment_score(group, candidate, counts, targets, task_names),
                _seeded_rank(seed, group["scaffold_key"], candidate),
            ),
        )
        _record_assignment(group, split, assignments, counts)

    rows = [
        {
            "scaffold_key": group["scaffold_key"],
            "split": assignments[group["scaffold_key"]],
            "row_count": group["row_count"],
            "endpoints": "|".join(sorted(group["endpoint_counts"])),
        }
        for group in groups
    ]
    return assignments, pd.DataFrame(rows, columns=SCAFFOLD_ASSIGNMENT_COLUMNS).sort_values(
        "scaffold_key", kind="stable"
    )


def _scaffold_groups(records: pd.DataFrame) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for scaffold_key, frame in records.groupby("scaffold_key", sort=True):
        groups.append(
            {
                "scaffold_key": scaffold_key,
                "row_count": int(len(frame)),
                "endpoint_counts": Counter(frame["task_name"]),
                "class_counts": Counter(
                    (str(row.task_name), int(row.target)) for row in frame.itertuples()
                ),
            }
        )
    return groups


def _empty_assignment_counts() -> dict[str, Counter[Any]]:
    return {
        "global": Counter({split: 0 for split in REQUIRED_SPLITS}),
        "endpoints": Counter(),
        "classes": Counter(),
    }


def _record_assignment(
    group: Mapping[str, Any],
    split: str,
    assignments: dict[str, str],
    counts: Mapping[str, Counter[Any]],
) -> None:
    assignments[str(group["scaffold_key"])] = split
    counts["global"][split] += int(group["row_count"])
    for task, count in group["endpoint_counts"].items():
        counts["endpoints"][(split, task)] += int(count)
    for (task, label), count in group["class_counts"].items():
        counts["classes"][(split, task, label)] += int(count)


def _assignment_targets(
    records: pd.DataFrame,
    task_names: tuple[str, ...],
    fractions: Mapping[str, float],
) -> dict[str, dict[Any, float]]:
    endpoint_totals = Counter(records["task_name"])
    class_totals = Counter((str(row.task_name), int(row.target)) for row in records.itertuples())
    return {
        "global": {split: len(records) * fractions[split] for split in REQUIRED_SPLITS},
        "endpoints": {
            (split, task): endpoint_totals[task] * fractions[split]
            for split in REQUIRED_SPLITS
            for task in task_names
        },
        "classes": {
            (split, task, label): class_totals[(task, label)] * fractions[split]
            for split in REQUIRED_SPLITS
            for task in task_names
            for label in (0, 1)
        },
    }


def _assignment_score(
    group: Mapping[str, Any],
    candidate: str,
    counts: Mapping[str, Counter[Any]],
    targets: Mapping[str, Mapping[Any, float]],
    task_names: tuple[str, ...],
) -> float:
    score = 0.0
    for split in REQUIRED_SPLITS:
        added = int(group["row_count"]) if split == candidate else 0
        score += _relative_square(counts["global"][split] + added, targets["global"][split])
        for task in task_names:
            endpoint_added = int(group["endpoint_counts"].get(task, 0)) if split == candidate else 0
            score += 2.0 * _relative_square(
                counts["endpoints"][(split, task)] + endpoint_added,
                targets["endpoints"][(split, task)],
            )
            for label in (0, 1):
                class_added = int(group["class_counts"].get((task, label), 0)) if split == candidate else 0
                score += _relative_square(
                    counts["classes"][(split, task, label)] + class_added,
                    targets["classes"][(split, task, label)],
                )
    return score


def _relative_square(actual: float, target: float) -> float:
    scale = max(target, 1.0)
    return ((actual - target) / scale) ** 2


def _seeded_rank(seed: int, *parts: str) -> str:
    value = "|".join((str(seed), *map(str, parts)))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_coordinated_records(records: pd.DataFrame, task_names: tuple[str, ...]) -> None:
    duplicate_count = int(records.duplicated(["endpoint_id", "canonical_smiles"]).sum())
    if duplicate_count:
        raise ValueError(f"Coordinated output still contains {duplicate_count} endpoint/canonical duplicate(s).")
    conflicts = records.groupby(["endpoint_id", "canonical_smiles"])["target"].nunique()
    if (conflicts > 1).any():
        raise ValueError("Coordinated output still contains within-endpoint conflicting labels.")
    if (records.groupby("canonical_smiles")["split"].nunique() > 1).any():
        raise ValueError("Coordinated output assigns an exact molecule to more than one split.")
    if (records.groupby("scaffold_key")["split"].nunique() > 1).any():
        raise ValueError("Coordinated output assigns a scaffold to more than one split.")
    for task in task_names:
        for split in REQUIRED_SPLITS:
            labels = set(records.loc[(records["task_name"] == task) & (records["split"] == split), "target"])
            if labels != {0, 1}:
                raise ValueError(
                    f"Coordinated output {task}/{split} must contain both binary classes; found {sorted(labels)}."
                )


def _write_split_files(
    records: pd.DataFrame,
    config: MultiTaskConfig,
    destination: Path,
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    columns = ["molecule_id", "smiles", "canonical_smiles", "target", "split"]
    for task_name, task in config.tasks.items():
        endpoint_root = destination / task.endpoint_id
        endpoint_root.mkdir(parents=True, exist_ok=True)
        for split in REQUIRED_SPLITS:
            output = records.loc[
                (records["task_name"] == task_name) & (records["split"] == split), columns
            ].sort_values(["canonical_smiles", "molecule_id"], kind="stable")
            path = endpoint_root / SPLIT_FILE_NAMES[split]
            output.to_csv(path, index=False, lineterminator="\n")
            hashes[f"{task_name}/{split}"] = _sha256(path)
    return hashes


def _write_csv(frame: pd.DataFrame, path: Path, columns: tuple[str, ...]) -> None:
    frame.loc[:, list(columns)].to_csv(path, index=False, lineterminator="\n")


def _build_manifest(
    *,
    config: MultiTaskConfig,
    datasets: Mapping[str, EndpointDatasetSplits],
    retained: pd.DataFrame,
    conflicts: pd.DataFrame,
    deduplications: pd.DataFrame,
    scaffold_assignments: pd.DataFrame,
    fractions: Mapping[str, float],
    seed: int,
    output_hashes: Mapping[str, str],
    audit: AuditResult,
) -> dict[str, Any]:
    endpoint_summaries: dict[str, Any] = {}
    total_input = 0
    for task_name, endpoint_splits in datasets.items():
        input_count = sum(len(frame) for frame in endpoint_splits.by_name().values())
        total_input += input_count
        task_records = retained[retained["task_name"] == task_name]
        split_counts: dict[str, Any] = {}
        for split in REQUIRED_SPLITS:
            split_frame = task_records[task_records["split"] == split]
            counts = split_frame["target"].value_counts().sort_index()
            split_counts[split] = {
                "row_count": int(len(split_frame)),
                "class_counts": {str(label): int(counts.get(label, 0)) for label in (0, 1)},
                "target_fraction": fractions[split],
                "achieved_fraction": float(len(split_frame) / len(task_records)),
            }
        endpoint_conflicts = conflicts[conflicts["endpoint_id"] == endpoint_splits.endpoint.endpoint_id]
        endpoint_dedup = deduplications[
            deduplications["endpoint_id"] == endpoint_splits.endpoint.endpoint_id
        ]
        endpoint_summaries[task_name] = {
            "endpoint_id": endpoint_splits.endpoint.endpoint_id,
            "input_rows": int(input_count),
            "output_rows": int(len(task_records)),
            "conflict_groups_quarantined": int(len(endpoint_conflicts)),
            "conflict_records_quarantined": int(endpoint_conflicts["row_count"].sum()),
            "duplicate_groups_collapsed": int(len(endpoint_dedup)),
            "duplicate_records_removed": int(endpoint_dedup["removed_row_count"].sum()),
            "splits": split_counts,
        }
    global_splits = {
        split: {
            "row_count": int((retained["split"] == split).sum()),
            "target_fraction": fractions[split],
            "achieved_fraction": float((retained["split"] == split).mean()),
        }
        for split in REQUIRED_SPLITS
    }
    source_hashes = {
        f"{task}/{split}": _sha256(endpoint_splits.paths[split])
        for task, endpoint_splits in datasets.items()
        for split in REQUIRED_SPLITS
    }
    return {
        "schema_version": "1.0.0",
        "split_track": "coordinated_multitask",
        "seed": seed,
        "target_split_fractions": dict(fractions),
        "rdkit_version": getattr(rdBase, "rdkitVersion", None),
        "input_rows": int(total_input),
        "output_rows": int(len(retained)),
        "invalid_molecules": 0,
        "conflicts": {
            "groups_quarantined": int(len(conflicts)),
            "records_quarantined": int(conflicts["row_count"].sum()),
        },
        "deduplication": {
            "groups_collapsed": int(len(deduplications)),
            "records_removed": int(deduplications["removed_row_count"].sum()),
        },
        "global_scaffold_groups": {
            "total": int(len(scaffold_assignments)),
            "by_split": {
                split: int((scaffold_assignments["split"] == split).sum())
                for split in REQUIRED_SPLITS
            },
        },
        "global_splits": global_splits,
        "endpoints": endpoint_summaries,
        "source_file_sha256": source_hashes,
        "output_file_sha256": dict(output_hashes),
        "audit_summary": audit.summary,
        "artifacts": {
            "conflicts": "quarantined_conflicts.csv",
            "deduplication": "deduplication_provenance.csv",
            "scaffold_assignments": "global_scaffold_assignments.csv",
            "audit_directory": "audit",
        },
    }


def _validate_split_fractions(values: Mapping[str, float]) -> dict[str, float]:
    if set(values) != set(REQUIRED_SPLITS):
        raise ValueError(f"Split fractions must define exactly: {', '.join(REQUIRED_SPLITS)}.")
    fractions = {split: float(values[split]) for split in REQUIRED_SPLITS}
    if any(value <= 0 or value >= 1 for value in fractions.values()):
        raise ValueError("Every split fraction must be greater than zero and less than one.")
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1.0.")
    return fractions


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "CoordinatedSplitResult",
    "DEFAULT_SPLIT_FRACTIONS",
    "build_coordinated_multitask_splits",
]
