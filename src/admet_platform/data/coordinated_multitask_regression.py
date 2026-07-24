"""Leakage-safe coordinated splits for continuous multi-task ADMET endpoints."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase

from admet_platform.config import _load_yaml_mapping
from admet_platform.data.scaffolds import safe_murcko_scaffold


REGRESSION_SCHEMA_VERSION = "1.0.0"
REGRESSION_SPLITS = ("train", "validation", "test")
OUTPUT_FILE_NAMES = {"train": "train.csv", "validation": "valid.csv", "test": "test.csv"}
INVALID_COLUMNS = (
    "task_name",
    "endpoint_id",
    "source_split",
    "source_row_index",
    "molecule_id",
    "smiles",
    "target_original",
    "reason",
)
QUARANTINE_COLUMNS = (
    "task_name",
    "endpoint_id",
    "canonical_smiles",
    "targets",
    "source_splits",
    "molecule_ids",
    "row_count",
    "target_min",
    "target_max",
    "target_range",
    "reason",
)
DEDUPLICATION_COLUMNS = (
    "task_name",
    "endpoint_id",
    "canonical_smiles",
    "target_original",
    "kept_molecule_id",
    "kept_source_split",
    "removed_molecule_ids",
    "removed_source_splits",
    "input_row_count",
    "removed_row_count",
)


@dataclass(frozen=True)
class RegressionEndpointSpec:
    """Source and scientific metadata for one continuous endpoint."""

    task_name: str
    endpoint_id: str
    tdc_name: str
    units: str
    target_definition: str
    recommended_transform: str


@dataclass(frozen=True)
class RegressionSplitConfig:
    """Validated coordinated-regression source configuration."""

    source_path: Path
    run_name: str
    split_track: str
    pytdc_version: str
    random_seed: int
    source_root: Path
    output_root: Path
    split_fractions: Mapping[str, float]
    source_files: Mapping[str, str]
    tasks: Mapping[str, RegressionEndpointSpec]
    duplicate_policy: Mapping[str, str]


@dataclass(frozen=True)
class CoordinatedRegressionSplitResult:
    """Generated regression split metadata."""

    output_root: Path
    manifest: Mapping[str, Any]


def load_regression_split_config(path: str | Path) -> RegressionSplitConfig:
    """Load a regression-only coordinated split configuration."""

    source_path = Path(path).resolve()
    raw = _load_yaml_mapping(source_path.read_text(encoding="utf-8"), source=str(source_path))
    required = {
        "schema_version",
        "run_name",
        "split_track",
        "pytdc_version",
        "random_seed",
        "source_root",
        "output_root",
        "split_fractions",
        "source_files",
        "duplicate_policy",
        "tasks",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError("Regression split config is missing: " + ", ".join(missing))
    if raw["schema_version"] != REGRESSION_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be '{REGRESSION_SCHEMA_VERSION}'.")
    if raw["split_track"] != "coordinated_multitask_regression":
        raise ValueError("split_track must be 'coordinated_multitask_regression'.")
    seed = raw["random_seed"]
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("random_seed must be a non-negative integer.")
    fractions = _validate_split_fractions(raw["split_fractions"])
    source_files = _validate_source_files(raw["source_files"])
    tasks = _parse_tasks(raw["tasks"])
    duplicate_policy = raw["duplicate_policy"]
    expected_policy = {
        "identical_continuous_labels": "keep_first_deterministically",
        "distinct_continuous_labels": "quarantine_entire_endpoint_structure_group",
    }
    if duplicate_policy != expected_policy:
        raise ValueError(f"duplicate_policy must equal {expected_policy}.")
    return RegressionSplitConfig(
        source_path=source_path,
        run_name=_nonempty(raw["run_name"], "run_name"),
        split_track=raw["split_track"],
        pytdc_version=_nonempty(raw["pytdc_version"], "pytdc_version"),
        random_seed=seed,
        source_root=(source_path.parent / _nonempty(raw["source_root"], "source_root")).resolve(),
        output_root=(source_path.parent / _nonempty(raw["output_root"], "output_root")).resolve(),
        split_fractions=fractions,
        source_files=source_files,
        tasks=tasks,
        duplicate_policy=dict(duplicate_policy),
    )


def build_coordinated_multitask_regression_splits(
    config: RegressionSplitConfig,
    *,
    source_root: str | Path | None = None,
    output_root: str | Path | None = None,
    seed: int | None = None,
) -> CoordinatedRegressionSplitResult:
    """Build deterministic global scaffold splits without transforming targets."""

    source = Path(source_root).resolve() if source_root is not None else config.source_root
    destination = Path(output_root).resolve() if output_root is not None else config.output_root
    if source == destination or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Regression source and coordinated output roots must be separate.")
    split_seed = config.random_seed if seed is None else seed
    if not isinstance(split_seed, int) or isinstance(split_seed, bool) or split_seed < 0:
        raise ValueError("seed must be a non-negative integer.")

    records, invalid, source_hashes = load_regression_source_records(config, source)
    retained, quarantined, deduplicated = resolve_continuous_duplicates(records)
    assignments, scaffold_assignments = assign_regression_scaffolds(
        retained, tuple(config.tasks), config.split_fractions, split_seed
    )
    retained = retained.copy()
    retained["split"] = retained["scaffold_key"].map(assignments)
    validate_regression_split_records(retained, tuple(config.tasks))

    destination.mkdir(parents=True, exist_ok=True)
    output_hashes = _write_split_files(retained, config, destination)
    _write_csv(invalid, destination / "invalid_source_records.csv", INVALID_COLUMNS)
    _write_csv(
        quarantined,
        destination / "quarantined_continuous_duplicates.csv",
        QUARANTINE_COLUMNS,
    )
    _write_csv(
        deduplicated,
        destination / "deduplication_provenance.csv",
        DEDUPLICATION_COLUMNS,
    )
    scaffold_assignments.to_csv(
        destination / "global_scaffold_assignments.csv", index=False, lineterminator="\n"
    )
    manifest = _build_manifest(
        config=config,
        source_hashes=source_hashes,
        retained=retained,
        invalid=invalid,
        quarantined=quarantined,
        deduplicated=deduplicated,
        output_hashes=output_hashes,
        seed=split_seed,
    )
    manifest_path = destination / "coordinated_regression_split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return CoordinatedRegressionSplitResult(output_root=destination, manifest=manifest)


def load_regression_source_records(
    config: RegressionSplitConfig,
    source_root: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Load finite continuous labels and normalize molecular identities."""

    root = Path(source_root).resolve() if source_root is not None else config.source_root
    rows: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    source_order = 0
    for task_name, endpoint in config.tasks.items():
        for source_split, filename in config.source_files.items():
            path = root / endpoint.endpoint_id / filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing regression source split: {path}")
            hashes[f"{task_name}/{source_split}"] = _sha256(path)
            frame = pd.read_csv(path)
            missing_columns = [
                name for name in ("molecule_id",) if name not in frame.columns
            ]
            smiles_column = (
                "canonical_smiles"
                if "canonical_smiles" in frame.columns
                else "smiles" if "smiles" in frame.columns else None
            )
            target_column = (
                "target_original"
                if "target_original" in frame.columns
                else "target" if "target" in frame.columns else None
            )
            if missing_columns or smiles_column is None or target_column is None:
                raise ValueError(
                    f"Regression source {path} must contain molecule_id, a SMILES column, "
                    "and target or target_original."
                )
            if source_split != "raw" and "split" in frame.columns:
                observed = set(frame["split"].dropna().astype(str).str.strip())
                aliases = (
                    {"validation", "valid"}
                    if source_split == "validation"
                    else {source_split}
                )
                if not observed or not observed.issubset(aliases):
                    raise ValueError(f"Regression source {path} has unexpected split labels.")
            numeric_targets = pd.to_numeric(frame[target_column], errors="coerce")
            for row_position, (row_index, row) in enumerate(frame.iterrows()):
                supplied = str(row[smiles_column]).strip() if pd.notna(row[smiles_column]) else ""
                target = numeric_targets.iloc[row_position]
                reason = None
                molecule = Chem.MolFromSmiles(supplied) if supplied else None
                if not np.isfinite(target):
                    reason = "missing_or_nonfinite_continuous_target"
                elif molecule is None:
                    reason = "invalid_smiles"
                if reason is not None:
                    invalid.append(
                        {
                            "task_name": task_name,
                            "endpoint_id": endpoint.endpoint_id,
                            "source_split": source_split,
                            "source_row_index": int(row_index),
                            "molecule_id": str(row["molecule_id"]),
                            "smiles": supplied,
                            "target_original": None if not np.isfinite(target) else float(target),
                            "reason": reason,
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
                            "source_split": source_split,
                            "source_row_index": int(row_index),
                            "molecule_id": str(row["molecule_id"]),
                            "smiles": supplied,
                            "target_original": float(target),
                            "reason": f"scaffold_generation_failed:{type(exc).__name__}:{exc}",
                        }
                    )
                    source_order += 1
                    continue
                rows.append(
                    {
                        "task_name": task_name,
                        "endpoint_id": endpoint.endpoint_id,
                        "molecule_id": str(row["molecule_id"]),
                        "smiles": str(row.get("smiles", supplied)),
                        "canonical_smiles": canonical,
                        "target_original": float(target),
                        "source_split": source_split,
                        "source_row_index": int(row_index),
                        "source_order": source_order,
                        "scaffold": scaffold,
                        "scaffold_key": scaffold if scaffold else f"ACYCLIC::{canonical}",
                    }
                )
                source_order += 1
    return (
        pd.DataFrame(rows),
        pd.DataFrame(invalid, columns=INVALID_COLUMNS),
        hashes,
    )


def resolve_continuous_duplicates(
    records: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Collapse identical measurements and quarantine conflicting measurements."""

    retained_rows: list[pd.Series] = []
    quarantined_rows: list[dict[str, Any]] = []
    deduplication_rows: list[dict[str, Any]] = []
    grouped = records.groupby(["task_name", "endpoint_id", "canonical_smiles"], sort=True)
    for (task_name, endpoint_id, canonical), group in grouped:
        ordered = group.sort_values("source_order", kind="stable")
        targets = sorted(set(float(value) for value in ordered["target_original"]))
        if len(targets) > 1:
            quarantined_rows.append(
                {
                    "task_name": task_name,
                    "endpoint_id": endpoint_id,
                    "canonical_smiles": canonical,
                    "targets": "|".join(format(value, ".17g") for value in targets),
                    "source_splits": "|".join(map(str, ordered["source_split"])),
                    "molecule_ids": "|".join(map(str, ordered["molecule_id"])),
                    "row_count": int(len(ordered)),
                    "target_min": min(targets),
                    "target_max": max(targets),
                    "target_range": max(targets) - min(targets),
                    "reason": "distinct_continuous_labels",
                }
            )
            continue
        representative = ordered.iloc[0]
        retained_rows.append(representative)
        if len(ordered) > 1:
            removed = ordered.iloc[1:]
            deduplication_rows.append(
                {
                    "task_name": task_name,
                    "endpoint_id": endpoint_id,
                    "canonical_smiles": canonical,
                    "target_original": targets[0],
                    "kept_molecule_id": str(representative["molecule_id"]),
                    "kept_source_split": str(representative["source_split"]),
                    "removed_molecule_ids": "|".join(map(str, removed["molecule_id"])),
                    "removed_source_splits": "|".join(map(str, removed["source_split"])),
                    "input_row_count": int(len(ordered)),
                    "removed_row_count": int(len(removed)),
                }
            )
    retained = pd.DataFrame(retained_rows)
    if retained.empty:
        raise ValueError("No regression records remain after duplicate resolution.")
    retained = retained.sort_values("source_order", kind="stable").reset_index(drop=True)
    return (
        retained,
        pd.DataFrame(quarantined_rows, columns=QUARANTINE_COLUMNS),
        pd.DataFrame(deduplication_rows, columns=DEDUPLICATION_COLUMNS),
    )


def assign_regression_scaffolds(
    records: pd.DataFrame,
    task_names: tuple[str, ...],
    fractions: Mapping[str, float],
    seed: int,
) -> tuple[dict[str, str], pd.DataFrame]:
    """Assign each global scaffold group to exactly one split."""

    groups = []
    for scaffold_key, frame in records.groupby("scaffold_key", sort=True):
        groups.append(
            {
                "scaffold_key": str(scaffold_key),
                "row_count": int(len(frame)),
                "endpoint_counts": Counter(frame["task_name"]),
            }
        )
    assignments: dict[str, str] = {}
    global_counts: Counter[str] = Counter()
    endpoint_counts: Counter[tuple[str, str]] = Counter()

    for split in ("validation", "test", "train"):
        required = set(task_names)
        while required:
            candidates = [group for group in groups if group["scaffold_key"] not in assignments]
            available = {
                task: [group for group in candidates if group["endpoint_counts"].get(task, 0)]
                for task in required
            }
            impossible = [task for task, options in available.items() if not options]
            if impossible:
                raise ValueError(
                    f"Cannot place endpoint(s) in split '{split}': {', '.join(sorted(impossible))}."
                )
            rarest = min(
                required,
                key=lambda task: (len(available[task]), _seeded_rank(seed, split, task)),
            )
            chosen = min(
                available[rarest],
                key=lambda group: (
                    -sum(bool(group["endpoint_counts"].get(task, 0)) for task in required),
                    group["row_count"],
                    _seeded_rank(seed, split, group["scaffold_key"]),
                ),
            )
            _record_assignment(
                chosen, split, assignments, global_counts, endpoint_counts
            )
            required = {
                task for task in required if endpoint_counts[(split, task)] == 0
            }

    global_targets = {
        split: len(records) * fractions[split] for split in REGRESSION_SPLITS
    }
    task_totals = Counter(records["task_name"])
    endpoint_targets = {
        (split, task): task_totals[task] * fractions[split]
        for split in REGRESSION_SPLITS
        for task in task_names
    }
    remaining = [group for group in groups if group["scaffold_key"] not in assignments]
    remaining.sort(
        key=lambda group: (
            -group["row_count"],
            _seeded_rank(seed, "group-order", group["scaffold_key"]),
        )
    )
    for group in remaining:
        split = min(
            REGRESSION_SPLITS,
            key=lambda candidate: (
                _assignment_score(
                    group,
                    candidate,
                    task_names,
                    global_counts,
                    endpoint_counts,
                    global_targets,
                    endpoint_targets,
                ),
                _seeded_rank(seed, group["scaffold_key"], candidate),
            ),
        )
        _record_assignment(group, split, assignments, global_counts, endpoint_counts)

    assignment_rows = [
        {
            "scaffold_key": group["scaffold_key"],
            "split": assignments[group["scaffold_key"]],
            "row_count": group["row_count"],
            "endpoints": "|".join(sorted(group["endpoint_counts"])),
        }
        for group in groups
    ]
    return assignments, pd.DataFrame(assignment_rows).sort_values(
        "scaffold_key", kind="stable"
    )


def validate_regression_split_records(
    records: pd.DataFrame, task_names: tuple[str, ...]
) -> None:
    """Block exact, scaffold, duplicate, and non-finite regression outputs."""

    if records.empty:
        raise ValueError("No valid regression records remain after source filtering.")
    if records.duplicated(["endpoint_id", "canonical_smiles"]).any():
        raise ValueError("Regression output contains endpoint/canonical duplicates.")
    if records.groupby("canonical_smiles")["split"].nunique().gt(1).any():
        raise ValueError("An exact molecule is assigned to more than one split.")
    if records.groupby("scaffold_key")["split"].nunique().gt(1).any():
        raise ValueError("A Murcko scaffold is assigned to more than one split.")
    if not np.isfinite(records["target_original"].to_numpy(dtype=float)).all():
        raise ValueError("Regression output contains non-finite continuous targets.")
    for task in task_names:
        for split in REGRESSION_SPLITS:
            if records[(records["task_name"] == task) & (records["split"] == split)].empty:
                raise ValueError(f"Regression output {task}/{split} is empty.")


def _parse_tasks(raw: Any) -> dict[str, RegressionEndpointSpec]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError("tasks must be a non-empty mapping.")
    tasks: dict[str, RegressionEndpointSpec] = {}
    endpoint_ids: set[str] = set()
    for task_name, value in raw.items():
        if not isinstance(value, dict):
            raise ValueError(f"Task '{task_name}' must be a mapping.")
        required = (
            "endpoint_id",
            "tdc_name",
            "task_group",
            "task_type",
            "units",
            "target_definition",
            "recommended_transform",
        )
        missing = [field for field in required if field not in value]
        if missing:
            raise ValueError(f"Task '{task_name}' is missing: {', '.join(missing)}.")
        if value["task_type"] != "regression":
            raise ValueError(f"Task '{task_name}' must use task_type: regression.")
        endpoint_id = _nonempty(value["endpoint_id"], f"tasks.{task_name}.endpoint_id")
        if endpoint_id in endpoint_ids:
            raise ValueError(f"Duplicate regression endpoint_id '{endpoint_id}'.")
        endpoint_ids.add(endpoint_id)
        tasks[task_name] = RegressionEndpointSpec(
            task_name=task_name,
            endpoint_id=endpoint_id,
            tdc_name=_nonempty(value["tdc_name"], f"tasks.{task_name}.tdc_name"),
            units=_nonempty(value["units"], f"tasks.{task_name}.units"),
            target_definition=_nonempty(
                value["target_definition"], f"tasks.{task_name}.target_definition"
            ),
            recommended_transform=_nonempty(
                value["recommended_transform"],
                f"tasks.{task_name}.recommended_transform",
            ),
        )
    return tasks


def _validate_source_files(raw: Any) -> dict[str, str]:
    allowed_shapes = ({"raw"}, set(REGRESSION_SPLITS))
    if not isinstance(raw, dict) or set(raw) not in allowed_shapes:
        raise ValueError(
            "source_files must define either raw or train, validation, and test."
        )
    return {
        split: _nonempty(raw[split], f"source_files.{split}") for split in raw
    }


def _validate_split_fractions(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict) or set(raw) != set(REGRESSION_SPLITS):
        raise ValueError("split_fractions must define train, validation, and test.")
    values = {split: float(raw[split]) for split in REGRESSION_SPLITS}
    if any(not 0 < value < 1 for value in values.values()) or not np.isclose(
        sum(values.values()), 1.0
    ):
        raise ValueError("split_fractions must be positive and sum to 1.")
    return values


def _write_split_files(
    records: pd.DataFrame, config: RegressionSplitConfig, destination: Path
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    columns = [
        "molecule_id",
        "smiles",
        "canonical_smiles",
        "target",
        "target_original",
        "split",
    ]
    for task_name, endpoint in config.tasks.items():
        endpoint_root = destination / endpoint.endpoint_id
        endpoint_root.mkdir(parents=True, exist_ok=True)
        for split in REGRESSION_SPLITS:
            frame = records[
                (records["task_name"] == task_name) & (records["split"] == split)
            ].copy()
            frame["target"] = frame["target_original"]
            frame = frame.sort_values(["canonical_smiles", "molecule_id"], kind="stable")
            path = endpoint_root / OUTPUT_FILE_NAMES[split]
            frame.loc[:, columns].to_csv(path, index=False, lineterminator="\n")
            hashes[f"{task_name}/{split}"] = _sha256(path)
    return hashes


def _build_manifest(
    *,
    config: RegressionSplitConfig,
    source_hashes: Mapping[str, str],
    retained: pd.DataFrame,
    invalid: pd.DataFrame,
    quarantined: pd.DataFrame,
    deduplicated: pd.DataFrame,
    output_hashes: Mapping[str, str],
    seed: int,
) -> dict[str, Any]:
    endpoints: dict[str, Any] = {}
    for task_name, endpoint in config.tasks.items():
        task_frame = retained[retained["task_name"] == task_name]
        task_invalid = invalid[invalid["task_name"] == task_name]
        task_quarantined = quarantined[quarantined["task_name"] == task_name]
        task_deduplicated = deduplicated[deduplicated["task_name"] == task_name]
        quarantined_rows = int(task_quarantined["row_count"].sum())
        identical_rows_removed = int(task_deduplicated["removed_row_count"].sum())
        source_rows = (
            len(task_frame)
            + len(task_invalid)
            + quarantined_rows
            + identical_rows_removed
        )
        split_summary = {}
        for split in REGRESSION_SPLITS:
            values = task_frame.loc[task_frame["split"] == split, "target_original"]
            split_summary[split] = (
                _distribution(values)
                if split != "test"
                else {"row_count": int(len(values)), "target_statistics_locked": True}
            )
            split_summary[split]["sha256"] = output_hashes[f"{task_name}/{split}"]
        endpoints[task_name] = {
            "endpoint_id": endpoint.endpoint_id,
            "tdc_name": endpoint.tdc_name,
            "target_definition": endpoint.target_definition,
            "units": endpoint.units,
            "recommended_transform": endpoint.recommended_transform,
            "source_row_count": int(source_rows),
            "valid_canonicalized_records": int(source_rows - len(task_invalid)),
            "unique_valid_canonical_structures": int(
                task_frame["canonical_smiles"].nunique() + len(task_quarantined)
            ),
            "invalid_records": int(len(task_invalid)),
            "exact_duplicate_groups": int(
                len(task_deduplicated) + len(task_quarantined)
            ),
            "identical_label_duplicate_groups_collapsed": int(
                len(task_deduplicated)
            ),
            "identical_duplicate_rows_removed": identical_rows_removed,
            "conflicting_label_duplicate_groups_quarantined": int(
                len(task_quarantined)
            ),
            "conflicting_duplicate_rows_quarantined": quarantined_rows,
            "retained_rows": int(len(task_frame)),
            "splits": split_summary,
        }
    cross_endpoint_overlap = {}
    for split in REGRESSION_SPLITS:
        split_frame = retained[retained["split"] == split]
        overlap = split_frame.groupby("canonical_smiles")["task_name"].nunique()
        cross_endpoint_overlap[split] = {
            "exact_structure_groups": int(overlap.gt(1).sum()),
            "allowed": True,
        }
    return {
        "schema_version": REGRESSION_SCHEMA_VERSION,
        "run_name": config.run_name,
        "split_track": config.split_track,
        "seed": seed,
        "pytdc_version": config.pytdc_version,
        "rdkit_version": rdBase.rdkitVersion,
        "split_fractions": dict(config.split_fractions),
        "duplicate_policy": dict(config.duplicate_policy),
        "target_transforms_fitted": False,
        "target_normalization_statistics_used": [],
        "input_file_sha256": dict(source_hashes),
        "output_file_sha256": dict(output_hashes),
        "invalid_source_records": int(len(invalid)),
        "quarantined_conflicting_groups": int(len(quarantined)),
        "identical_duplicate_groups_collapsed": int(len(deduplicated)),
        "global_summary": {
            "total_retained_endpoint_records": int(len(retained)),
            "unique_canonical_structures": int(retained["canonical_smiles"].nunique()),
            "unique_scaffold_groups": int(retained["scaffold_key"].nunique()),
            "within_split_cross_endpoint_overlap": cross_endpoint_overlap,
        },
        "endpoints": endpoints,
        "leakage_audit": {
            "exact_smiles_cross_split_groups": 0,
            "murcko_scaffold_cross_split_groups": 0,
            "endpoint_structure_duplicates": 0,
            "continuous_targets_finite": True,
            "status": "passed",
        },
    }


def _distribution(values: pd.Series) -> dict[str, Any]:
    return {
        "row_count": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "p05": float(values.quantile(0.05)),
        "median": float(values.median()),
        "p95": float(values.quantile(0.95)),
        "max": float(values.max()),
    }


def _record_assignment(
    group: Mapping[str, Any],
    split: str,
    assignments: dict[str, str],
    global_counts: Counter[str],
    endpoint_counts: Counter[tuple[str, str]],
) -> None:
    assignments[str(group["scaffold_key"])] = split
    global_counts[split] += int(group["row_count"])
    for task, count in group["endpoint_counts"].items():
        endpoint_counts[(split, task)] += int(count)


def _assignment_score(
    group: Mapping[str, Any],
    candidate: str,
    task_names: tuple[str, ...],
    global_counts: Counter[str],
    endpoint_counts: Counter[tuple[str, str]],
    global_targets: Mapping[str, float],
    endpoint_targets: Mapping[tuple[str, str], float],
) -> float:
    score = 0.0
    for split in REGRESSION_SPLITS:
        added = int(group["row_count"]) if split == candidate else 0
        score += _relative_square(global_counts[split] + added, global_targets[split])
        for task in task_names:
            endpoint_added = (
                int(group["endpoint_counts"].get(task, 0)) if split == candidate else 0
            )
            score += 2.0 * _relative_square(
                endpoint_counts[(split, task)] + endpoint_added,
                endpoint_targets[(split, task)],
            )
    return score


def _relative_square(actual: float, target: float) -> float:
    return ((actual - target) / max(target, 1.0)) ** 2


def _seeded_rank(seed: int, *parts: str) -> str:
    payload = "|".join((str(seed), *map(str, parts)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path, columns: tuple[str, ...]) -> None:
    frame.loc[:, list(columns)].to_csv(path, index=False, lineterminator="\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    return value.strip()


__all__ = [
    "CoordinatedRegressionSplitResult",
    "RegressionEndpointSpec",
    "RegressionSplitConfig",
    "assign_regression_scaffolds",
    "build_coordinated_multitask_regression_splits",
    "load_regression_source_records",
    "load_regression_split_config",
    "resolve_continuous_duplicates",
    "validate_regression_split_records",
]
