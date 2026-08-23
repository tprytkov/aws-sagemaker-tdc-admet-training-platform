"""Create deterministic scaffold-grouped nested OOF manifests from frozen TRAIN only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Final, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from admet_platform.gmc_mpnn.model_data import (  # noqa: E402
    TRAIN_SPLIT_CONTRACT,
    FrozenTrainingIdentity,
    FrozenTrainingManifestData,
    load_frozen_training_manifest,
)


OOF_MANIFEST_VERSION: Final = "gmc-mpnn-bbb-train-nested-oof-v1"
EXPECTED_TRAIN_COUNT: Final = 1558
OUTER_FOLD_COUNT: Final = 5
INNER_FOLD_COUNT: Final = 8
SPLIT_SEED: Final = 1729
OUTER_FILENAME: Final = "outer_fold_manifest.csv"
INNER_FILENAME: Final = "inner_split_manifest.csv"
SUMMARY_FILENAME: Final = "split_summary.json"
DEFAULT_TRAIN_DIR: Final = ROOT / "outputs" / "gpu" / "pilot" / "gmc_mpnn_training_preprocessing_v2"


class GMCOOFSplitError(RuntimeError):
    """A deterministic nested OOF split contract violation."""


@dataclass(frozen=True)
class OOFConfig:
    training_preprocessing_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class SplitTables:
    outer: pd.DataFrame
    inner: pd.DataFrame


def create_oof_manifest(
    config: OOFConfig,
    *,
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Validate frozen TRAIN, generate the split twice, and atomically publish it."""

    output_dir = config.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"OOF output directory already exists: {output_dir}")
    frozen = load_frozen_training_manifest(config.training_preprocessing_dir)
    if len(frozen.records) != EXPECTED_TRAIN_COUNT:
        raise GMCOOFSplitError("Frozen TRAIN must contain exactly 1,558 successful molecules.")
    _validate_train_only_provenance(frozen)

    first = _generate_split_tables(frozen.records)
    repeated = _generate_split_tables(frozen.records)
    if not first.outer.equals(repeated.outer) or not first.inner.equals(repeated.inner):
        raise GMCOOFSplitError("Repeated deterministic split generation changed assignments.")
    checks, outer_folds, inner_splits = _validate_split_tables(first)
    summary = {
        "oof_manifest_version": OOF_MANIFEST_VERSION,
        "train_count": len(frozen.records),
        "outer_fold_count": OUTER_FOLD_COUNT,
        "inner_fold_count": INNER_FOLD_COUNT,
        "inner_early_stop_target_fraction": 1.0 / INNER_FOLD_COUNT,
        "split_seed": SPLIT_SEED,
        "split_algorithm": {
            "outer": (
                "sklearn.model_selection.StratifiedGroupKFold("
                "n_splits=5, shuffle=True, random_state=1729)"
            ),
            "inner": (
                "best class-complete holdout from sklearn.model_selection."
                "StratifiedGroupKFold(n_splits=8, shuffle=True, "
                "random_state=1729+1000+outer_fold), minimizing target-size then "
                "class-ratio deviation with fold index as the final tie-break"
            ),
            "stratification_label": "BBB binary TRAIN label",
            "group": "Bemis-Murcko scaffold",
            "scaffold_input": "frozen parent-fragment geometry molecule",
            "acyclic_fallback": "RDKit canonical isomeric SMILES",
        },
        "unique_scaffold_count": int(first.outer["scaffold_key"].nunique()),
        "outer_folds": outer_folds,
        "inner_splits": inner_splits,
        "checks": {
            **checks,
            "repeated_generation_identical": True,
        },
        "assignment_hashes": {
            "outer_fold_manifest_sha256": _dataframe_sha256(first.outer),
            "inner_split_manifest_sha256": _dataframe_sha256(first.inner),
        },
        "frozen_train_provenance": {
            "source_row_count": frozen.source_row_count,
            "successful_molecule_count": len(frozen.records),
            "feature_manifest_sha256": frozen.feature_manifest_sha256,
            "molecule_status_sha256": frozen.molecule_status_sha256,
            "ordered_input_artifact_sha256": frozen.ordered_input_artifact_sha256,
            **dict(frozen.preprocessing_provenance),
        },
        "rdkit_version": rdBase.rdkitVersion,
        "scikit_learn_version": metadata.version("scikit-learn"),
        "git_commit": git_commit or _git_commit(),
        "validation_artifact_accessed": False,
        "test_artifact_accessed": False,
    }
    _publish(output_dir, first, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create the fixed TRAIN-only nested scaffold OOF membership manifests."
    )
    parser.add_argument("--training-preprocessing-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = create_oof_manifest(
        OOFConfig(
            training_preprocessing_dir=args.training_preprocessing_dir,
            output_dir=args.output_dir,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


def _generate_split_tables(records: Sequence[FrozenTrainingIdentity]) -> SplitTables:
    labels = np.asarray([record.label for record in records], dtype=np.int64)
    scaffold_keys = tuple(_scaffold_key(record) for record in records)
    _validate_binary_labels(labels)
    outer_assignments = np.full(len(records), -1, dtype=np.int64)
    splitter = StratifiedGroupKFold(
        n_splits=OUTER_FOLD_COUNT,
        shuffle=True,
        random_state=SPLIT_SEED,
    )
    placeholder = np.zeros(len(records), dtype=np.int8)
    for outer_fold, (_, holdout_indices) in enumerate(
        splitter.split(placeholder, labels, groups=scaffold_keys)
    ):
        if np.any(outer_assignments[holdout_indices] != -1):
            raise GMCOOFSplitError("An outer record was assigned more than once.")
        outer_assignments[holdout_indices] = outer_fold
    if np.any(outer_assignments < 0):
        raise GMCOOFSplitError("At least one TRAIN record lacks an outer-fold assignment.")

    outer = pd.DataFrame(
        {
            "record_key": [record.record_key for record in records],
            "molecule_id": [record.molecule_id for record in records],
            "canonical_smiles": [record.canonical_smiles for record in records],
            "label": labels,
            "scaffold_key": scaffold_keys,
            "outer_fold": outer_assignments,
        }
    )
    inner_rows: list[dict[str, Any]] = []
    for outer_fold in range(OUTER_FOLD_COUNT):
        development_indices = np.flatnonzero(outer_assignments != outer_fold)
        development_labels = labels[development_indices]
        development_groups = np.asarray(scaffold_keys, dtype=object)[development_indices]
        inner_splitter = StratifiedGroupKFold(
            n_splits=INNER_FOLD_COUNT,
            shuffle=True,
            random_state=SPLIT_SEED + 1000 + outer_fold,
        )
        inner_train_local, early_stop_local = _select_inner_split(
            inner_splitter,
            development_labels,
            development_groups,
        )
        role_by_index = {
            int(development_indices[index]): "inner_train" for index in inner_train_local
        }
        role_by_index.update(
            {
                int(development_indices[index]): "inner_early_stop_validation"
                for index in early_stop_local
            }
        )
        if set(role_by_index) != set(development_indices.tolist()):
            raise GMCOOFSplitError("An outer-development record lacks an inner role.")
        for index in development_indices:
            inner_rows.append(
                {
                    "outer_fold": outer_fold,
                    "record_key": records[int(index)].record_key,
                    "inner_role": role_by_index[int(index)],
                }
            )
    inner = pd.DataFrame(
        inner_rows,
        columns=("outer_fold", "record_key", "inner_role"),
    )
    return SplitTables(outer=outer, inner=inner)


def _select_inner_split(
    splitter: StratifiedGroupKFold,
    labels: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target_count = len(labels) / INNER_FOLD_COUNT
    overall_positive_fraction = float(labels.mean())
    candidates: list[tuple[tuple[float, float, int], np.ndarray, np.ndarray]] = []
    for fold_index, (train_indices, early_stop_indices) in enumerate(
        splitter.split(
            np.zeros(len(labels), dtype=np.int8),
            labels,
            groups=groups,
        )
    ):
        if set(labels[early_stop_indices].tolist()) != {0, 1}:
            continue
        if set(labels[train_indices].tolist()) != {0, 1}:
            continue
        score = (
            abs(len(early_stop_indices) - target_count),
            abs(float(labels[early_stop_indices].mean()) - overall_positive_fraction),
            fold_index,
        )
        candidates.append((score, train_indices, early_stop_indices))
    if not candidates:
        raise GMCOOFSplitError(
            "No deterministic inner scaffold split contains both classes in both roles."
        )
    _, train_indices, early_stop_indices = min(candidates, key=lambda candidate: candidate[0])
    return train_indices, early_stop_indices


def _scaffold_key(record: FrozenTrainingIdentity) -> str:
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=record.molecule,
        includeChirality=False,
    )
    if scaffold:
        scaffold_molecule = Chem.MolFromSmiles(scaffold)
        if scaffold_molecule is None:
            raise GMCOOFSplitError(f"Cannot canonicalize scaffold for {record.record_key}.")
        canonical_scaffold = Chem.MolToSmiles(
            scaffold_molecule,
            canonical=True,
            isomericSmiles=False,
        )
        return f"murcko:{canonical_scaffold}"
    canonical_fallback = Chem.MolToSmiles(
        record.molecule,
        canonical=True,
        isomericSmiles=True,
    )
    if not canonical_fallback:
        raise GMCOOFSplitError(f"Cannot create acyclic fallback for {record.record_key}.")
    return f"acyclic:{canonical_fallback}"


def _validate_split_tables(
    tables: SplitTables,
) -> tuple[dict[str, bool], dict[str, Any], dict[str, Any]]:
    outer = tables.outer
    inner = tables.inner
    if len(outer) != EXPECTED_TRAIN_COUNT or outer["record_key"].duplicated().any():
        raise GMCOOFSplitError("Outer assignment is missing or duplicating TRAIN records.")
    if set(outer["outer_fold"]) != set(range(OUTER_FOLD_COUNT)):
        raise GMCOOFSplitError("Outer assignment does not contain exactly five folds.")
    labels_by_key = outer.set_index("record_key")["label"]
    scaffolds_by_key = outer.set_index("record_key")["scaffold_key"]

    outer_summary: dict[str, Any] = {}
    inner_summary: dict[str, Any] = {}
    for outer_fold in range(OUTER_FOLD_COUNT):
        holdout = outer.loc[outer["outer_fold"] == outer_fold]
        development = outer.loc[outer["outer_fold"] != outer_fold]
        _require_both_classes(holdout["label"], f"outer fold {outer_fold} holdout")
        outer_overlap = set(holdout["scaffold_key"]) & set(development["scaffold_key"])
        if outer_overlap:
            raise GMCOOFSplitError(f"Outer fold {outer_fold} has scaffold leakage.")

        fold_inner = inner.loc[inner["outer_fold"] == outer_fold]
        if fold_inner["record_key"].duplicated().any():
            raise GMCOOFSplitError(f"Outer fold {outer_fold} has duplicate inner records.")
        if set(fold_inner["record_key"]) != set(development["record_key"]):
            raise GMCOOFSplitError(f"Outer fold {outer_fold} inner membership is incomplete.")
        early_keys = fold_inner.loc[
            fold_inner["inner_role"] == "inner_early_stop_validation", "record_key"
        ]
        train_keys = fold_inner.loc[fold_inner["inner_role"] == "inner_train", "record_key"]
        early_labels = labels_by_key.loc[early_keys]
        train_labels = labels_by_key.loc[train_keys]
        _require_both_classes(early_labels, f"outer fold {outer_fold} inner early stop")
        _require_both_classes(train_labels, f"outer fold {outer_fold} inner train")
        inner_overlap = set(scaffolds_by_key.loc[early_keys]) & set(
            scaffolds_by_key.loc[train_keys]
        )
        if inner_overlap:
            raise GMCOOFSplitError(f"Outer fold {outer_fold} inner split has scaffold leakage.")

        outer_summary[str(outer_fold)] = {
            "development_count": int(len(development)),
            "development_class_counts": _class_counts(development["label"]),
            "holdout_count": int(len(holdout)),
            "holdout_class_counts": _class_counts(holdout["label"]),
            "development_unique_scaffold_count": int(development["scaffold_key"].nunique()),
            "holdout_unique_scaffold_count": int(holdout["scaffold_key"].nunique()),
            "scaffold_overlap_count": 0,
        }
        inner_summary[str(outer_fold)] = {
            "inner_train_count": int(len(train_keys)),
            "inner_train_class_counts": _class_counts(train_labels),
            "inner_early_stop_validation_count": int(len(early_keys)),
            "inner_early_stop_validation_class_counts": _class_counts(early_labels),
            "inner_early_stop_fraction_of_outer_development": float(
                len(early_keys) / len(development)
            ),
            "scaffold_overlap_count": 0,
            "outer_holdout_used": False,
        }
    checks = {
        "outer_assignment_complete_and_unique": True,
        "outer_scaffold_overlap_absent": True,
        "inner_membership_complete_and_unique": True,
        "inner_scaffold_overlap_absent": True,
        "outer_holdout_excluded_from_inner_splits": True,
        "both_classes_present_in_every_outer_holdout": True,
        "both_classes_present_in_every_inner_early_stop_split": True,
    }
    return checks, outer_summary, inner_summary


def _validate_binary_labels(labels: np.ndarray) -> None:
    if labels.ndim != 1 or not np.isin(labels, (0, 1)).all():
        raise GMCOOFSplitError("TRAIN labels must be a one-dimensional binary array.")
    _require_both_classes(pd.Series(labels), "TRAIN")


def _require_both_classes(labels: pd.Series, label: str) -> None:
    if set(int(value) for value in labels.unique()) != {0, 1}:
        raise GMCOOFSplitError(f"Both classes are required in {label}.")


def _class_counts(labels: pd.Series) -> dict[str, int]:
    counts = labels.value_counts().to_dict()
    return {"0": int(counts.get(0, 0)), "1": int(counts.get(1, 0))}


def _validate_train_only_provenance(frozen: FrozenTrainingManifestData) -> None:
    if frozen.source_row_count != TRAIN_SPLIT_CONTRACT.source_rows:
        raise GMCOOFSplitError("Frozen TRAIN source-row count differs from the contract.")
    if frozen.preprocessing_provenance.get("validation_artifact_accessed") is not False:
        raise GMCOOFSplitError("Frozen TRAIN provenance indicates validation artifact access.")
    if frozen.preprocessing_provenance.get("test_artifact_accessed") is not False:
        raise GMCOOFSplitError("Frozen TRAIN provenance indicates test artifact access.")


def _dataframe_sha256(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _publish(output_dir: Path, tables: SplitTables, summary: dict[str, Any]) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        tables.outer.to_csv(
            temporary_dir / OUTER_FILENAME,
            index=False,
            lineterminator="\n",
        )
        tables.inner.to_csv(
            temporary_dir / INNER_FILENAME,
            index=False,
            lineterminator="\n",
        )
        _write_json(temporary_dir / SUMMARY_FILENAME, summary)
        if output_dir.exists():
            raise FileExistsError(f"OOF output directory appeared during run: {output_dir}")
        os.replace(temporary_dir, output_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise GMCOOFSplitError("Unable to record the Git commit.") from exc


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
