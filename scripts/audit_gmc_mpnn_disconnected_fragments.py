"""Audit disconnected BBB development structures without changing source data.

This analysis considers a deterministic largest-heavy-atom parent candidate only
to measure collisions, label conflicts, and train/validation leakage. It does not
approve or implement parent-fragment standardization for geometry or training.
"""

from __future__ import annotations

import argparse
import builtins
import json
import os
import subprocess
import sys
from collections import Counter
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence
from unittest.mock import patch

import pandas as pd
from rdkit import Chem, rdBase

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from admet_platform.chemprop.config import (  # noqa: E402
    ChempropExperimentConfig,
    load_chemprop_config,
)
from admet_platform.data.scaffolds import safe_murcko_scaffold  # noqa: E402
from admet_platform.gmc_mpnn.data import (  # noqa: E402
    BBBDevelopmentData,
    load_bbb_development_data,
)


DEFAULT_CONFIG = ROOT / "configs" / "chemprop" / "bbb_martins.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "gpu" / "pilot" / "gmc_mpnn_fragment_audit"
PARENT_RULE_DESCRIPTION = (
    "Audit only: select the fragment with the largest heavy-atom count; break ties by "
    "ascending canonical isomeric SMILES. Source structures and split membership are unchanged."
)
OUTPUT_FILENAMES = (
    "fragment_audit_summary.json",
    "disconnected_molecules.csv",
    "parent_collision_groups.csv",
    "train_validation_parent_overlap.csv",
    "train_validation_parent_scaffold_overlap.csv",
)
DISCONNECTED_COLUMNS = (
    "split",
    "molecule_id",
    "canonical_smiles",
    "target",
    "fragment_count",
    "fragment_smiles",
    "fragment_heavy_atom_counts",
    "fragment_formal_charges",
    "total_heavy_atom_count",
    "total_formal_charge",
    "candidate_parent_smiles",
    "candidate_parent_heavy_atom_count",
    "candidate_parent_formal_charge",
    "removed_fragment_smiles",
    "removed_fragment_heavy_atom_counts",
    "removed_fragment_formal_charges",
    "removed_fragment_categories",
)
COLLISION_COLUMNS = (
    "candidate_parent_smiles",
    "collision_group_size",
    "has_label_conflict",
    "splits_present",
    "molecule_ids",
    "original_smiles",
    "targets",
)
PARENT_OVERLAP_COLUMNS = (
    "candidate_parent_smiles",
    "train_row_count",
    "validation_row_count",
    "has_label_conflict",
    "train_identities",
    "validation_identities",
)
SCAFFOLD_OVERLAP_COLUMNS = (
    "candidate_parent_scaffold",
    "train_row_count",
    "validation_row_count",
    "train_parent_smiles",
    "validation_parent_smiles",
    "train_identities",
    "validation_identities",
)
ANALYSIS_COLUMNS = (*DISCONNECTED_COLUMNS, "candidate_parent_scaffold")


class LockedTestAccessError(RuntimeError):
    """Raised before the configured BBB test artifact can be inspected."""


@dataclass
class TestArtifactAccessGuard:
    """Reject and record attempted access to the configured test artifact."""

    test_path: Path
    test_artifact_accessed: bool = False

    def reject(self, candidate: object) -> None:
        try:
            candidate_path = Path(os.path.abspath(os.fspath(candidate)))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return
        if candidate_path == self.test_path:
            self.test_artifact_accessed = True
            raise LockedTestAccessError(
                "Locked BBB test artifact access is prohibited in the fragment audit."
            )


@dataclass(frozen=True)
class FragmentRecord:
    """Canonical and composition facts for one disconnected component."""

    canonical_smiles: str
    heavy_atom_count: int
    formal_charge: int


@dataclass(frozen=True)
class MoleculeFragmentAudit:
    """Read-only fragment analysis for one source canonical SMILES."""

    fragments: tuple[FragmentRecord, ...]
    candidate_parent: FragmentRecord
    removed_fragments: tuple[FragmentRecord, ...]
    removed_fragment_categories: tuple[str, ...]
    total_heavy_atom_count: int
    total_formal_charge: int

    @property
    def fragment_count(self) -> int:
        return len(self.fragments)


@dataclass(frozen=True)
class AuditAnalysis:
    """In-memory audit tables and statistics before provenance is attached."""

    statistics: dict[str, Any]
    disconnected_molecules: pd.DataFrame
    parent_collision_groups: pd.DataFrame
    train_validation_parent_overlap: pd.DataFrame
    train_validation_parent_scaffold_overlap: pd.DataFrame


ConfigLoader = Callable[[str | Path], ChempropExperimentConfig]
DevelopmentLoader = Callable[[ChempropExperimentConfig], BBBDevelopmentData]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only audit of disconnected BBB_Martins train/validation structures, "
            "candidate parent collapse, label conflicts, and leakage."
        ),
        epilog=(
            "This command produces evidence only. It does not approve or apply a fragment "
            "standardization policy."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="BBB Chemprop configuration (default: configs/chemprop/bbb_martins.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the JSON summary and four CSV audit tables.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the five known audit files if they already exist.",
    )
    return parser


def analyze_molecule_fragments(canonical_smiles: str) -> MoleculeFragmentAudit:
    """Describe fragments and choose a deterministic audit-only parent candidate."""

    molecule = Chem.MolFromSmiles(canonical_smiles)
    if molecule is None:
        raise ValueError(f"RDKit could not parse canonical SMILES: {canonical_smiles!r}")
    fragment_molecules = Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True)
    fragments = tuple(
        sorted(
            (_fragment_record(fragment) for fragment in fragment_molecules),
            key=lambda item: (item.canonical_smiles, item.heavy_atom_count, item.formal_charge),
        )
    )
    if not fragments:  # pragma: no cover - RDKit molecules always have at least one fragment
        raise ValueError("Molecule contains no fragments.")
    parent_index = min(
        range(len(fragments)),
        key=lambda index: (-fragments[index].heavy_atom_count, fragments[index].canonical_smiles),
    )
    candidate_parent = fragments[parent_index]
    removed = tuple(fragment for index, fragment in enumerate(fragments) if index != parent_index)
    return MoleculeFragmentAudit(
        fragments=fragments,
        candidate_parent=candidate_parent,
        removed_fragments=removed,
        removed_fragment_categories=tuple(
            categorize_removed_fragment(fragment) for fragment in removed
        ),
        total_heavy_atom_count=sum(fragment.heavy_atom_count for fragment in fragments),
        total_formal_charge=sum(fragment.formal_charge for fragment in fragments),
    )


def categorize_removed_fragment(fragment: FragmentRecord) -> str:
    """Assign one transparent, deliberately small descriptive category."""

    molecule = Chem.MolFromSmiles(fragment.canonical_smiles)
    if molecule is None:  # pragma: no cover - fragment SMILES originated from RDKit
        return "other"
    atoms = tuple(molecule.GetAtoms())
    contains_carbon = any(atom.GetAtomicNum() == 6 for atom in atoms)
    if len(atoms) == 1 and not contains_carbon and fragment.formal_charge != 0:
        return "simple_inorganic_or_monatomic_counterion"
    if not contains_carbon and fragment.heavy_atom_count <= 6:
        return "small_inorganic_fragment"
    if contains_carbon:
        return "organic_coformer_or_counterion"
    return "other"


def analyze_development_frames(train: pd.DataFrame, validation: pd.DataFrame) -> AuditAnalysis:
    """Analyze copies of train/validation rows without changing source frames or membership."""

    train_rows = _analyze_split(train, "train")
    validation_rows = _analyze_split(validation, "validation")
    all_rows = pd.concat([train_rows, validation_rows], ignore_index=True)

    disconnected = all_rows.loc[all_rows["fragment_count"] > 1, DISCONNECTED_COLUMNS].copy()
    collision_records = _parent_collision_records(all_rows)
    parent_overlap_records = _parent_overlap_records(train_rows, validation_rows)
    scaffold_overlap_records = _scaffold_overlap_records(train_rows, validation_rows)

    train_stats = _split_statistics(train_rows)
    validation_stats = _split_statistics(validation_rows)
    combined_identity = _identity_statistics(all_rows)
    combined_collision = _collision_statistics(all_rows)
    label_conflict_records = _label_conflict_group_details(all_rows)
    statistics = {
        "splits": {"train": train_stats, "validation": validation_stats},
        "combined_identity_statistics": combined_identity,
        "total_disconnected_structures": int((all_rows["fragment_count"] > 1).sum()),
        "total_structures_affected_by_candidate_parent_selection": int(
            (all_rows["canonical_smiles"] != all_rows["candidate_parent_smiles"]).sum()
        ),
        "total_parent_collision_groups": combined_collision["parent_collision_group_count"],
        "maximum_parent_collision_size": combined_collision["maximum_collision_group_size"],
        "total_label_conflict_groups": len(label_conflict_records),
        "label_conflict_groups": label_conflict_records,
        "candidate_parent_train_validation_overlap_count": len(parent_overlap_records),
        "candidate_parent_train_validation_overlap_smiles": [
            record["candidate_parent_smiles"] for record in parent_overlap_records
        ],
        "candidate_parent_scaffold_overlap_count": len(scaffold_overlap_records),
        "candidate_parent_scaffold_overlap_scaffolds": [
            record["candidate_parent_scaffold"] for record in scaffold_overlap_records
        ],
    }
    return AuditAnalysis(
        statistics=statistics,
        disconnected_molecules=pd.DataFrame(disconnected, columns=DISCONNECTED_COLUMNS),
        parent_collision_groups=pd.DataFrame(collision_records, columns=COLLISION_COLUMNS),
        train_validation_parent_overlap=pd.DataFrame(
            parent_overlap_records, columns=PARENT_OVERLAP_COLUMNS
        ),
        train_validation_parent_scaffold_overlap=pd.DataFrame(
            scaffold_overlap_records, columns=SCAFFOLD_OVERLAP_COLUMNS
        ),
    )


def run_disconnected_fragment_audit(
    config_path: str | Path,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    overwrite: bool = False,
    config_loader: ConfigLoader = load_chemprop_config,
    development_loader: DevelopmentLoader = load_bbb_development_data,
) -> dict[str, Any]:
    """Load only verified development data, run the audit, and write five artifacts."""

    _reject_test_named_config(config_path)
    config = config_loader(config_path)
    _validate_development_config(config)
    with _deny_test_artifact_access(config) as access_guard:
        development = development_loader(config)
        analysis = analyze_development_frames(development.train, development.validation)

    summary = {
        "dataset": config.dataset,
        "dataset_version": config.dataset_version,
        "loaded_splits": ["train", "validation"],
        "train_row_count": len(development.train),
        "validation_row_count": len(development.validation),
        "train_disconnected_count": analysis.statistics["splits"]["train"][
            "disconnected_molecule_count"
        ],
        "validation_disconnected_count": analysis.statistics["splits"]["validation"][
            "disconnected_molecule_count"
        ],
        "parent_rule_description": PARENT_RULE_DESCRIPTION,
        "original_exact_train_validation_overlap_count": (
            development.leakage.exact_canonical_smiles_count
        ),
        "candidate_parent_train_validation_overlap_count": analysis.statistics[
            "candidate_parent_train_validation_overlap_count"
        ],
        "original_murcko_scaffold_overlap_count": development.leakage.murcko_scaffold_count,
        "candidate_parent_murcko_scaffold_overlap_count": analysis.statistics[
            "candidate_parent_scaffold_overlap_count"
        ],
        **analysis.statistics,
        "rdkit_version": rdBase.rdkitVersion,
        "git_commit": _git_commit(),
        "test_artifact_accessed": access_guard.test_artifact_accessed,
        "policy_approved": False,
    }
    _write_audit_outputs(Path(output_dir), summary, analysis, overwrite=overwrite)
    return summary


def _fragment_record(fragment: Chem.Mol) -> FragmentRecord:
    return FragmentRecord(
        canonical_smiles=Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True),
        heavy_atom_count=sum(atom.GetAtomicNum() != 1 for atom in fragment.GetAtoms()),
        formal_charge=int(Chem.GetFormalCharge(fragment)),
    )


def _analyze_split(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    required = {"molecule_id", "canonical_smiles", "target"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{split} frame is missing required columns: {missing}")
    rows: list[dict[str, Any]] = []
    for source in frame.loc[:, ["molecule_id", "canonical_smiles", "target"]].itertuples(
        index=False
    ):
        audit = analyze_molecule_fragments(str(source.canonical_smiles))
        parent_scaffold = _parent_scaffold(audit.candidate_parent.canonical_smiles)
        rows.append(
            {
                "split": split,
                "molecule_id": str(source.molecule_id),
                "canonical_smiles": str(source.canonical_smiles),
                "target": int(source.target),
                "fragment_count": audit.fragment_count,
                "fragment_smiles": _serialize(
                    [fragment.canonical_smiles for fragment in audit.fragments]
                ),
                "fragment_heavy_atom_counts": _serialize(
                    [fragment.heavy_atom_count for fragment in audit.fragments]
                ),
                "fragment_formal_charges": _serialize(
                    [fragment.formal_charge for fragment in audit.fragments]
                ),
                "total_heavy_atom_count": audit.total_heavy_atom_count,
                "total_formal_charge": audit.total_formal_charge,
                "candidate_parent_smiles": audit.candidate_parent.canonical_smiles,
                "candidate_parent_heavy_atom_count": (audit.candidate_parent.heavy_atom_count),
                "candidate_parent_formal_charge": audit.candidate_parent.formal_charge,
                "candidate_parent_scaffold": parent_scaffold,
                "removed_fragment_smiles": _serialize(
                    [fragment.canonical_smiles for fragment in audit.removed_fragments]
                ),
                "removed_fragment_heavy_atom_counts": _serialize(
                    [fragment.heavy_atom_count for fragment in audit.removed_fragments]
                ),
                "removed_fragment_formal_charges": _serialize(
                    [fragment.formal_charge for fragment in audit.removed_fragments]
                ),
                "removed_fragment_categories": _serialize(list(audit.removed_fragment_categories)),
            }
        )
    return pd.DataFrame(rows, columns=ANALYSIS_COLUMNS)


def _split_statistics(rows: pd.DataFrame) -> dict[str, Any]:
    total = len(rows)
    disconnected = int((rows["fragment_count"] > 1).sum()) if total else 0
    distribution = Counter(int(value) for value in rows.get("fragment_count", []))
    return {
        "total_molecule_count": total,
        "connected_molecule_count": total - disconnected,
        "disconnected_molecule_count": disconnected,
        "disconnected_fraction": disconnected / total if total else 0.0,
        "fragment_count_distribution": {
            str(count): int(frequency) for count, frequency in sorted(distribution.items())
        },
        **_identity_statistics(rows),
        "parent_collision_statistics": _collision_statistics(rows),
    }


def _identity_statistics(rows: pd.DataFrame) -> dict[str, int]:
    source_rows = len(rows)
    unique_original = int(rows["canonical_smiles"].nunique()) if source_rows else 0
    unique_parent = int(rows["candidate_parent_smiles"].nunique()) if source_rows else 0
    return {
        "source_row_count": source_rows,
        "unique_original_canonical_smiles_count": unique_original,
        "unique_candidate_parent_smiles_count": unique_parent,
        "structures_collapsed_by_parent_selection": unique_original - unique_parent,
    }


def _collision_statistics(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {
            "duplicate_candidate_parent_smiles": [],
            "parent_collision_group_count": 0,
            "maximum_collision_group_size": 0,
            "multiple_original_structure_parent_count": 0,
            "candidate_parents_with_multiple_original_structures": [],
            "label_conflict_group_count": 0,
            "label_conflict_parent_smiles": [],
        }
    grouped = rows.groupby("candidate_parent_smiles", sort=True, dropna=False)
    collisions = [(str(parent), group) for parent, group in grouped if len(group) > 1]
    multiple_original = [
        parent for parent, group in collisions if group["canonical_smiles"].nunique() > 1
    ]
    conflicts = [parent for parent, group in collisions if group["target"].nunique() > 1]
    return {
        "duplicate_candidate_parent_smiles": [parent for parent, _ in collisions],
        "parent_collision_group_count": len(collisions),
        "maximum_collision_group_size": max((len(group) for _, group in collisions), default=0),
        "multiple_original_structure_parent_count": len(multiple_original),
        "candidate_parents_with_multiple_original_structures": multiple_original,
        "label_conflict_group_count": len(conflicts),
        "label_conflict_parent_smiles": conflicts,
    }


def _parent_collision_records(rows: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for parent, group in rows.groupby("candidate_parent_smiles", sort=True, dropna=False):
        if len(group) < 2:
            continue
        ordered = _ordered_rows(group)
        records.append(
            {
                "candidate_parent_smiles": str(parent),
                "collision_group_size": len(ordered),
                "has_label_conflict": bool(ordered["target"].nunique() > 1),
                "splits_present": _serialize(_ordered_splits(ordered["split"])),
                "molecule_ids": _serialize(ordered["molecule_id"].tolist()),
                "original_smiles": _serialize(ordered["canonical_smiles"].tolist()),
                "targets": _serialize([int(value) for value in ordered["target"]]),
            }
        )
    return records


def _label_conflict_group_details(rows: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for parent, group in rows.groupby("candidate_parent_smiles", sort=True, dropna=False):
        if group["target"].nunique() < 2:
            continue
        ordered = _ordered_rows(group)
        records.append(
            {
                "candidate_parent_smiles": str(parent),
                "collision_group_size": len(ordered),
                "splits_present": _ordered_splits(ordered["split"]),
                "identities": [
                    {
                        "split": str(row.split),
                        "molecule_id": str(row.molecule_id),
                        "canonical_smiles": str(row.canonical_smiles),
                        "target": int(row.target),
                    }
                    for row in ordered.itertuples(index=False)
                ],
            }
        )
    return records


def _parent_overlap_records(train: pd.DataFrame, validation: pd.DataFrame) -> list[dict[str, Any]]:
    shared = sorted(
        set(train["candidate_parent_smiles"]) & set(validation["candidate_parent_smiles"])
    )
    records: list[dict[str, Any]] = []
    for parent in shared:
        train_group = _ordered_rows(train.loc[train["candidate_parent_smiles"] == parent])
        validation_group = _ordered_rows(
            validation.loc[validation["candidate_parent_smiles"] == parent]
        )
        records.append(
            {
                "candidate_parent_smiles": parent,
                "train_row_count": len(train_group),
                "validation_row_count": len(validation_group),
                "has_label_conflict": bool(
                    pd.concat([train_group, validation_group])["target"].nunique() > 1
                ),
                "train_identities": _serialize(_identity_records(train_group)),
                "validation_identities": _serialize(_identity_records(validation_group)),
            }
        )
    return records


def _scaffold_overlap_records(
    train: pd.DataFrame, validation: pd.DataFrame
) -> list[dict[str, Any]]:
    shared = sorted(
        set(train["candidate_parent_scaffold"]) & set(validation["candidate_parent_scaffold"])
    )
    records: list[dict[str, Any]] = []
    for scaffold in shared:
        train_group = _ordered_rows(train.loc[train["candidate_parent_scaffold"] == scaffold])
        validation_group = _ordered_rows(
            validation.loc[validation["candidate_parent_scaffold"] == scaffold]
        )
        records.append(
            {
                "candidate_parent_scaffold": scaffold,
                "train_row_count": len(train_group),
                "validation_row_count": len(validation_group),
                "train_parent_smiles": _serialize(
                    sorted(set(train_group["candidate_parent_smiles"]))
                ),
                "validation_parent_smiles": _serialize(
                    sorted(set(validation_group["candidate_parent_smiles"]))
                ),
                "train_identities": _serialize(_identity_records(train_group)),
                "validation_identities": _serialize(_identity_records(validation_group)),
            }
        )
    return records


def _identity_records(rows: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "molecule_id": str(row.molecule_id),
            "canonical_smiles": str(row.canonical_smiles),
            "target": int(row.target),
        }
        for row in rows.itertuples(index=False)
    ]


def _ordered_rows(rows: pd.DataFrame) -> pd.DataFrame:
    split_order = pd.Categorical(rows["split"], categories=["train", "validation"], ordered=True)
    return (
        rows.assign(_split_order=split_order)
        .sort_values(
            ["_split_order", "molecule_id", "canonical_smiles", "target"],
            kind="stable",
        )
        .drop(columns="_split_order")
    )


def _ordered_splits(values: pd.Series) -> list[str]:
    present = set(str(value) for value in values)
    return [split for split in ("train", "validation") if split in present]


def _parent_scaffold(parent_smiles: str) -> str:
    molecule = Chem.MolFromSmiles(parent_smiles)
    if molecule is None:  # pragma: no cover - candidate parent originated from RDKit
        raise ValueError(f"Candidate parent SMILES could not be parsed: {parent_smiles}")
    scaffold = safe_murcko_scaffold(molecule).scaffold
    return scaffold or f"ACYCLIC::{parent_smiles}"


def _serialize(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _validate_development_config(config: ChempropExperimentConfig) -> None:
    if config.endpoint_id != "bbb_martins" or config.task_type != "binary_classification":
        raise ValueError("Fragment audit supports only binary BBB_Martins development data.")
    names = {split: config.split_files.get(split) for split in ("train", "validation", "test")}
    if any(not name for name in names.values()):
        raise LockedTestAccessError("Config must identify train, validation, and test files.")
    if len(set(names.values())) != 3:
        raise LockedTestAccessError(
            "Train, validation, and locked-test artifacts must be distinct."
        )


def _reject_test_named_config(config_path: str | Path) -> None:
    if Path(config_path).name.lower() in {"test.csv", "locked.csv"}:
        raise LockedTestAccessError("A test artifact cannot be used as audit config.")


@contextmanager
def _deny_test_artifact_access(
    config: ChempropExperimentConfig,
) -> Iterator[TestArtifactAccessGuard]:
    """Block resolution, metadata inspection, and reads of the configured test artifact."""

    guard = TestArtifactAccessGuard(
        test_path=Path(os.path.abspath(config.prepared_root / config.split_files["test"]))
    )
    original_read_bytes = Path.read_bytes
    original_read_text = Path.read_text
    original_path_open = Path.open
    original_resolve = Path.resolve
    original_exists = Path.exists
    original_is_file = Path.is_file
    original_is_dir = Path.is_dir
    original_stat = Path.stat
    original_builtin_open = builtins.open
    original_read_csv = pd.read_csv

    def guarded_read_bytes(path: Path) -> bytes:
        guard.reject(path)
        return original_read_bytes(path)

    def guarded_read_text(path: Path, *args: Any, **kwargs: Any) -> str:
        guard.reject(path)
        return original_read_text(path, *args, **kwargs)

    def guarded_path_open(path: Path, *args: Any, **kwargs: Any):
        guard.reject(path)
        return original_path_open(path, *args, **kwargs)

    def guarded_resolve(path: Path, *args: Any, **kwargs: Any) -> Path:
        guard.reject(path)
        return original_resolve(path, *args, **kwargs)

    def guarded_exists(path: Path) -> bool:
        guard.reject(path)
        return original_exists(path)

    def guarded_is_file(path: Path) -> bool:
        guard.reject(path)
        return original_is_file(path)

    def guarded_is_dir(path: Path) -> bool:
        guard.reject(path)
        return original_is_dir(path)

    def guarded_stat(path: Path, *args: Any, **kwargs: Any):
        guard.reject(path)
        return original_stat(path, *args, **kwargs)

    def guarded_builtin_open(file: Any, *args: Any, **kwargs: Any):
        guard.reject(file)
        return original_builtin_open(file, *args, **kwargs)

    def guarded_read_csv(source: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
        guard.reject(source)
        return original_read_csv(source, *args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(patch.object(Path, "read_bytes", guarded_read_bytes))
        stack.enter_context(patch.object(Path, "read_text", guarded_read_text))
        stack.enter_context(patch.object(Path, "open", guarded_path_open))
        stack.enter_context(patch.object(Path, "resolve", guarded_resolve))
        stack.enter_context(patch.object(Path, "exists", guarded_exists))
        stack.enter_context(patch.object(Path, "is_file", guarded_is_file))
        stack.enter_context(patch.object(Path, "is_dir", guarded_is_dir))
        stack.enter_context(patch.object(Path, "stat", guarded_stat))
        stack.enter_context(patch.object(builtins, "open", guarded_builtin_open))
        stack.enter_context(patch.object(pd, "read_csv", guarded_read_csv))
        yield guard


def _write_audit_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    analysis: AuditAnalysis,
    *,
    overwrite: bool,
) -> None:
    existing = [
        output_dir / filename for filename in OUTPUT_FILENAMES if (output_dir / filename).exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            f"Audit outputs already exist; pass --overwrite to replace known files: {existing}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fragment_audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    analysis.disconnected_molecules.to_csv(output_dir / "disconnected_molecules.csv", index=False)
    analysis.parent_collision_groups.to_csv(output_dir / "parent_collision_groups.csv", index=False)
    analysis.train_validation_parent_overlap.to_csv(
        output_dir / "train_validation_parent_overlap.csv", index=False
    )
    analysis.train_validation_parent_scaffold_overlap.to_csv(
        output_dir / "train_validation_parent_scaffold_overlap.csv", index=False
    )


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit if len(commit) == 40 else None


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_disconnected_fragment_audit(
        args.config,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
