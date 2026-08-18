"""Read-only BBB train/validation adapter for GMC-MPNN development.

The public loaders deliberately support only the existing training and validation
splits. They never construct, open, or hash a test-data path.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from rdkit import Chem

from admet_platform.chemprop.config import ChempropExperimentConfig
from admet_platform.data.scaffolds import safe_murcko_scaffold


DevelopmentSplit = Literal["train", "validation"]
DEVELOPMENT_SPLITS: tuple[DevelopmentSplit, ...] = ("train", "validation")
REQUIRED_COLUMNS = frozenset({"molecule_id", "canonical_smiles", "target", "split"})


@dataclass(frozen=True)
class SplitProvenance:
    """Immutable source and content facts for one development split."""

    source_path: Path
    expected_sha256: str
    observed_sha256: str
    row_count: int
    label_counts: dict[str, int]


@dataclass(frozen=True)
class DevelopmentProvenance:
    """Train/validation-only provenance suitable for later run records."""

    split_manifest_path: Path
    split_manifest_sha256: str
    split_manifest_id: str
    splits: dict[str, SplitProvenance]


@dataclass(frozen=True)
class DevelopmentLeakageReport:
    """Observed overlap without dropping rows or changing split membership."""

    exact_canonical_smiles: tuple[str, ...]
    murcko_scaffolds: tuple[str, ...]

    @property
    def exact_canonical_smiles_count(self) -> int:
        return len(self.exact_canonical_smiles)

    @property
    def murcko_scaffold_count(self) -> int:
        return len(self.murcko_scaffolds)

    @property
    def has_overlap(self) -> bool:
        return bool(self.exact_canonical_smiles or self.murcko_scaffolds)


@dataclass(frozen=True)
class BBBDevelopmentData:
    """Validated BBB_Martins development rows and their integrity metadata."""

    train: pd.DataFrame
    validation: pd.DataFrame
    leakage: DevelopmentLeakageReport
    provenance: DevelopmentProvenance


def load_bbb_development_data(config: ChempropExperimentConfig) -> BBBDevelopmentData:
    """Load and validate only configured BBB_Martins train/validation artifacts.

    The existing Chemprop configuration supplies the prepared root, file names,
    manifest location, manifest identity, and dotted hash keys. Only the `train`
    and `validation` entries are selected from those mappings.
    """

    if config.endpoint_id != "bbb_martins" or config.task_type != "binary_classification":
        raise ValueError("The GMC-MPNN development adapter supports only binary BBB_Martins.")

    manifest_bytes = config.split_manifest.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    configured_manifest_sha256 = config.raw.get("split_manifest_sha256")
    if configured_manifest_sha256 and manifest_sha256 != configured_manifest_sha256:
        raise ValueError("Split manifest SHA-256 does not match the endpoint configuration.")
    manifest = json.loads(manifest_bytes)
    manifest_id = str(manifest.get("split_manifest_id", ""))
    configured_manifest_id = config.raw.get("split_manifest_id")
    if configured_manifest_id and manifest_id != configured_manifest_id:
        raise ValueError("Split manifest identity does not match the endpoint configuration.")

    frames: dict[str, pd.DataFrame] = {}
    split_provenance: dict[str, SplitProvenance] = {}
    for split in DEVELOPMENT_SPLITS:
        expected_sha256 = str(
            _resolve_manifest_key(manifest, config.split_hash_keys[split])
        )
        source_path = config.prepared_root / config.split_files[split]
        frame, observed_sha256 = _read_and_validate_split(source_path, split)
        if observed_sha256 != expected_sha256:
            raise ValueError(
                f"Prepared {split} SHA-256 does not match the frozen manifest."
            )
        frames[split] = frame
        split_provenance[split] = SplitProvenance(
            source_path=source_path,
            expected_sha256=expected_sha256,
            observed_sha256=observed_sha256,
            row_count=len(frame),
            label_counts=_label_counts(frame),
        )

    leakage = _development_leakage_report(frames["train"], frames["validation"])
    return BBBDevelopmentData(
        train=frames["train"],
        validation=frames["validation"],
        leakage=leakage,
        provenance=DevelopmentProvenance(
            split_manifest_path=config.split_manifest,
            split_manifest_sha256=manifest_sha256,
            split_manifest_id=manifest_id,
            splits=split_provenance,
        ),
    )


def load_bbb_development_split(
    path: str | Path, *, split: DevelopmentSplit | str
) -> pd.DataFrame:
    """Load one allowed development split; reject test before touching a path."""

    validated_split = _require_development_split(split)
    frame, _ = _read_and_validate_split(Path(path), validated_split)
    return frame


def _require_development_split(split: str) -> DevelopmentSplit:
    if split == "test":
        raise ValueError(
            "The locked/previously evaluated BBB test data is outside the Phase-1 "
            "development loader. Only 'train' and 'validation' are allowed."
        )
    if split not in DEVELOPMENT_SPLITS:
        raise ValueError("GMC-MPNN development split must be 'train' or 'validation'.")
    return split  # type: ignore[return-value]


def _read_and_validate_split(
    path: Path, split: DevelopmentSplit
) -> tuple[pd.DataFrame, str]:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    frame = pd.read_csv(io.BytesIO(payload))
    return _validate_frame(frame, split), digest


def _validate_frame(frame: pd.DataFrame, split: DevelopmentSplit) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Prepared {split} split is missing columns: {missing}")

    if frame.duplicated(keep=False).any():
        raise ValueError(f"Prepared {split} split contains exact duplicate rows.")
    _require_nonempty(frame, "molecule_id", split)
    _require_nonempty(frame, "canonical_smiles", split)

    output = frame.copy()
    numeric_targets = pd.to_numeric(output["target"], errors="coerce")
    if numeric_targets.isna().any():
        raise ValueError(f"Prepared {split} split contains non-numeric targets.")
    if not numeric_targets.isin([0, 1]).all():
        raise ValueError("BBB targets must be binary 0/1 values.")
    output["target"] = numeric_targets.astype("int64")

    observed_splits = set(output["split"].astype(str).str.strip().str.lower())
    expected_labels = {"validation", "valid", "val"} if split == "validation" else {"train"}
    if not observed_splits or not observed_splits.issubset(expected_labels):
        raise ValueError(
            f"Prepared {split} split contains unexpected split labels: "
            f"{sorted(observed_splits)}"
        )

    _reject_conflicting_labels(output, split)
    if output["molecule_id"].duplicated().any():
        raise ValueError(f"Prepared {split} split contains duplicate molecule IDs.")
    if output["canonical_smiles"].duplicated().any():
        raise ValueError(f"Prepared {split} split contains duplicate canonical SMILES.")
    return _with_verified_scaffolds(output)


def _require_nonempty(frame: pd.DataFrame, column: str, split: str) -> None:
    values = frame[column]
    if values.isna().any() or values.astype(str).str.strip().eq("").any():
        raise ValueError(f"Prepared {split} split contains missing {column} values.")


def _reject_conflicting_labels(frame: pd.DataFrame, split: str) -> None:
    for identity_column in ("molecule_id", "canonical_smiles"):
        label_counts = frame.groupby(identity_column, dropna=False)["target"].nunique()
        if (label_counts > 1).any():
            raise ValueError(
                f"Prepared {split} split contains conflicting labels for {identity_column}."
            )


def _with_verified_scaffolds(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    scaffolds: list[str] = []
    for smiles in output["canonical_smiles"].astype(str):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"Invalid canonical SMILES in prepared split: {smiles}")
        canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        if canonical != smiles:
            raise ValueError("Prepared canonical SMILES changed under the pinned RDKit version.")
        scaffold = safe_murcko_scaffold(molecule).scaffold
        scaffolds.append(scaffold or f"ACYCLIC::{canonical}")
    output["murcko_scaffold"] = scaffolds
    return output


def _development_leakage_report(
    train: pd.DataFrame, validation: pd.DataFrame
) -> DevelopmentLeakageReport:
    exact = sorted(set(train["canonical_smiles"]) & set(validation["canonical_smiles"]))
    scaffolds = sorted(set(train["murcko_scaffold"]) & set(validation["murcko_scaffold"]))
    return DevelopmentLeakageReport(
        exact_canonical_smiles=tuple(exact),
        murcko_scaffolds=tuple(scaffolds),
    )


def _label_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["target"].value_counts().reindex([0, 1], fill_value=0)
    return {str(label): int(counts[label]) for label in (0, 1)}


def _resolve_manifest_key(value: Any, dotted_key: str) -> Any:
    current = value
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ValueError(f"Split manifest is missing hash key: {dotted_key}")
        current = current[part]
    return current


__all__ = [
    "BBBDevelopmentData",
    "DEVELOPMENT_SPLITS",
    "DevelopmentLeakageReport",
    "DevelopmentProvenance",
    "REQUIRED_COLUMNS",
    "SplitProvenance",
    "load_bbb_development_data",
    "load_bbb_development_split",
]
