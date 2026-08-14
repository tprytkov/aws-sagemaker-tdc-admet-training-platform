"""Read-only adapters for immutable prepared train/validation splits."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import pandas as pd
from rdkit import Chem

from admet_platform.chemprop.config import ChempropExperimentConfig
from admet_platform.data.scaffolds import safe_murcko_scaffold


REQUIRED_COLUMNS = {"molecule_id", "canonical_smiles", "target", "split"}


@dataclass(frozen=True)
class VerifiedDevelopmentData:
    train: pd.DataFrame
    validation: pd.DataFrame
    expected_hashes: dict[str, str]
    actual_hashes: dict[str, str]
    split_manifest_sha256: str
    label_counts: dict[str, dict[str, int]]


def load_verified_development_data(config: ChempropExperimentConfig) -> VerifiedDevelopmentData:
    """Load train/validation only. The locked test path is never opened or hashed."""

    manifest_bytes = config.split_manifest.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    configured_hash = config.raw.get("split_manifest_sha256")
    if configured_hash and manifest_hash != configured_hash:
        raise ValueError("Split manifest SHA-256 does not match the endpoint configuration.")
    manifest = json.loads(manifest_bytes)
    configured_id = config.raw.get("split_manifest_id")
    if configured_id and manifest.get("split_manifest_id") != configured_id:
        raise ValueError("Split manifest identity does not match the endpoint configuration.")
    expected = _expected_hashes(config, manifest)
    frames: dict[str, pd.DataFrame] = {}
    actual: dict[str, str] = {}
    label_counts: dict[str, dict[str, int]] = {}
    if config.tasks:
        for split in ("train", "validation"):
            endpoint_frames: dict[str, pd.DataFrame] = {}
            for endpoint in config.tasks:
                path = config.prepared_root / endpoint / config.split_files[split]
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                key = f"{endpoint}/{split}"
                if digest != expected[key]:
                    raise ValueError(f"Prepared {endpoint} {split} SHA-256 mismatch.")
                frame = pd.read_csv(path)
                _validate_frame(frame, split, "regression")
                endpoint_frames[endpoint] = frame
                actual[key] = digest
                label_counts.setdefault(endpoint, {})[split] = len(frame)
            frames[split] = _combine_regression_endpoints(endpoint_frames)
    else:
        for split in ("train", "validation"):
            path = config.prepared_root / config.split_files[split]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected[split]:
                raise ValueError(f"Prepared {split} SHA-256 does not match the frozen manifest.")
            frame = pd.read_csv(path)
            _validate_frame(frame, split, config.task_type)
            frames[split] = _with_scaffold(frame)
            actual[split] = digest
            label_counts.setdefault(config.endpoint_id, {})[split] = len(frame)
    assert_no_development_leakage(frames["train"], frames["validation"])
    _assert_manifest_leakage_status(manifest)
    return VerifiedDevelopmentData(
        train=frames["train"], validation=frames["validation"], expected_hashes=expected,
        actual_hashes=actual, split_manifest_sha256=manifest_hash, label_counts=label_counts,
    )


def _expected_hashes(
    config: ChempropExperimentConfig, manifest: dict[str, Any]
) -> dict[str, str]:
    if not config.tasks:
        return {
            split: str(_resolve_manifest_key(manifest, key))
            for split, key in config.split_hash_keys.items()
        }
    return {
        f"{endpoint}/{split}": str(_resolve_manifest_key(manifest, key))
        for endpoint, task in config.tasks.items()
        for split, key in task["split_hash_keys"].items()
    }


def _combine_regression_endpoints(
    endpoint_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Create one sparse target matrix while retaining endpoint source identifiers."""

    combined: pd.DataFrame | None = None
    for endpoint, frame in endpoint_frames.items():
        selected = frame[["canonical_smiles", "molecule_id", "target"]].rename(
            columns={"molecule_id": f"source_id__{endpoint}", "target": endpoint}
        )
        combined = selected if combined is None else combined.merge(
            selected, on="canonical_smiles", how="outer", validate="one_to_one"
        )
    assert combined is not None
    combined = combined.sort_values("canonical_smiles").reset_index(drop=True)
    combined.insert(
        0,
        "molecule_id",
        [
            f"coordinated_{hashlib.sha256(smiles.encode('utf-8')).hexdigest()[:16]}"
            for smiles in combined["canonical_smiles"]
        ],
    )
    return _with_scaffold(combined)


def assert_no_development_leakage(train: pd.DataFrame, validation: pd.DataFrame) -> None:
    canonical_overlap = set(train["canonical_smiles"]) & set(validation["canonical_smiles"])
    if canonical_overlap:
        raise ValueError("Canonical-SMILES overlap exists between train and validation.")
    scaffold_overlap = set(train["murcko_scaffold"]) & set(validation["murcko_scaffold"])
    if scaffold_overlap:
        raise ValueError("Murcko-scaffold overlap exists between train and validation.")


def _with_scaffold(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    scaffolds = []
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


def _validate_frame(frame: pd.DataFrame, split: str, task_type: str) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Prepared {split} split is missing columns: {missing}")
    values = set(frame["split"].astype(str).str.lower())
    aliases = {"validation", "valid", "val"} if split == "validation" else {split}
    if not values or not values.issubset(aliases):
        raise ValueError(f"Prepared {split} split contains unexpected split labels: {sorted(values)}")
    targets = pd.to_numeric(frame["target"], errors="coerce")
    if targets.isna().any():
        raise ValueError(f"Prepared {split} split contains non-numeric targets.")
    if task_type == "binary_classification" and not targets.isin([0, 1]).all():
        raise ValueError("BBB targets must be binary 0/1 values.")
    if frame["canonical_smiles"].duplicated().any():
        raise ValueError(f"Prepared {split} split contains canonical-SMILES duplicates.")


def _resolve_manifest_key(value: Any, dotted_key: str) -> Any:
    current = value
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ValueError(f"Split manifest is missing hash key: {dotted_key}")
        current = current[part]
    return current


def _assert_manifest_leakage_status(manifest: dict[str, Any]) -> None:
    exact = manifest.get("exact_molecule_overlap_count_between_splits")
    scaffold = manifest.get("scaffold_overlap_count_between_splits")
    if exact is not None and exact != 0:
        raise ValueError("Frozen manifest reports exact-molecule leakage.")
    if scaffold is not None and scaffold != 0:
        raise ValueError("Frozen manifest reports scaffold leakage.")
    audit = manifest.get("leakage_audit", {})
    if audit and audit.get("status") != "passed":
        raise ValueError("Frozen regression manifest did not pass its leakage audit.")
    summary = manifest.get("audit_summary", {})
    if summary and summary.get("status") != "passed":
        raise ValueError("Frozen classification manifest did not pass its leakage audit.")
    global_audit = manifest.get("global", {})
    if global_audit:
        if global_audit.get("exact_smiles_leakage_count") != 0:
            raise ValueError("Frozen regression manifest reports exact-molecule leakage.")
        if global_audit.get("scaffold_leakage_count") != 0:
            raise ValueError("Frozen regression manifest reports scaffold leakage.")


__all__ = ["VerifiedDevelopmentData", "assert_no_development_leakage", "load_verified_development_data"]
