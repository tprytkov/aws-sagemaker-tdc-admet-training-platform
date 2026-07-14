"""RDKit descriptor and fingerprint featurization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors


FeatureType = Literal["descriptors", "morgan"]
DEFAULT_MORGAN_RADIUS = 2
DEFAULT_MORGAN_BITS = 2048
DESCRIPTOR_NAMES = [
    "molecular_weight",
    "logp",
    "tpsa",
    "h_bond_donors",
    "h_bond_acceptors",
    "rotatable_bonds",
    "ring_count",
    "aromatic_ring_count",
    "heavy_atom_count",
    "fraction_csp3",
]
REJECTED_ROW_COLUMNS = ["row_index", "canonical_smiles", "rejection_reason"]


@dataclass(frozen=True)
class FeatureConfig:
    """Serializable configuration for local molecular featurization."""

    feature_type: FeatureType
    smiles_column: str = "canonical_smiles"
    target_column: str = "target"
    morgan_radius: int = DEFAULT_MORGAN_RADIUS
    morgan_bits: int = DEFAULT_MORGAN_BITS

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "feature_type": self.feature_type,
            "smiles_column": self.smiles_column,
            "target_column": self.target_column,
            "rdkit_version": _rdkit_version(),
        }
        if self.feature_type == "descriptors":
            payload["descriptor_names"] = list(DESCRIPTOR_NAMES)
        elif self.feature_type == "morgan":
            payload["morgan_radius"] = self.morgan_radius
            payload["morgan_bits"] = self.morgan_bits
            payload["feature_columns"] = _morgan_column_names(self.morgan_bits)
        else:
            raise ValueError("feature_type must be either 'descriptors' or 'morgan'.")
        return payload


def featurize_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    config: FeatureConfig,
    metadata_json: str | Path | None = None,
    rejected_csv: str | Path | None = None,
) -> dict[str, Any]:
    """Read a prepared CSV, featurize valid rows, and write outputs."""

    input_path = Path(input_csv)
    df = pd.read_csv(input_path)
    features_df, rejected_df, metadata = featurize_dataframe(df, config)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)

    if metadata_json is not None:
        metadata_path = Path(metadata_json)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    if rejected_csv is not None:
        rejected_path = Path(rejected_csv)
        rejected_path.parent.mkdir(parents=True, exist_ok=True)
        rejected_df.to_csv(rejected_path, index=False)

    return metadata


def featurize_dataframe(
    df: pd.DataFrame,
    config: FeatureConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Featurize a prepared ADMET DataFrame with RDKit features."""

    _validate_config(config)
    if df.empty:
        raise ValueError("Input DataFrame must not be empty.")
    if config.smiles_column not in df.columns:
        raise ValueError(f"Input DataFrame is missing required column '{config.smiles_column}'.")

    preserved_columns = _preserved_columns(df, config)
    feature_columns = _feature_column_names(config)
    accepted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []

    for row_index, row in df.iterrows():
        smiles = row[config.smiles_column]
        molecule = _parse_molecule(smiles)
        if molecule is None:
            rejected_rows.append(
                {
                    "row_index": int(row_index),
                    "canonical_smiles": "" if pd.isna(smiles) else str(smiles),
                    "rejection_reason": "invalid_or_missing_canonical_smiles",
                }
            )
            continue

        row_payload = {column: row[column] for column in preserved_columns}
        row_payload.update(_compute_features(molecule, config))
        accepted_rows.append(row_payload)

    features_df = pd.DataFrame(accepted_rows, columns=[*preserved_columns, *feature_columns])
    rejected_df = pd.DataFrame(rejected_rows, columns=REJECTED_ROW_COLUMNS)
    metadata = _metadata(config, len(df), len(features_df), len(rejected_df), feature_columns)
    return features_df, rejected_df, metadata


def _validate_config(config: FeatureConfig) -> None:
    if config.feature_type not in {"descriptors", "morgan"}:
        raise ValueError("feature_type must be either 'descriptors' or 'morgan'.")
    if config.morgan_radius < 0:
        raise ValueError("morgan_radius must be non-negative.")
    if config.morgan_bits <= 0:
        raise ValueError("morgan_bits must be positive.")


def _preserved_columns(df: pd.DataFrame, config: FeatureConfig) -> list[str]:
    excluded_feature_inputs = {"smiles"}
    return [
        column
        for column in df.columns
        if column not in excluded_feature_inputs
        and not column.startswith("morgan_")
        and column not in DESCRIPTOR_NAMES
        and column != config.target_column + "_feature"
    ]


def _parse_molecule(smiles: object) -> Chem.Mol | None:
    if pd.isna(smiles):
        return None
    smiles_text = str(smiles).strip()
    if not smiles_text:
        return None
    return Chem.MolFromSmiles(smiles_text)


def _compute_features(molecule: Chem.Mol, config: FeatureConfig) -> dict[str, float | int]:
    if config.feature_type == "descriptors":
        return {
            "molecular_weight": float(Descriptors.MolWt(molecule)),
            "logp": float(Crippen.MolLogP(molecule)),
            "tpsa": float(rdMolDescriptors.CalcTPSA(molecule)),
            "h_bond_donors": int(Lipinski.NumHDonors(molecule)),
            "h_bond_acceptors": int(Lipinski.NumHAcceptors(molecule)),
            "rotatable_bonds": int(Lipinski.NumRotatableBonds(molecule)),
            "ring_count": int(Lipinski.RingCount(molecule)),
            "aromatic_ring_count": int(rdMolDescriptors.CalcNumAromaticRings(molecule)),
            "heavy_atom_count": int(molecule.GetNumHeavyAtoms()),
            "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(molecule)),
        }

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=config.morgan_radius,
        fpSize=config.morgan_bits,
    )
    fingerprint = generator.GetFingerprint(molecule)
    return {
        column_name: int(fingerprint.GetBit(index))
        for index, column_name in enumerate(_morgan_column_names(config.morgan_bits))
    }


def _feature_column_names(config: FeatureConfig) -> list[str]:
    if config.feature_type == "descriptors":
        return list(DESCRIPTOR_NAMES)
    if config.feature_type == "morgan":
        return _morgan_column_names(config.morgan_bits)
    raise ValueError("feature_type must be either 'descriptors' or 'morgan'.")


def _morgan_column_names(bit_count: int) -> list[str]:
    width = max(4, len(str(bit_count - 1)))
    return [f"morgan_{index:0{width}d}" for index in range(bit_count)]


def _metadata(
    config: FeatureConfig,
    input_row_count: int,
    accepted_row_count: int,
    rejected_row_count: int,
    feature_columns: list[str],
) -> dict[str, Any]:
    payload = config.to_dict()
    payload.update(
        {
            "input_row_count": int(input_row_count),
            "accepted_row_count": int(accepted_row_count),
            "rejected_row_count": int(rejected_row_count),
            "n_features": int(len(feature_columns)),
        }
    )
    if config.feature_type == "descriptors":
        payload["feature_columns"] = list(feature_columns)
    return payload


def _rdkit_version() -> str | None:
    return getattr(rdBase, "rdkitVersion", None)
