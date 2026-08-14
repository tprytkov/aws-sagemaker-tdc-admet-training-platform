"""Training-space similarity and scaffold-familiarity calculations."""

from __future__ import annotations

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from admet_platform.data.scaffolds import safe_murcko_scaffold


def applicability_domain(
    train_smiles: list[str], query_smiles: list[str], similarity_threshold: float = 0.40,
) -> pd.DataFrame:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    train_molecules = [_molecule(smiles) for smiles in train_smiles]
    train_fps = [generator.GetFingerprint(molecule) for molecule in train_molecules]
    train_scaffolds = {_scaffold(molecule, smiles) for molecule, smiles in zip(train_molecules, train_smiles)}
    rows = []
    for smiles in query_smiles:
        molecule = _molecule(smiles)
        fp = generator.GetFingerprint(molecule)
        maximum = max(DataStructs.BulkTanimotoSimilarity(fp, train_fps), default=0.0)
        familiar = _scaffold(molecule, smiles) in train_scaffolds
        rows.append({
            "canonical_smiles": smiles,
            "maximum_train_tanimoto": float(maximum),
            "scaffold_in_training": bool(familiar),
            "applicability_domain": "in_domain" if maximum >= similarity_threshold and familiar else "out_of_domain",
        })
    return pd.DataFrame(rows)


def _molecule(smiles: str) -> Chem.Mol:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES for applicability analysis: {smiles}")
    return molecule


def _scaffold(molecule: Chem.Mol, smiles: str) -> str:
    value = safe_murcko_scaffold(molecule).scaffold
    return value or f"ACYCLIC::{smiles}"
