"""Inexpensive validation-only conventional baselines."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from admet_platform.chemprop.metrics import classification_metrics, regression_metrics


DESCRIPTORS = (
    Descriptors.MolWt, Crippen.MolLogP, Descriptors.TPSA, Lipinski.NumHDonors,
    Lipinski.NumHAcceptors, Lipinski.NumRotatableBonds, Descriptors.RingCount,
    Descriptors.FractionCSP3, Descriptors.MolMR, Descriptors.HeavyAtomCount,
)


def run_regression_baselines(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    include_esol: bool = True,
    evaluation_targets: np.ndarray | None = None,
    inverse_prediction: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, Any]:
    y_train = train["target"].to_numpy(dtype=float)
    y_valid_model_space = validation["target"].to_numpy(dtype=float)
    y_valid = y_valid_model_space if evaluation_targets is None else evaluation_targets
    inverse = inverse_prediction or (lambda values: values)
    median_prediction = inverse(np.full(len(validation), np.median(y_train)))
    results: dict[str, Any] = {
        "training_median": regression_metrics(y_valid, median_prediction),
    }
    if include_esol:
        esol_prediction = np.asarray([esol(smiles) for smiles in validation["canonical_smiles"]])
        results["esol"] = regression_metrics(y_valid, esol_prediction)
    for name, feature_func in (("morgan_ridge", _morgan), ("rdkit_descriptor_ridge", _descriptors)):
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ])
        model.fit(feature_func(train), y_train)
        prediction = inverse(model.predict(feature_func(validation)))
        results[name] = regression_metrics(y_valid, prediction)
    return results


def run_classification_baselines(train: pd.DataFrame, validation: pd.DataFrame) -> dict[str, Any]:
    y_train = train["target"].to_numpy(dtype=int)
    y_valid = validation["target"].to_numpy(dtype=int)
    prevalence = float(y_train.mean())
    majority = float(int(prevalence >= 0.5))
    results: dict[str, Any] = {
        "majority": classification_metrics(y_valid, np.full(len(y_valid), majority)),
        "prevalence": classification_metrics(y_valid, np.full(len(y_valid), prevalence)),
    }
    for name, feature_func in (("morgan_logistic", _morgan), ("rdkit_descriptor_logistic", _descriptors)):
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ])
        model.fit(feature_func(train), y_train)
        probability = model.predict_proba(feature_func(validation))[:, 1]
        results[name] = classification_metrics(y_valid, probability)
    return results


def esol(smiles: str) -> float:
    """Delaney ESOL implementation in log(mol/L)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES for ESOL: {smiles}")
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    rotors = Lipinski.NumRotatableBonds(mol)
    atom_count = max(1, mol.GetNumAtoms())
    aromatic_fraction = sum(atom.GetIsAromatic() for atom in mol.GetAtoms()) / atom_count
    # Original Delaney coefficients, evaluated with the matching RDKit descriptors.
    return float(0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rotors - 0.74 * aromatic_fraction)


def _morgan(frame: pd.DataFrame) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    rows = []
    for smiles in frame["canonical_smiles"]:
        fp = generator.GetFingerprint(Chem.MolFromSmiles(str(smiles)))
        array = np.zeros(2048, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, array)
        rows.append(array)
    return np.asarray(rows)


def _descriptors(frame: pd.DataFrame) -> np.ndarray:
    return np.asarray([
        [float(function(Chem.MolFromSmiles(str(smiles)))) for function in DESCRIPTORS]
        for smiles in frame["canonical_smiles"]
    ])
