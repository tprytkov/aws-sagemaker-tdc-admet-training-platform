"""Matched BBB prediction contract; model loading is intentionally external."""

from __future__ import annotations

import pandas as pd


def align_bbb_predictions(chemprop: pd.DataFrame, chemberta: pd.DataFrame) -> pd.DataFrame:
    required = {"canonical_smiles", "target", "probability"}
    for provider, frame in (("chemprop", chemprop), ("chemberta", chemberta)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{provider} predictions are missing columns: {missing}")
        if frame["canonical_smiles"].duplicated().any():
            raise ValueError(f"{provider} predictions contain duplicate canonical SMILES.")
    merged = chemprop.merge(
        chemberta, on="canonical_smiles", how="outer", validate="one_to_one",
        suffixes=("_chemprop", "_chemberta"), indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("Chemprop and frozen ChemBERTa molecule membership differs.")
    if not merged["target_chemprop"].eq(merged["target_chemberta"]).all():
        raise ValueError("Chemprop and frozen ChemBERTa labels differ.")
    return merged.drop(columns="_merge")
