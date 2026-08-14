"""Deterministic real-data subsets for bounded train/validation execution checks."""

from __future__ import annotations

import pandas as pd


def endpoint_balanced_sparse_subset(
    frame: pd.DataFrame, endpoints: list[str], per_endpoint: int
) -> pd.DataFrame:
    if per_endpoint <= 0:
        raise ValueError("per_endpoint must be positive.")
    selected: set[int] = set()
    for endpoint in endpoints:
        candidates = frame.loc[frame[endpoint].notna()].sort_values("canonical_smiles")
        if candidates.empty:
            raise ValueError(f"No observed labels are available for {endpoint}.")
        selected.update(candidates.head(per_endpoint).index.tolist())
    return frame.loc[sorted(selected)].reset_index(drop=True)


def balanced_binary_subset(frame: pd.DataFrame, per_class: int) -> pd.DataFrame:
    if per_class <= 0:
        raise ValueError("per_class must be positive.")
    parts = []
    for label in (0, 1):
        candidates = frame.loc[frame["target"].astype(int) == label].sort_values(
            "canonical_smiles"
        )
        if candidates.empty:
            raise ValueError(f"No BBB rows are available for class {label}.")
        parts.append(candidates.head(per_class))
    return pd.concat(parts).sort_values("canonical_smiles").reset_index(drop=True)


__all__ = ["balanced_binary_subset", "endpoint_balanced_sparse_subset"]
