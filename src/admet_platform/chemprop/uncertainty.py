"""Per-endpoint ensemble uncertainty from independently initialized Chemprop seeds."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def aggregate_endpoint_seed_predictions(
    predictions_by_seed: Mapping[int, pd.DataFrame],
    *,
    prediction_column: str = "prediction",
) -> pd.DataFrame:
    """Require identical membership and return ensemble mean/std without confidence claims."""

    if not predictions_by_seed:
        raise ValueError("At least one seed prediction frame is required.")
    seeds: Sequence[int] = sorted(predictions_by_seed)
    required = {"molecule_id", "canonical_smiles", "target", prediction_column}
    reference = predictions_by_seed[seeds[0]].sort_values("canonical_smiles").reset_index(drop=True)
    if missing := required - set(reference):
        raise ValueError(f"Seed predictions are missing columns: {sorted(missing)}")
    matrix = []
    for seed in seeds:
        frame = predictions_by_seed[seed].sort_values("canonical_smiles").reset_index(drop=True)
        if not frame[["canonical_smiles", "target"]].equals(
            reference[["canonical_smiles", "target"]]
        ):
            raise ValueError("Seed prediction membership or labels differ.")
        matrix.append(frame[prediction_column].to_numpy(dtype=float))
    values = np.vstack(matrix)
    output = reference[["molecule_id", "canonical_smiles", "target"]].copy()
    output["ensemble_mean"] = values.mean(axis=0)
    output["ensemble_std"] = values.std(axis=0, ddof=1 if len(seeds) > 1 else 0)
    output["seed_count"] = len(seeds)
    output["uncertainty_interpretation"] = "seed_variability_not_calibrated_confidence_interval"
    return output


__all__ = ["aggregate_endpoint_seed_predictions"]
