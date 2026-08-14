import pandas as pd

from admet_platform.chemprop.pipeline_check import (
    balanced_binary_subset,
    endpoint_balanced_sparse_subset,
)


def test_sparse_subset_preserves_observations_for_every_endpoint() -> None:
    frame = pd.DataFrame({
        "canonical_smiles": ["C", "CC", "CCC", "CCCC"],
        "left": [1.0, float("nan"), 2.0, float("nan")],
        "right": [float("nan"), 3.0, float("nan"), 4.0],
    })
    subset = endpoint_balanced_sparse_subset(frame, ["left", "right"], 1)
    assert subset["left"].notna().any()
    assert subset["right"].notna().any()
    assert subset[["left", "right"]].isna().any().any()


def test_binary_subset_is_balanced() -> None:
    frame = pd.DataFrame({
        "canonical_smiles": ["C", "CC", "CCC", "CCCC", "CCCCC", "CCCCCC"],
        "target": [0, 0, 0, 1, 1, 1],
    })
    subset = balanced_binary_subset(frame, 2)
    assert subset["target"].value_counts().to_dict() == {0: 2, 1: 2}
