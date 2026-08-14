"""Network-free synthetic CPU smoke runs for both Chemprop task types."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from admet_platform.chemprop.config import load_chemprop_config
from admet_platform.chemprop.training import train_graph_model


def run_synthetic_smoke(
    task: str,
    output_dir: str | Path,
    seed: int = 13,
    accelerator: str | None = None,
) -> dict:
    if task in {"regression", "multitask_regression"}:
        config = load_chemprop_config("configs/chemprop/multitask_admet_regression.yaml")
    elif task == "binary_classification":
        config = load_chemprop_config("configs/chemprop/bbb_martins.yaml")
        targets = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    else:
        raise ValueError("task must be regression or binary_classification")
    smiles = [
        "CCO", "CCN", "CCC", "CCCl", "c1ccccc1", "c1ccncc1",
        "C1CCCCC1", "C1CCNCC1", "CC(=O)O", "CCS", "COC", "CN(C)C",
    ]
    frame = pd.DataFrame({
        "molecule_id": [f"synthetic_{index}" for index in range(len(smiles))],
        "canonical_smiles": smiles,
    })
    if config.tasks:
        frame["caco2_wang"] = [-5.2, -4.8, -5.5, -6.0, -4.5, -5.1, -5.7, -4.9, -5.3, -5.0, -4.7, -5.6]
        frame["lipophilicity_astrazeneca"] = [1.0, 0.8, 1.5, 2.0, 2.7, 1.8, 2.3, 1.2, 0.4, 1.1, 2.1, 1.6]
        frame["solubility_aqsoldb"] = [-0.2, -0.8, -1.1, -1.7, -2.2, -2.8, -3.0, -3.5, -0.5, -1.4, -2.5, -3.2]
        frame["ppbr_az"] = [40.0, 55.0, 62.0, 71.0, 85.0, 90.0, 94.0, 76.0, 45.0, 68.0, 88.0, 80.0]
        frame["vdss_lombardo"] = [0.2, 0.4, 0.8, 1.2, 2.0, 3.5, 5.0, 0.6, 0.3, 0.9, 2.5, 4.0]
        # Exercise Chemprop's missing-target mask on both development splits.
        for offset, endpoint in enumerate(config.tasks):
            frame.loc[(frame.index + offset) % 4 == 0, endpoint] = float("nan")
    else:
        frame["target"] = targets
    train = frame.iloc[:8].reset_index(drop=True)
    validation = frame.iloc[8:].reset_index(drop=True)
    return train_graph_model(
        config, train, validation, output_dir, seed=seed, smoke=True,
        accelerator_override=accelerator,
    )
