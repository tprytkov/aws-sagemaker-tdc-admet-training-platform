from pathlib import Path

import pandas as pd
import pytest

from admet_platform.chemprop.comparison import align_bbb_predictions
from admet_platform.chemprop.config import load_chemprop_config
from admet_platform.chemprop.export import build_export_manifest


ROOT = Path(__file__).resolve().parents[1]


def test_separate_export_requirements(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    resolved = tmp_path / "resolved.json"
    scaler = tmp_path / "scaler.json"
    calibrator = tmp_path / "calibrator.json"
    for path in (checkpoint, resolved, scaler, calibrator):
        path.write_text(path.name, encoding="utf-8")
    regression = load_chemprop_config(
        ROOT / "configs/chemprop/multitask_admet_regression.yaml"
    )
    bbb = load_chemprop_config(ROOT / "configs/chemprop/bbb_martins.yaml")
    regression_manifest = build_export_manifest(
        regression, checkpoint, resolved, "a" * 64, scaler=scaler,
        endpoint_artifacts={
            endpoint: {
                "predictions": resolved,
                "metadata": resolved,
                "applicability": resolved,
                "uncertainty": resolved,
            }
            for endpoint in regression.tasks
        },
    )
    bbb_manifest = build_export_manifest(
        bbb, checkpoint, resolved, "b" * 64, calibrator=calibrator, threshold=0.42,
        class_counts={"0": 10, "1": 20},
    )
    assert set(regression_manifest["endpoints"]) == set(regression.tasks)
    assert "scaler_sha256" in regression_manifest and "calibrator_sha256" not in regression_manifest
    assert "calibrator_sha256" in bbb_manifest and "scaler_sha256" not in bbb_manifest
    with pytest.raises(ValueError, match="calibrator"):
        build_export_manifest(bbb, checkpoint, resolved, "b" * 64)


def test_matched_comparison_requires_identical_molecules_and_labels() -> None:
    left = pd.DataFrame({"canonical_smiles": ["CCO"], "target": [1], "probability": [0.8]})
    right = pd.DataFrame({"canonical_smiles": ["CCO"], "target": [1], "probability": [0.7]})
    assert len(align_bbb_predictions(left, right)) == 1
    with pytest.raises(ValueError, match="membership differs"):
        align_bbb_predictions(left, right.assign(canonical_smiles="CCC"))
