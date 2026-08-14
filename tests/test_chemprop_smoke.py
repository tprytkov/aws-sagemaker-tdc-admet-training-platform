from pathlib import Path

import pytest

from admet_platform.chemprop.smoke import run_synthetic_smoke


@pytest.mark.parametrize("task", ["multitask_regression", "binary_classification"])
def test_synthetic_cpu_smoke(task: str, tmp_path: Path) -> None:
    result = run_synthetic_smoke(task, tmp_path / task, seed=13)
    assert Path(result["checkpoint"]).is_file()
    if task == "multitask_regression":
        assert result["target_scaler"]["fit_split"] == "train"
        assert set(result["validation_metrics"]) == {
            "caco2_wang", "lipophilicity_astrazeneca", "solubility_aqsoldb", "ppbr_az",
            "vdss_lombardo",
        }
        assert all(Path(path).is_file() for path in result["validation_prediction_files"].values())
        assert all(Path(path).is_file() for path in result["endpoint_metadata_files"].values())
        counts = {
            endpoint: metadata["train_label_count"]
            for endpoint, metadata in result["target_scaler"]["per_endpoint"].items()
        }
        contributions = [result["task_weights"][endpoint] * count for endpoint, count in counts.items()]
        assert contributions == pytest.approx([contributions[0]] * 5)
    else:
        assert (tmp_path / task / "validation_predictions.csv").is_file()
        assert result["calibrator"]["fit_split"] == "validation"
