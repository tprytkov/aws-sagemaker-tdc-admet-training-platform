import json
import math
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from admet_platform.models.baseline import train_local_baseline
from admet_platform.models.metrics import classification_metrics, regression_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
ARTIFACT_NAMES = {
    "model.joblib",
    "metrics.json",
    "predictions_validation.csv",
    "predictions_test.csv",
    "training_metadata.json",
    "feature_metadata.json",
}


def test_descriptor_classification_training(tmp_path: Path) -> None:
    split_paths = _write_classification_splits(tmp_path)
    output_dir = tmp_path / "descriptor_classification"

    result = train_local_baseline(
        **split_paths,
        config_path=CONFIG_DIR / "bbb_martins.yaml",
        feature_type="descriptors",
        output_dir=output_dir,
        random_seed=123,
    )

    metadata = result["training_metadata"]
    assert metadata["task_type"] == "binary_classification"
    assert metadata["feature_type"] == "descriptors"
    assert metadata["model_type"] == "descriptor_logistic_regression"
    assert metadata["feature_count"] == 10
    _assert_artifacts(output_dir)


def test_morgan_classification_training(tmp_path: Path) -> None:
    split_paths = _write_classification_splits(tmp_path)
    output_dir = tmp_path / "morgan_classification"

    result = train_local_baseline(
        **split_paths,
        config_path=CONFIG_DIR / "bbb_martins.yaml",
        feature_type="morgan",
        output_dir=output_dir,
        morgan_bits=64,
        random_seed=123,
    )

    metadata = result["training_metadata"]
    assert metadata["task_type"] == "binary_classification"
    assert metadata["feature_type"] == "morgan"
    assert metadata["model_type"] == "morgan_logistic_regression"
    assert metadata["feature_count"] == 64
    _assert_artifacts(output_dir)


def test_descriptor_regression_training(tmp_path: Path) -> None:
    split_paths = _write_regression_splits(tmp_path)
    output_dir = tmp_path / "descriptor_regression"

    result = train_local_baseline(
        **split_paths,
        config_path=CONFIG_DIR / "caco2_wang.yaml",
        feature_type="descriptors",
        output_dir=output_dir,
        random_seed=123,
    )

    metadata = result["training_metadata"]
    assert metadata["task_type"] == "regression"
    assert metadata["feature_type"] == "descriptors"
    assert metadata["model_type"] == "descriptor_ridge_regression"
    _assert_artifacts(output_dir)


def test_morgan_regression_training(tmp_path: Path) -> None:
    split_paths = _write_regression_splits(tmp_path)
    output_dir = tmp_path / "morgan_regression"

    result = train_local_baseline(
        **split_paths,
        config_path=CONFIG_DIR / "caco2_wang.yaml",
        feature_type="morgan",
        output_dir=output_dir,
        morgan_bits=64,
        random_seed=123,
    )

    metadata = result["training_metadata"]
    assert metadata["task_type"] == "regression"
    assert metadata["feature_type"] == "morgan"
    assert metadata["model_type"] == "morgan_ridge_regression"
    assert metadata["feature_count"] == 64
    _assert_artifacts(output_dir)


def test_metadata_and_target_columns_are_excluded_from_features(tmp_path: Path) -> None:
    split_paths = _write_classification_splits(tmp_path)
    output_dir = tmp_path / "feature_exclusion"

    train_local_baseline(
        **split_paths,
        config_path=CONFIG_DIR / "bbb_martins.yaml",
        feature_type="descriptors",
        output_dir=output_dir,
    )

    model_payload = joblib.load(output_dir / "model.joblib")
    feature_columns = model_payload["feature_columns"]
    assert "molecule_id" not in feature_columns
    assert "target" not in feature_columns
    assert "endpoint_id" not in feature_columns
    assert "split" not in feature_columns
    assert "canonical_smiles" not in feature_columns


def test_deterministic_predictions_with_same_seed(tmp_path: Path) -> None:
    split_paths = _write_classification_splits(tmp_path)
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"

    train_local_baseline(
        **split_paths,
        config_path=CONFIG_DIR / "bbb_martins.yaml",
        feature_type="morgan",
        output_dir=first_output,
        morgan_bits=64,
        random_seed=77,
    )
    train_local_baseline(
        **split_paths,
        config_path=CONFIG_DIR / "bbb_martins.yaml",
        feature_type="morgan",
        output_dir=second_output,
        morgan_bits=64,
        random_seed=77,
    )

    first_predictions = pd.read_csv(first_output / "predictions_test.csv")
    second_predictions = pd.read_csv(second_output / "predictions_test.csv")
    pd.testing.assert_frame_equal(first_predictions, second_predictions)


def test_classification_metric_calculation() -> None:
    metrics, warnings = classification_metrics(
        y_true=np.array([0, 1, 0, 1]),
        y_pred=np.array([0, 1, 1, 1]),
        y_probability=np.array([0.1, 0.9, 0.6, 0.8]),
    )

    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["accuracy"] == 0.75
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert warnings == []


def test_regression_metric_calculation() -> None:
    metrics, warnings = regression_metrics(
        y_true=np.array([1.0, 2.0, 3.0]),
        y_pred=np.array([1.0, 2.5, 2.5]),
    )

    assert metrics["rmse"] is not None
    assert metrics["mae"] is not None
    assert metrics["r2"] is not None
    assert metrics["pearson_correlation"] is not None
    assert metrics["spearman_correlation"] is not None
    assert warnings == []


def test_one_class_classification_split_records_unavailable_metrics(tmp_path: Path) -> None:
    split_paths = _write_classification_splits(tmp_path, one_class_validation=True)
    output_dir = tmp_path / "one_class"

    train_local_baseline(
        **split_paths,
        config_path=CONFIG_DIR / "bbb_martins.yaml",
        feature_type="descriptors",
        output_dir=output_dir,
    )

    metrics = _read_json(output_dir / "metrics.json")
    assert metrics["validation"]["roc_auc"] is None
    assert metrics["validation"]["pr_auc"] is None
    assert metrics["warnings"]


def test_constant_value_regression_metric_handling() -> None:
    metrics, warnings = regression_metrics(
        y_true=np.array([1.0, 1.0, 1.0]),
        y_pred=np.array([1.0, 1.0, 1.0]),
    )

    assert metrics["pearson_correlation"] is None
    assert metrics["spearman_correlation"] is None
    assert warnings


def test_prediction_schema_for_classification_and_regression(tmp_path: Path) -> None:
    classification_paths = _write_classification_splits(tmp_path / "classification")
    regression_paths = _write_regression_splits(tmp_path / "regression")
    classification_output = tmp_path / "classification_output"
    regression_output = tmp_path / "regression_output"

    train_local_baseline(
        **classification_paths,
        config_path=CONFIG_DIR / "bbb_martins.yaml",
        feature_type="descriptors",
        output_dir=classification_output,
    )
    train_local_baseline(
        **regression_paths,
        config_path=CONFIG_DIR / "caco2_wang.yaml",
        feature_type="descriptors",
        output_dir=regression_output,
    )

    classification_predictions = pd.read_csv(classification_output / "predictions_test.csv")
    regression_predictions = pd.read_csv(regression_output / "predictions_test.csv")
    assert list(classification_predictions.columns) == [
        "molecule_id",
        "canonical_smiles",
        "observed_target",
        "predicted_class",
        "predicted_probability",
    ]
    assert list(regression_predictions.columns) == [
        "molecule_id",
        "canonical_smiles",
        "observed_target",
        "predicted_value",
        "residual",
    ]


def test_model_serialization_and_reload(tmp_path: Path) -> None:
    split_paths = _write_classification_splits(tmp_path)
    output_dir = tmp_path / "serialized"

    train_local_baseline(
        **split_paths,
        config_path=CONFIG_DIR / "bbb_martins.yaml",
        feature_type="morgan",
        output_dir=output_dir,
        morgan_bits=64,
    )

    payload = joblib.load(output_dir / "model.joblib")
    assert payload["endpoint_id"] == "bbb_martins"
    assert payload["task_type"] == "binary_classification"
    assert payload["feature_columns"][0] == "morgan_0000"


def test_train_baseline_cli_smoke_execution(tmp_path: Path) -> None:
    split_paths = _write_classification_splits(tmp_path)
    output_dir = tmp_path / "cli_output"

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train_baseline.py"),
            "--train-csv",
            str(split_paths["train_csv"]),
            "--validation-csv",
            str(split_paths["validation_csv"]),
            "--test-csv",
            str(split_paths["test_csv"]),
            "--config",
            str(CONFIG_DIR / "bbb_martins.yaml"),
            "--feature-type",
            "morgan",
            "--morgan-bits",
            "64",
            "--output-dir",
            str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    _assert_artifacts(output_dir)
    assert "Wrote baseline artifacts" in result.stdout


def test_json_outputs_have_no_nan_or_numpy_scalars(tmp_path: Path) -> None:
    split_paths = _write_regression_splits(tmp_path)
    output_dir = tmp_path / "json_safe"

    train_local_baseline(
        **split_paths,
        config_path=CONFIG_DIR / "caco2_wang.yaml",
        feature_type="descriptors",
        output_dir=output_dir,
    )

    for json_name in ("metrics.json", "training_metadata.json", "feature_metadata.json"):
        payload = _read_json(output_dir / json_name)
        _assert_json_safe(payload)


def _write_classification_splits(
    tmp_path: Path,
    one_class_validation: bool = False,
) -> dict[str, Path]:
    rows = {
        "train": [
            ("cls_train_001", "CCO", 0),
            ("cls_train_002", "CCN", 1),
            ("cls_train_003", "c1ccccc1", 0),
            ("cls_train_004", "CC(=O)O", 1),
            ("cls_train_005", "C1CCCCC1", 0),
            ("cls_train_006", "CCOC", 1),
        ],
        "validation": [
            ("cls_valid_001", "CCO", 0),
            ("cls_valid_002", "CCN", 0 if one_class_validation else 1),
            ("cls_valid_003", "CC(=O)O", 0),
            ("cls_valid_004", "CCOC", 0 if one_class_validation else 1),
        ],
        "test": [
            ("cls_test_001", "CCO", 0),
            ("cls_test_002", "CCN", 1),
            ("cls_test_003", "CC(=O)O", 0),
            ("cls_test_004", "CCOC", 1),
        ],
    }
    return _write_splits(tmp_path, rows, "bbb_martins")


def _write_regression_splits(tmp_path: Path) -> dict[str, Path]:
    rows = {
        "train": [
            ("reg_train_001", "CCO", -4.8),
            ("reg_train_002", "CCN", -4.5),
            ("reg_train_003", "c1ccccc1", -3.9),
            ("reg_train_004", "CC(=O)O", -5.2),
            ("reg_train_005", "C1CCCCC1", -4.1),
            ("reg_train_006", "CCOC", -4.7),
        ],
        "validation": [
            ("reg_valid_001", "CCO", -4.9),
            ("reg_valid_002", "CCN", -4.4),
            ("reg_valid_003", "CCOC", -4.6),
        ],
        "test": [
            ("reg_test_001", "CCO", -5.0),
            ("reg_test_002", "CCN", -4.3),
            ("reg_test_003", "CCOC", -4.5),
        ],
    }
    return _write_splits(tmp_path, rows, "caco2_wang")


def _write_splits(tmp_path: Path, rows: dict[str, list[tuple[str, str, float]]], endpoint_id: str) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split_name, split_rows in rows.items():
        file_name = "valid.csv" if split_name == "validation" else f"{split_name}.csv"
        path = tmp_path / file_name
        pd.DataFrame(
            [
                {
                    "molecule_id": molecule_id,
                    "smiles": smiles,
                    "canonical_smiles": smiles,
                    "target": target,
                    "endpoint_id": endpoint_id,
                    "split": split_name,
                }
                for molecule_id, smiles, target in split_rows
            ]
        ).to_csv(path, index=False)
        key = "validation_csv" if split_name == "validation" else f"{split_name}_csv"
        paths[key] = path
    return paths


def _assert_artifacts(output_dir: Path) -> None:
    assert {path.name for path in output_dir.iterdir()} == ARTIFACT_NAMES


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_json_safe(value: object) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _assert_json_safe(item)
    elif isinstance(value, list):
        for item in value:
            _assert_json_safe(item)
    elif isinstance(value, float):
        assert not math.isnan(value)
