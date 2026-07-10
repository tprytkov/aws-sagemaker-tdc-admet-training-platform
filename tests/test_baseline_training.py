import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from admet_platform.data.prepare import prepare_dataset
from admet_platform.training.baseline import train_baseline_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"


def test_binary_classification_baseline_writes_model_and_metrics(tmp_path: Path) -> None:
    prepared_csv = _prepare_sample(tmp_path, "bbb_martins_sample.csv", "bbb_martins.yaml")
    model_path = tmp_path / "bbb_baseline.joblib"
    metrics_path = tmp_path / "bbb_baseline_metrics.json"

    metrics = train_baseline_model(
        input_csv=prepared_csv,
        config_path=CONFIG_DIR / "bbb_martins.yaml",
        model_output_path=model_path,
        metrics_json_path=metrics_path,
    )

    written_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert model_path.exists()
    assert metrics_path.exists()
    assert written_metrics == metrics
    assert written_metrics["endpoint_id"] == "bbb_martins"
    assert written_metrics["task_type"] == "binary_classification"
    assert written_metrics["model_type"] == "tfidf_logistic_regression"
    assert written_metrics["n_train"] == 3
    assert written_metrics["n_test"] == 2
    assert {"accuracy", "balanced_accuracy", "f1", "auroc"} <= set(written_metrics["metrics"])


def test_regression_baseline_writes_model_and_metrics(tmp_path: Path) -> None:
    prepared_csv = _prepare_sample(tmp_path, "caco2_wang_sample.csv", "caco2_wang.yaml")
    model_path = tmp_path / "caco2_baseline.joblib"
    metrics_path = tmp_path / "caco2_baseline_metrics.json"

    metrics = train_baseline_model(
        input_csv=prepared_csv,
        config_path=CONFIG_DIR / "caco2_wang.yaml",
        model_output_path=model_path,
        metrics_json_path=metrics_path,
    )

    written_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert model_path.exists()
    assert metrics_path.exists()
    assert written_metrics == metrics
    assert written_metrics["endpoint_id"] == "caco2_wang"
    assert written_metrics["task_type"] == "regression"
    assert written_metrics["model_type"] == "tfidf_ridge_regression"
    assert written_metrics["n_train"] == 3
    assert written_metrics["n_test"] == 2
    assert {"mae", "rmse", "r2"} <= set(written_metrics["metrics"])


def test_metrics_json_has_expected_top_level_fields(tmp_path: Path) -> None:
    prepared_csv = _prepare_sample(tmp_path, "herg_karim_sample.csv", "herg_karim.yaml")
    metrics_path = tmp_path / "herg_baseline_metrics.json"

    train_baseline_model(
        input_csv=prepared_csv,
        config_path=CONFIG_DIR / "herg_karim.yaml",
        model_output_path=tmp_path / "herg_baseline.joblib",
        metrics_json_path=metrics_path,
    )

    written_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert {
        "endpoint_id",
        "task_type",
        "model_type",
        "n_train",
        "n_test",
        "metrics",
    } <= set(written_metrics)


def test_train_baseline_cli_works_on_prepared_sample_dataset(tmp_path: Path) -> None:
    prepared_csv = _prepare_sample(tmp_path, "bbb_martins_sample.csv", "bbb_martins.yaml")
    model_path = tmp_path / "bbb_cli_baseline.joblib"
    metrics_path = tmp_path / "bbb_cli_baseline_metrics.json"

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train_baseline.py"),
            "--input-csv",
            str(prepared_csv),
            "--config",
            str(CONFIG_DIR / "bbb_martins.yaml"),
            "--model-output",
            str(model_path),
            "--metrics-json",
            str(metrics_path),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert model_path.exists()
    assert metrics_path.exists()
    assert "Wrote baseline model" in result.stdout


def _prepare_sample(tmp_path: Path, sample_name: str, config_name: str) -> Path:
    prepared_csv = tmp_path / sample_name.replace("_sample.csv", "_clean.csv")
    prepare_dataset(
        input_csv=SAMPLE_DIR / sample_name,
        config_path=CONFIG_DIR / config_name,
        output_csv=prepared_csv,
        summary_json=tmp_path / sample_name.replace("_sample.csv", "_summary.json"),
    )

    df = pd.read_csv(prepared_csv)
    assert not df.empty
    return prepared_csv
