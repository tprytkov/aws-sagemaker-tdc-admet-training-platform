import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from admet_platform.benchmarks.local import run_local_benchmarks, summarize_prepared_dataset
from admet_platform.config import load_endpoint_config
from admet_platform.data.prepare import prepare_dataset_artifacts
from admet_platform.models.artifacts import to_json_safe


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def test_benchmark_matrix_generation(tmp_path: Path) -> None:
    prepared_root = _prepared_root(tmp_path)
    result = run_local_benchmarks(
        config_paths=[CONFIG_DIR / "bbb_martins.yaml", CONFIG_DIR / "caco2_wang.yaml"],
        prepared_root=prepared_root,
        benchmark_root=tmp_path / "benchmarks",
        prepare_dataset_func=_unexpected_prepare,
        train_func=_fake_train,
    )

    pairs = {(row["endpoint_id"], row["feature_type"]) for row in result["benchmark_rows"]}
    assert pairs == {
        ("bbb_martins", "descriptors"),
        ("bbb_martins", "morgan"),
        ("caco2_wang", "descriptors"),
        ("caco2_wang", "morgan"),
    }


def test_aggregate_summary_schema_and_task_rows(tmp_path: Path) -> None:
    prepared_root = _prepared_root(tmp_path)
    benchmark_root = tmp_path / "benchmarks"

    run_local_benchmarks(
        config_paths=[CONFIG_DIR / "bbb_martins.yaml", CONFIG_DIR / "caco2_wang.yaml"],
        prepared_root=prepared_root,
        benchmark_root=benchmark_root,
        feature_types=["descriptors"],
        prepare_dataset_func=_unexpected_prepare,
        train_func=_fake_train,
    )

    summary = pd.read_csv(benchmark_root / "benchmark_summary.csv")
    expected_columns = {
        "endpoint_id",
        "task_type",
        "feature_type",
        "model_type",
        "train_rows",
        "validation_rows",
        "test_rows",
        "feature_count",
        "primary_validation_metric",
        "primary_test_metric",
        "output_directory",
        "warnings",
        "run_status",
    }
    assert expected_columns <= set(summary.columns)
    assert set(summary["task_type"]) == {"binary_classification", "regression"}


def test_failed_run_capture_without_losing_successful_runs(tmp_path: Path) -> None:
    prepared_root = _prepared_root(tmp_path)

    def train_with_failure(**kwargs):
        if kwargs["feature_type"] == "morgan":
            raise RuntimeError("forced failure")
        return _fake_train(**kwargs)

    result = run_local_benchmarks(
        config_paths=[CONFIG_DIR / "bbb_martins.yaml"],
        prepared_root=prepared_root,
        benchmark_root=tmp_path / "benchmarks",
        prepare_dataset_func=_unexpected_prepare,
        train_func=train_with_failure,
    )

    assert len(result["failures"]) == 1
    assert {row["run_status"] for row in result["benchmark_rows"]} == {"success", "failed"}


def test_cli_returns_nonzero_when_run_fails(tmp_path: Path) -> None:
    missing_prepared_root = tmp_path / "missing_prepared"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_local_benchmarks.py"),
            "--config",
            str(CONFIG_DIR / "bbb_martins.yaml"),
            "--prepared-root",
            str(missing_prepared_root),
            "--benchmark-root",
            str(tmp_path / "benchmarks"),
            "--feature-type",
            "descriptors",
            "--max-rows",
            "1",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0


def test_reuse_existing_prepared_data_and_force_rerun_behavior(tmp_path: Path) -> None:
    prepared_root = _prepared_root(tmp_path)
    calls = {"count": 0}

    def prepare_counter(**kwargs):
        calls["count"] += 1
        return {}

    run_local_benchmarks(
        config_paths=[CONFIG_DIR / "bbb_martins.yaml"],
        prepared_root=prepared_root,
        benchmark_root=tmp_path / "reuse",
        feature_types=["descriptors"],
        prepare_dataset_func=prepare_counter,
        train_func=_fake_train,
    )
    assert calls["count"] == 0

    run_local_benchmarks(
        config_paths=[CONFIG_DIR / "bbb_martins.yaml"],
        prepared_root=prepared_root,
        benchmark_root=tmp_path / "force",
        feature_types=["descriptors"],
        force_rerun=True,
        prepare_dataset_func=prepare_counter,
        train_func=_fake_train,
    )
    assert calls["count"] == 1


def test_dataset_statistics_overlap_and_target_summaries(tmp_path: Path) -> None:
    prepared_root = _prepared_root(tmp_path, include_overlap=True)
    bbb_config = load_endpoint_config(CONFIG_DIR / "bbb_martins.yaml")
    caco2_config = load_endpoint_config(CONFIG_DIR / "caco2_wang.yaml")

    bbb_summary = summarize_prepared_dataset(prepared_root / "bbb_martins", bbb_config)
    caco2_summary = summarize_prepared_dataset(prepared_root / "caco2_wang", caco2_config)

    assert bbb_summary["target_distribution"] == {"0": 3, "1": 3}
    assert bbb_summary["cross_split_overlap_counts"]["train_validation"] > 0
    assert bbb_summary["warnings"]
    assert caco2_summary["target_min"] == -5.2
    assert caco2_summary["target_max"] == -3.9
    assert "target_mean" in caco2_summary


def test_json_safe_metadata(tmp_path: Path) -> None:
    prepared_root = _prepared_root(tmp_path)
    result = run_local_benchmarks(
        config_paths=[CONFIG_DIR / "bbb_martins.yaml"],
        prepared_root=prepared_root,
        benchmark_root=tmp_path / "benchmarks",
        feature_types=["descriptors"],
        prepare_dataset_func=_unexpected_prepare,
        train_func=_fake_train,
    )

    json.dumps(to_json_safe(result))


def test_cli_smoke_execution_with_synthetic_prepared_data(tmp_path: Path) -> None:
    prepared_root = _prepared_root(tmp_path)
    benchmark_root = tmp_path / "benchmarks"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_local_benchmarks.py"),
            "--config",
            str(CONFIG_DIR / "bbb_martins.yaml"),
            "--prepared-root",
            str(prepared_root),
            "--benchmark-root",
            str(benchmark_root),
            "--feature-type",
            "descriptors",
            "--morgan-bits",
            "64",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert (benchmark_root / "benchmark_summary.csv").exists()
    assert (benchmark_root / "dataset_summary.csv").exists()
    assert "Wrote benchmark artifacts" in result.stdout


def _prepared_root(tmp_path: Path, include_overlap: bool = False) -> Path:
    root = tmp_path / "prepared"
    _write_prepared_endpoint(
        root / "bbb_martins",
        CONFIG_DIR / "bbb_martins.yaml",
        [
            ("bbb_train_001", "CCO", 0, "train"),
            ("bbb_train_002", "CCN", 1, "train"),
            ("bbb_train_003", "CCOC", 0, "train"),
            ("bbb_valid_001", "CCO" if include_overlap else "CCCl", 1, "validation"),
            ("bbb_valid_002", "CCBr", 0, "validation"),
            ("bbb_test_001", "CCCC", 1, "test"),
        ],
    )
    _write_prepared_endpoint(
        root / "caco2_wang",
        CONFIG_DIR / "caco2_wang.yaml",
        [
            ("caco2_train_001", "CCO", -4.8, "train"),
            ("caco2_train_002", "CCN", -5.2, "train"),
            ("caco2_valid_001", "CCCl", -4.1, "validation"),
            ("caco2_valid_002", "CCBr", -3.9, "validation"),
            ("caco2_test_001", "CCCC", -4.5, "test"),
        ],
    )
    return root


def _write_prepared_endpoint(
    output_dir: Path,
    config_path: Path,
    rows: list[tuple[str, str, float, str]],
) -> None:
    input_csv = output_dir.parent / f"{output_dir.name}_input.csv"
    input_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"molecule_id": molecule_id, "smiles": smiles, "target": target, "split": split}
            for molecule_id, smiles, target, split in rows
        ]
    ).to_csv(input_csv, index=False)
    prepare_dataset_artifacts(input_csv, config_path, output_dir)


def _fake_train(**kwargs) -> dict[str, object]:
    config = load_endpoint_config(kwargs["config_path"])
    feature_type = kwargs["feature_type"]
    model_type = (
        f"{'descriptor' if feature_type == 'descriptors' else 'morgan'}_"
        f"{'logistic_regression' if config.task_type == 'binary_classification' else 'ridge_regression'}"
    )
    output_dir = Path(kwargs["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_metrics = (
        {"roc_auc": 0.5, "pr_auc": 0.5}
        if config.task_type == "binary_classification"
        else {"rmse": 1.0, "mae": 0.8}
    )
    test_metrics = (
        {"roc_auc": 0.6, "pr_auc": 0.55}
        if config.task_type == "binary_classification"
        else {"rmse": 1.2, "mae": 0.9}
    )
    return {
        "metrics": {
            "endpoint_id": config.endpoint_id,
            "task_type": config.task_type,
            "feature_type": feature_type,
            "model_type": model_type,
            "validation": validation_metrics,
            "test": test_metrics,
            "warnings": [],
        },
        "training_metadata": {
            "model_type": model_type,
            "training_row_count": 3,
            "validation_row_count": 2,
            "test_row_count": 1,
            "feature_count": 10 if feature_type == "descriptors" else 64,
        },
    }


def _unexpected_prepare(**kwargs) -> dict[str, object]:
    raise AssertionError("prepared data should have been reused")
