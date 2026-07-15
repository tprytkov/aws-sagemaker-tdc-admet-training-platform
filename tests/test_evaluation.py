import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from admet_platform.evaluation import ComparisonOptions, compare_runs, discover_run_dirs, evaluate_model_runs, load_model_run
from admet_platform.evaluation.model_card import REQUIRED_SECTIONS


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_classical_classification_artifact_loading(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "run-a", task_type="binary_classification", feature_type="descriptors")

    run = load_model_run(run_dir)

    assert run.model_family == "classical"
    assert run.feature_type == "descriptors"
    assert run.validation_metrics["roc_auc"] == 0.8


def test_classical_regression_artifact_loading(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "run-r", task_type="regression", feature_type="morgan")

    run = load_model_run(run_dir)

    assert run.task_type == "regression"
    assert run.validation_metrics["rmse"] == 1.2
    assert run.feature_type == "morgan"


def test_chemberta_artifact_loading(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "run-c", chemberta=True)

    run = load_model_run(run_dir)

    assert run.model_family == "chemberta"
    assert run.pretrained_checkpoint == "seyonec/ChemBERTa-zinc-base-v1"
    assert run.tokenizer_path is not None


def test_missing_required_artifact_handling(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "run-a")
    (run_dir / "metrics.json").unlink()

    with pytest.raises(ValueError, match="missing required artifact"):
        load_model_run(run_dir)


def test_malformed_json_handling(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "run-a")
    (run_dir / "metrics.json").write_text("{bad", encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed JSON"):
        load_model_run(run_dir)


def test_endpoint_and_task_mismatch_rejection(tmp_path: Path) -> None:
    run_a = load_model_run(_write_run(tmp_path, "run-a", endpoint_id="bbb_martins"))
    run_b = load_model_run(_write_run(tmp_path, "run-b", endpoint_id="herg_karim"))
    with pytest.raises(ValueError, match="Endpoint mismatch"):
        compare_runs([run_a, run_b])

    run_c = load_model_run(_write_run(tmp_path, "run-c", task_type="regression", endpoint_id="bbb_martins"))
    with pytest.raises(ValueError, match="Task-type mismatch"):
        compare_runs([run_a, run_c])


def test_split_provenance_mismatch_rejection(tmp_path: Path) -> None:
    run_a = load_model_run(_write_run(tmp_path, "run-a", train_rows=10))
    run_b = load_model_run(_write_run(tmp_path, "run-b", train_rows=11))

    with pytest.raises(ValueError, match="Split-provenance mismatch"):
        compare_runs([run_a, run_b])


def test_classification_comparison_and_validation_only_selection(tmp_path: Path) -> None:
    run_a = load_model_run(_write_run(tmp_path, "run-a", validation_metrics={"roc_auc": 0.7, "pr_auc": 0.9}, test_metrics={"roc_auc": 0.99}))
    run_b = load_model_run(_write_run(tmp_path, "run-b", validation_metrics={"roc_auc": 0.8, "pr_auc": 0.7}, test_metrics={"roc_auc": 0.1}))

    result = compare_runs([run_a, run_b])

    assert result.recommended_run_id == "run-b"
    assert result.comparison_metric == "roc_auc"
    assert result.rows[0]["primary_test_metric"] == 0.99


def test_regression_comparison(tmp_path: Path) -> None:
    run_a = load_model_run(_write_run(tmp_path, "run-a", task_type="regression", validation_metrics={"rmse": 1.2, "mae": 1.0}, test_metrics={"rmse": 0.1}))
    run_b = load_model_run(_write_run(tmp_path, "run-b", task_type="regression", validation_metrics={"rmse": 0.9, "mae": 0.8}, test_metrics={"rmse": 9.0}))

    result = compare_runs([run_a, run_b])

    assert result.recommended_run_id == "run-b"
    assert result.higher_is_better is False


def test_development_run_exclusion_and_override(tmp_path: Path) -> None:
    dev = load_model_run(_write_run(tmp_path, "run-dev", development=True, validation_metrics={"roc_auc": 0.99}))
    regular = load_model_run(_write_run(tmp_path, "run-regular", validation_metrics={"roc_auc": 0.6}))

    excluded = compare_runs([dev, regular])
    included = compare_runs([dev, regular], ComparisonOptions(include_development_runs=True))

    assert excluded.recommended_run_id == "run-regular"
    assert included.recommended_run_id == "run-dev"


def test_missing_roc_auc_fallback_and_missing_regression_metric(tmp_path: Path) -> None:
    run_a = load_model_run(_write_run(tmp_path, "run-a", validation_metrics={"roc_auc": None, "pr_auc": 0.4}))
    run_b = load_model_run(_write_run(tmp_path, "run-b", validation_metrics={"roc_auc": None, "pr_auc": 0.7}))
    assert compare_runs([run_a, run_b]).comparison_metric == "pr_auc"

    reg = load_model_run(_write_run(tmp_path, "run-r", task_type="regression", validation_metrics={"rmse": None, "mae": 1.0}))
    result = compare_runs([reg])
    assert result.recommendation_status == "no_eligible_model"


def test_near_tie_and_no_eligible_model(tmp_path: Path) -> None:
    run_a = load_model_run(_write_run(tmp_path, "run-a", validation_metrics={"roc_auc": 0.801}))
    run_b = load_model_run(_write_run(tmp_path, "run-b", validation_metrics={"roc_auc": 0.8}))

    tied = compare_runs([run_a, run_b], ComparisonOptions(near_tie_tolerance=0.01))
    assert tied.recommendation_status == "near_tie"
    assert tied.near_tie_run_ids == ["run-b"]

    dev = load_model_run(_write_run(tmp_path, "run-dev", development=True))
    none = compare_runs([dev])
    assert none.recommendation_status == "no_eligible_model"


def test_artifact_outputs_schemas_model_card_and_registry(tmp_path: Path) -> None:
    run_a = _write_run(tmp_path, "run-a", feature_type="descriptors")
    run_b = _write_run(tmp_path, "run-b", feature_type="morgan", validation_metrics={"roc_auc": 0.9})
    output_dir = tmp_path / "evaluation"

    result = evaluate_model_runs([run_a, run_b], output_dir)

    assert result.recommended_run_id == "run-b"
    expected = {
        "evaluation_summary.json",
        "model_comparison.csv",
        "model_comparison.json",
        "recommended_model.json",
        "evaluation_warnings.json",
        "model_card.md",
        "registry_entry.json",
    }
    assert expected == {path.name for path in output_dir.iterdir()}
    summary = _read_json(output_dir / "evaluation_summary.json")
    recommended = _read_json(output_dir / "recommended_model.json")
    registry = _read_json(output_dir / "registry_entry.json")
    card = (output_dir / "model_card.md").read_text(encoding="utf-8")

    assert summary["endpoint_id"] == "bbb_martins"
    assert recommended["test_metrics_descriptive_only"]
    assert registry["approval_status"] == "pending_review"
    assert registry["aws_model_registry_registered"] is False
    for section in REQUIRED_SECTIONS:
        assert f"## {section}" in card
    assert "unavailable" in card
    assert pd.read_csv(output_dir / "model_comparison.csv").columns.tolist()[0] == "run_id"


def test_json_safe_duplicate_run_id_and_discovery(tmp_path: Path) -> None:
    run_a = _write_run(tmp_path, "dup")
    run_b = _write_run(tmp_path, "dup-copy", run_id="dup")

    with pytest.raises(ValueError, match="Duplicate"):
        evaluate_model_runs([run_a, run_b], tmp_path / "out")

    discovered = discover_run_dirs(tmp_path)
    assert run_a in discovered
    assert run_b in discovered


def test_cli_smoke_execution_and_deterministic_comparison(tmp_path: Path) -> None:
    run_a = _write_run(tmp_path, "run-a")
    run_b = _write_run(tmp_path, "run-b", validation_metrics={"roc_auc": 0.9})
    out = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate_models.py"),
            "--run-dir",
            str(run_a),
            "--run-dir",
            str(run_b),
            "--output-dir",
            str(out),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Recommended run: run-b" in result.stdout
    first = (out / "model_comparison.json").read_text(encoding="utf-8")
    evaluate_model_runs([run_a, run_b], out)
    second = (out / "model_comparison.json").read_text(encoding="utf-8")
    assert first == second


def _write_run(
    tmp_path: Path,
    name: str,
    *,
    run_id: str | None = None,
    endpoint_id: str = "bbb_martins",
    task_type: str = "binary_classification",
    feature_type: str | None = "descriptors",
    validation_metrics: dict | None = None,
    test_metrics: dict | None = None,
    development: bool = False,
    chemberta: bool = False,
    train_rows: int = 10,
) -> Path:
    run_dir = tmp_path / name
    run_dir.mkdir()
    if task_type == "binary_classification":
        validation_metrics = validation_metrics or {"roc_auc": 0.8, "pr_auc": 0.7, "balanced_accuracy": 0.6, "f1": 0.5, "matthews_correlation_coefficient": 0.4}
        test_metrics = test_metrics or {"roc_auc": 0.75, "pr_auc": 0.65, "balanced_accuracy": 0.55, "f1": 0.45, "matthews_correlation_coefficient": 0.35}
        _write_classification_predictions(run_dir / "predictions_validation.csv")
        _write_classification_predictions(run_dir / "predictions_test.csv")
    else:
        validation_metrics = validation_metrics or {"rmse": 1.2, "mae": 1.0, "r2": 0.1, "spearman_correlation": 0.2}
        test_metrics = test_metrics or {"rmse": 1.4, "mae": 1.1, "r2": 0.0, "spearman_correlation": 0.1}
        _write_regression_predictions(run_dir / "predictions_validation.csv")
        _write_regression_predictions(run_dir / "predictions_test.csv")
    metrics = {
        "endpoint_id": endpoint_id,
        "task_type": task_type,
        "feature_type": None if chemberta else feature_type,
        "model_type": "chemberta" if chemberta else f"{feature_type}_model",
        "validation": validation_metrics,
        "test": test_metrics,
        "warnings": ["development row limit was used"] if development else [],
    }
    metadata = {
        "run_id": run_id or name,
        "endpoint_id": endpoint_id,
        "task_type": task_type,
        "source_dataset": "BBB_Martins",
        "feature_type": None if chemberta else feature_type,
        "model_type": "chemberta" if chemberta else f"{feature_type}_model",
        "training_row_count": train_rows,
        "validation_row_count": 3,
        "test_row_count": 3,
        "feature_count": None if chemberta else 10,
        "development_row_limit": 8 if development else None,
        "package_versions": {"python_packages": {"pandas": "test"}},
        "warnings": ["development row limit was used"] if development else [],
    }
    if chemberta:
        metadata["pretrained_model_name"] = "seyonec/ChemBERTa-zinc-base-v1"
        (run_dir / "model").mkdir()
        (run_dir / "tokenizer").mkdir()
        (run_dir / "model_config.json").write_text(
            json.dumps({"model_name": "seyonec/ChemBERTa-zinc-base-v1", "development_row_limit": 8 if development else None}),
            encoding="utf-8",
        )
    else:
        (run_dir / "model.joblib").write_text("fake", encoding="utf-8")
        (run_dir / "feature_metadata.json").write_text(
            json.dumps({"feature_type": feature_type, "n_features": 10}),
            encoding="utf-8",
        )
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run_dir / "training_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return run_dir


def _write_classification_predictions(path: Path) -> None:
    pd.DataFrame(
        {
            "molecule_id": ["m1", "m2", "m3"],
            "observed_target": [0, 1, 1],
            "predicted_class": [0, 1, 0],
            "predicted_probability": [0.2, 0.8, 0.4],
        }
    ).to_csv(path, index=False)


def _write_regression_predictions(path: Path) -> None:
    pd.DataFrame(
        {
            "molecule_id": ["m1", "m2", "m3"],
            "observed_target": [1.0, 2.0, 3.0],
            "predicted_value": [1.1, 1.9, 3.2],
        }
    ).to_csv(path, index=False)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
