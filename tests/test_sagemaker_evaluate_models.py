import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

from admet_platform.sagemaker import evaluate_models as sm_eval


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_processing_input_and_output_path_resolution(tmp_path: Path) -> None:
    args = Namespace(
        runs_dir=str(tmp_path / "runs"),
        config_dir=str(tmp_path / "config"),
        config=None,
        output_dir=str(tmp_path / "out"),
    )
    paths = sm_eval.resolve_processing_paths(args, {})

    assert paths["runs_dir"] == tmp_path / "runs"
    assert paths["output_dir"] == tmp_path / "out"
    assert paths["config_path"] is None


def test_candidate_discovery_classical_chemberta_and_mixed_evaluation(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _write_run(runs, "descriptor", feature_type="descriptors", validation_roc=0.7)
    _write_run(runs, "morgan", feature_type="morgan", validation_roc=0.8)
    _write_run(runs, "chemberta", chemberta=True, validation_roc=0.75)

    manifest = sm_eval.run_processing_evaluation(runs_dir=runs, output_dir=tmp_path / "out", run_id="eval-1")

    assert manifest["status"] == "completed"
    assert len(manifest["discovered_candidate_run_ids"]) == 3
    assert manifest["recommended_run_id"] == "morgan"
    _assert_output_contract(tmp_path / "out")


def test_missing_input_and_malformed_run_rejection(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        sm_eval.run_processing_evaluation(runs_dir=tmp_path / "missing", output_dir=tmp_path / "out")

    runs = tmp_path / "runs"
    bad = runs / "bad"
    bad.mkdir(parents=True)
    (bad / "metrics.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed model-run"):
        sm_eval.run_processing_evaluation(runs_dir=runs, output_dir=tmp_path / "out")


def test_endpoint_task_and_split_mismatch_rejections(tmp_path: Path) -> None:
    runs = tmp_path / "runs_endpoint"
    _write_run(runs, "a", endpoint_id="bbb_martins")
    _write_run(runs, "b", endpoint_id="herg_karim")
    with pytest.raises(ValueError, match="Endpoint mismatch"):
        sm_eval.run_processing_evaluation(runs_dir=runs, output_dir=tmp_path / "out")

    runs = tmp_path / "runs_task"
    _write_run(runs, "a")
    _write_run(runs, "b", task_type="regression")
    with pytest.raises(ValueError, match="Task-type mismatch"):
        sm_eval.run_processing_evaluation(runs_dir=runs, output_dir=tmp_path / "out2")

    runs = tmp_path / "runs_split"
    _write_run(runs, "a", train_rows=10)
    _write_run(runs, "b", train_rows=11)
    with pytest.raises(ValueError, match="Split-provenance mismatch"):
        sm_eval.run_processing_evaluation(runs_dir=runs, output_dir=tmp_path / "out3")


def test_development_exclusion_override_validation_only_and_near_tie(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _write_run(runs, "regular", validation_roc=0.6, test_roc=0.99)
    _write_run(runs, "dev", validation_roc=0.95, test_roc=0.1, development=True)

    excluded = sm_eval.run_processing_evaluation(runs_dir=runs, output_dir=tmp_path / "out_excluded")
    included = sm_eval.run_processing_evaluation(
        runs_dir=runs,
        output_dir=tmp_path / "out_included",
        include_development_runs=True,
    )

    assert excluded["recommended_run_id"] == "regular"
    assert included["recommended_run_id"] == "dev"
    assert included["include_development_runs"] is True

    runs_tie = tmp_path / "runs_tie"
    _write_run(runs_tie, "a", validation_roc=0.801)
    _write_run(runs_tie, "b", validation_roc=0.8)
    tie = sm_eval.run_processing_evaluation(runs_dir=runs_tie, output_dir=tmp_path / "out_tie")
    assert tie["recommendation_status"] == "near_tie"


def test_processing_manifest_inventory_failure_and_redaction(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _write_run(runs, "a")
    manifest = sm_eval.run_processing_evaluation(runs_dir=runs, output_dir=tmp_path / "out", run_id="eval-1")
    inventory = json.loads((tmp_path / "out" / "metadata" / "artifact_inventory.json").read_text(encoding="utf-8"))

    assert {
        "processing_run_id",
        "endpoint_id",
        "eligible_run_ids",
        "recommendation_status",
        "generated_artifact_inventory",
        "package_versions",
        "status",
    } <= set(manifest)
    assert inventory["evaluation"][0]["exists"] is True
    assert inventory["metadata"][0]["exists"] is True

    failed = sm_eval.write_failed_manifest(tmp_path / "failed", RuntimeError("bad aws_secret_access_key=abc"))
    failed_manifest = json.loads(failed.read_text(encoding="utf-8"))
    assert failed_manifest["status"] == "failed"
    assert "[REDACTED]" in failed_manifest["error"]["message"]


def test_evaluation_config_loading_precedence_minimal_deps_and_cli_smoke(tmp_path: Path) -> None:
    config_path = tmp_path / "evaluation.yaml"
    config_path.write_text("endpoint_id: bbb_martins\nnear_tie_tolerance: 0.2\ninclude_development_runs: true\n", encoding="utf-8")
    loaded = sm_eval.load_evaluation_config(config_path)
    effective = sm_eval.build_effective_config(loaded, {"near_tie_tolerance": 0.01})

    assert effective["near_tie_tolerance"] == 0.01
    reqs = (PROJECT_ROOT / "sagemaker" / "evaluation_requirements.txt").read_text(encoding="utf-8").lower()
    assert "pandas" in reqs and "numpy" in reqs and "pyyaml" in reqs
    assert "torch" not in reqs and "transformers" not in reqs and "rdkit" not in reqs and "tdc" not in reqs

    runs = tmp_path / "runs"
    _write_run(runs, "a")
    output = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "sagemaker" / "evaluate_models.py"),
            "--runs-dir",
            str(runs),
            "--output-dir",
            str(output),
            "--endpoint-id",
            "bbb_martins",
        ],
        cwd=PROJECT_ROOT / "sagemaker",
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    _assert_output_contract(output)


def _write_run(
    root: Path,
    name: str,
    *,
    endpoint_id: str = "bbb_martins",
    task_type: str = "binary_classification",
    feature_type: str = "descriptors",
    validation_roc: float = 0.8,
    test_roc: float = 0.75,
    train_rows: int = 10,
    development: bool = False,
    chemberta: bool = False,
) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    if task_type == "binary_classification":
        metrics = {
            "roc_auc": validation_roc,
            "pr_auc": 0.7,
            "balanced_accuracy": 0.6,
            "f1": 0.5,
            "matthews_correlation_coefficient": 0.4,
        }
        test = {**metrics, "roc_auc": test_roc}
        _classification_predictions(run_dir / "predictions_validation.csv")
        _classification_predictions(run_dir / "predictions_test.csv")
    else:
        metrics = {"rmse": 1.0, "mae": 0.8, "r2": 0.2, "spearman_correlation": 0.3}
        test = {"rmse": 1.2, "mae": 0.9, "r2": 0.1, "spearman_correlation": 0.2}
        _regression_predictions(run_dir / "predictions_validation.csv")
        _regression_predictions(run_dir / "predictions_test.csv")
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "endpoint_id": endpoint_id,
                "task_type": task_type,
                "feature_type": None if chemberta else feature_type,
                "model_type": "chemberta" if chemberta else f"{feature_type}_model",
                "validation": metrics,
                "test": test,
                "warnings": ["development row limit was used"] if development else [],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "training_metadata.json").write_text(
        json.dumps(
            {
                "run_id": name,
                "endpoint_id": endpoint_id,
                "task_type": task_type,
                "source_dataset": "BBB_Martins",
                "feature_type": None if chemberta else feature_type,
                "model_type": "chemberta" if chemberta else f"{feature_type}_model",
                "pretrained_model_name": "seyonec/ChemBERTa-zinc-base-v1" if chemberta else None,
                "training_row_count": train_rows,
                "validation_row_count": 3,
                "test_row_count": 3,
                "feature_count": None if chemberta else 10,
                "development_row_limit": 8 if development else None,
                "package_versions": {"pandas": "test"},
                "warnings": ["development row limit was used"] if development else [],
            }
        ),
        encoding="utf-8",
    )
    if chemberta:
        (run_dir / "model").mkdir()
        (run_dir / "tokenizer").mkdir()
        (run_dir / "model_config.json").write_text(json.dumps({"model_name": "seyonec/ChemBERTa-zinc-base-v1"}), encoding="utf-8")
    else:
        (run_dir / "model.joblib").write_text("fake", encoding="utf-8")
        (run_dir / "feature_metadata.json").write_text(json.dumps({"feature_type": feature_type, "n_features": 10}), encoding="utf-8")
    return run_dir


def _classification_predictions(path: Path) -> None:
    pd.DataFrame(
        {"observed_target": [0, 1, 1], "predicted_class": [0, 1, 0], "predicted_probability": [0.2, 0.8, 0.4]}
    ).to_csv(path, index=False)


def _regression_predictions(path: Path) -> None:
    pd.DataFrame({"observed_target": [1.0, 2.0, 3.0], "predicted_value": [1.1, 2.1, 2.8]}).to_csv(path, index=False)


def _assert_output_contract(output: Path) -> None:
    assert (output / "evaluation" / "evaluation_summary.json").exists()
    assert (output / "evaluation" / "model_comparison.csv").exists()
    assert (output / "evaluation" / "model_comparison.json").exists()
    assert (output / "evaluation" / "recommended_model.json").exists()
    assert (output / "evaluation" / "evaluation_warnings.json").exists()
    assert (output / "model_card" / "model_card.md").exists()
    assert (output / "registry" / "registry_entry.json").exists()
    assert (output / "metadata" / "evaluation_processing_manifest.json").exists()
    assert (output / "metadata" / "artifact_inventory.json").exists()
