import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from admet_platform.training.multitask_calibration import (
    RAW_PREDICTION_COLUMNS,
    apply_platt_calibration,
    calibrate_endpoint_predictions,
    calibration_metrics,
    expected_calibration_error,
    fit_platt_calibrator,
    run_multitask_calibration,
    sigmoid,
)
from admet_platform.training.multitask_run import run_multitask_training
from admet_platform.training.multitask_trainer import MultiTaskTrainer


DYNAMIC_TASKS = ("dynamic_clearance", "dynamic_toxicity")


def test_raw_logit_probability_consistency_and_calibrated_range() -> None:
    logits = np.asarray([-100.0, -1.0, 0.0, 1.0, 100.0])
    probabilities = sigmoid(logits)
    predictions = pd.DataFrame(
        {
            "molecule_id": [f"mol-{index}" for index in range(len(logits))],
            "canonical_smiles": ["CCO"] * len(logits),
            "target": [0, 0, 0, 1, 1],
            "raw_logit": logits,
            "probability": probabilities,
            "prediction": (probabilities >= 0.5).astype(int),
        }
    )

    calibrated, parameters, _ = calibrate_endpoint_predictions(
        predictions,
        minimum_class_support=1,
    )

    assert parameters["fit_status"] == "fitted"
    assert np.allclose(sigmoid(logits), probabilities)
    assert calibrated["calibrated_probability"].between(0.0, 1.0).all()


def test_platt_fitting_is_deterministic_and_parameters_apply_exactly() -> None:
    logits = np.linspace(-3.0, 3.0, 40)
    targets = np.asarray([0] * 20 + [1] * 20)

    first = fit_platt_calibrator(logits, targets, seed=42)
    second = fit_platt_calibrator(logits, targets, seed=42)

    assert first == second
    assert first["fit_status"] == "fitted"
    manual = sigmoid(
        first["coefficient_a"] * logits + first["intercept_b"]
    )
    assert np.allclose(apply_platt_calibration(logits, first), manual)


def test_calibration_metrics_on_small_fixed_example() -> None:
    targets = np.asarray([0, 1])
    probabilities = np.asarray([0.25, 0.75])

    metrics = calibration_metrics(targets, probabilities)

    assert metrics["brier_score"] == pytest.approx(0.0625)
    assert metrics["binary_log_loss"] == pytest.approx(-np.log(0.75))
    assert metrics["expected_calibration_error_10_bins"] == pytest.approx(0.25)
    assert expected_calibration_error(targets, probabilities) == pytest.approx(0.25)
    assert metrics["roc_auc"] == 1.0
    assert metrics["average_precision"] == 1.0
    assert metrics["class_counts"] == {"class_0": 1, "class_1": 1, "total": 2}


def test_hia_support_rule_retains_uncalibrated_probabilities() -> None:
    logits = np.linspace(-2.0, 2.0, 37)
    targets = np.asarray([0] * 7 + [1] * 30)
    stored_probabilities = sigmoid(logits)
    stored_probabilities[0] += 1e-8
    predictions = pd.DataFrame(
        {
            "molecule_id": [f"hia-{index}" for index in range(len(logits))],
            "canonical_smiles": ["CCO"] * len(logits),
            "target": targets,
            "raw_logit": logits,
            "probability": stored_probabilities,
            "prediction": (stored_probabilities >= 0.5).astype(int),
        }
    )

    parameters = fit_platt_calibrator(logits, targets)
    retained = apply_platt_calibration(logits, parameters)
    calibrated, _, _ = calibrate_endpoint_predictions(predictions)

    assert parameters["fit_status"] == "insufficient_class_support"
    assert parameters["class_counts"] == {
        "class_0": 7,
        "class_1": 30,
        "total": 37,
    }
    assert parameters["coefficient_a"] is None
    assert parameters["intercept_b"] is None
    assert np.allclose(retained, sigmoid(logits))
    assert np.array_equal(
        calibrated["calibrated_probability"].to_numpy(),
        stored_probabilities,
    )


def test_calibration_rejects_non_validation_source_before_file_access(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="must be 'validation'"):
        run_multitask_calibration(
            config_path=tmp_path / "missing.yaml",
            checkpoint_path=tmp_path / "missing.pt",
            prepared_root=tmp_path / "prepared",
            output_dir=tmp_path / "output",
            source_split="test",
        )


def test_dynamic_validation_only_calibration_writes_contract_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tiny_model_tokenizer_dir: Path,
) -> None:
    config_path, prepared_root = _write_dynamic_fixture(tmp_path)
    run_root = tmp_path / "training-run"
    run_multitask_training(
        config_path=config_path,
        prepared_root=prepared_root,
        output_dir=run_root,
        checkpoint=str(tiny_model_tokenizer_dir),
        max_steps=2,
        seed=42,
        device="cpu",
        offline=True,
    )
    checkpoint = run_root / "best_composite" / "checkpoint.pt"
    assert checkpoint.is_file()

    real_read_csv = pd.read_csv
    real_read_bytes = Path.read_bytes
    real_read_text = Path.read_text
    opened_prepared_files: list[str] = []

    def guarded_read_csv(path, *args, **kwargs):
        source = Path(path)
        if prepared_root in source.parents:
            opened_prepared_files.append(source.name)
            if source.name != "valid.csv":
                raise AssertionError(f"Calibration opened non-validation data: {source}")
        return real_read_csv(path, *args, **kwargs)

    def reject_training(*args, **kwargs):
        raise AssertionError("Calibration attempted an optimizer/training step.")

    def guarded_read_bytes(path, *args, **kwargs):
        if path.name in {"train.csv", "test.csv"}:
            raise AssertionError(f"Calibration hashed non-validation data: {path}")
        return real_read_bytes(path, *args, **kwargs)

    def guarded_read_text(path, *args, **kwargs):
        if path.name in {"train.csv", "test.csv"}:
            raise AssertionError(f"Calibration parsed non-validation data: {path}")
        return real_read_text(path, *args, **kwargs)

    def reject_optimizer(*args, **kwargs):
        raise AssertionError("Calibration constructed an optimizer.")

    monkeypatch.setattr(pd, "read_csv", guarded_read_csv)
    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    monkeypatch.setattr(MultiTaskTrainer, "train_step", reject_training)
    monkeypatch.setattr(torch.optim, "AdamW", reject_optimizer)
    calibration_output = tmp_path / "calibration"
    manifest = run_multitask_calibration(
        config_path=config_path,
        checkpoint_path=checkpoint,
        prepared_root=prepared_root,
        output_dir=calibration_output,
        device="cpu",
    )

    assert opened_prepared_files == ["valid.csv", "valid.csv"]
    assert manifest["source_split"] == "validation"
    assert manifest["test_data_accessed"] is False
    assert manifest["checkpoint_selection_performed"] is False
    assert manifest["training_performed"] is False
    assert manifest["threshold_optimization_performed"] is False
    assert manifest["endpoint_order"] == list(DYNAMIC_TASKS)
    required = {
        "calibration_parameters.json",
        "calibration_metrics_by_endpoint.json",
        "calibration_metrics_summary.csv",
        "calibration_manifest.json",
    }
    assert required <= {path.name for path in calibration_output.iterdir()}

    parameters = json.loads(
        (calibration_output / "calibration_parameters.json").read_text(
            encoding="utf-8"
        )
    )
    assert parameters["endpoint_order"] == list(DYNAMIC_TASKS)
    assert list(parameters["endpoints"]) == list(DYNAMIC_TASKS)
    assert parameters["coordinated_split_manifest"]["identifier"] == (
        "synthetic-validation-only-manifest"
    )
    assert parameters["checkpoint"]["sha256"] == hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    assert parameters["config"]["sha256"] == hashlib.sha256(
        config_path.read_bytes()
    ).hexdigest()
    assert manifest["artifacts"]["calibration_manifest"] == (
        "calibration_manifest.json"
    )
    assert list(manifest["artifacts"]["validation_predictions_raw"]) == list(
        DYNAMIC_TASKS
    )
    assert list(
        manifest["artifacts"]["validation_predictions_calibrated"]
    ) == list(DYNAMIC_TASKS)
    for endpoint in DYNAMIC_TASKS:
        raw = pd.read_csv(
            calibration_output / f"validation_predictions_raw_{endpoint}.csv"
        )
        calibrated = pd.read_csv(
            calibration_output
            / f"validation_predictions_calibrated_{endpoint}.csv"
        )
        assert list(raw) == list(RAW_PREDICTION_COLUMNS)
        assert np.allclose(sigmoid(raw["raw_logit"]), raw["probability"])
        assert calibrated["calibrated_probability"].between(0.0, 1.0).all()
        assert parameters["endpoints"][endpoint]["source_split"] == "validation"
        assert parameters["endpoints"][endpoint]["fit_status"] == (
            "insufficient_class_support"
        )


def _write_dynamic_fixture(tmp_path: Path) -> tuple[Path, Path]:
    prepared_root = tmp_path / "prepared"
    smiles = ["CCO", "CCN", "CCC", "COC"]
    targets = [0, 1, 0, 1]
    for task in DYNAMIC_TASKS:
        endpoint_root = prepared_root / task
        endpoint_root.mkdir(parents=True)
        for split, filename in (("train", "train.csv"), ("validation", "valid.csv")):
            pd.DataFrame(
                {
                    "molecule_id": [
                        f"{task}-{split}-{index}" for index in range(len(smiles))
                    ],
                    "canonical_smiles": smiles,
                    "target": targets,
                    "split": [split] * len(smiles),
                }
            ).to_csv(endpoint_root / filename, index=False)
    (prepared_root / "coordinated_split_manifest.json").write_text(
        json.dumps(
            {"split_manifest_id": "synthetic-validation-only-manifest"},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    tasks_yaml = "\n".join(
        f"""  {task}:
    endpoint_id: {task}
    tdc_name: {task}
    task_group: ADME
    task_type: binary_classification
    primary_metric: roc_auc"""
        for task in DYNAMIC_TASKS
    )
    config_path = tmp_path / "dynamic-calibration.yaml"
    config_path.write_text(
        f'''schema_version: "1.0.0"
run_name: dynamic-calibration-fixture
split_track: coordinated_multitask
prepared_root: prepared
tasks:
{tasks_yaml}
split_files:
  train: train.csv
  validation: valid.csv
  test: test.csv
audit:
  enforce_exact_smiles_exclusion: true
  enforce_scaffold_exclusion: true
training:
  random_seed: 42
  task_sampling: round_robin
  class_weighted_loss: false
  train_batch_size: 2
  evaluation_batch_size: 2
  max_sequence_length: 16
  max_steps: 2
  evaluation_interval_steps: 2
  checkpoint_interval_steps: 2
  dropout: 0.0
''',
        encoding="utf-8",
    )
    return config_path, prepared_root
