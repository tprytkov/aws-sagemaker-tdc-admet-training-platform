import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import admet_platform.training.expanded_locked_test_evaluation as expanded_evaluation
import admet_platform.training.multitask_calibration as calibration_module
import admet_platform.training.multitask_control as control_module
from admet_platform.training.expanded_locked_test_evaluation import (
    EXPECTED_ENDPOINT_ORDER,
    apply_frozen_calibration,
    expanded_binary_metrics,
    load_expanded_evaluation_config,
    run_expanded_locked_test_evaluation,
    stratified_bootstrap_confidence_intervals,
)
from admet_platform.training.multitask_calibration import sigmoid
from admet_platform.training.multitask_trainer import MultiTaskTrainer


def test_dry_run_validates_dynamic_ten_head_contract_without_test_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_frozen_fixture(tmp_path)
    real_open = Path.open
    real_read_text = Path.read_text

    def guarded_open(path: Path, *args, **kwargs):
        if path.name == "test.csv":
            raise AssertionError(f"Dry run opened locked test data: {path}")
        return real_open(path, *args, **kwargs)

    def guarded_read_text(path: Path, *args, **kwargs):
        if path.name == "test.csv":
            raise AssertionError(f"Dry run parsed locked test data: {path}")
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    monkeypatch.setattr(
        torch.optim,
        "AdamW",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Dry run constructed an optimizer.")
        ),
    )
    monkeypatch.setattr(
        MultiTaskTrainer,
        "train_step",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Dry run attempted a training step.")
        ),
    )
    monkeypatch.setattr(
        calibration_module,
        "fit_platt_calibrator",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Dry run attempted calibration fitting.")
        ),
    )
    monkeypatch.setattr(
        control_module,
        "update_checkpoint_selection",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Dry run attempted checkpoint selection.")
        ),
    )
    result = run_expanded_locked_test_evaluation(
        evaluation_config=config_path,
        output_dir=tmp_path / "planned-output",
        device="cpu",
        dry_run=True,
    )

    assert result["test_data_accessed"] is False
    assert result["evaluation_executed"] is False
    assert result["endpoint_order"] == list(EXPECTED_ENDPOINT_ORDER)
    assert result["number_of_model_checkpoints_evaluated"] == 1
    assert len(result["expected_output_plan"]) == 25
    freeze_identity = result["verified_hashes"]["calibration_freeze_record"]
    assert Path(freeze_identity["path"]).name == "calibration_freeze_record.json"
    assert freeze_identity["sha256"] == _digest(
        tmp_path
        / "run"
        / "validation_calibration"
        / "calibration_freeze_record.json"
    )
    assert "calibration_artifact_hashes.txt" in result["verified_hashes"][
        "calibration"
    ]
    coordinated_identity = result["verified_hashes"]["coordinated_split_manifest"]
    assert Path(coordinated_identity["path"]).name == (
        "coordinated_split_manifest.json"
    )
    assert coordinated_identity["sha256"] == _digest(
        tmp_path / "prepared" / "coordinated_split_manifest.json"
    )
    assert not (tmp_path / "planned-output").exists()


@pytest.mark.parametrize(
    ("target", "message"),
    (
        ("checkpoint", "Checkpoint SHA-256 mismatch"),
        ("training_config", "Training config SHA-256 mismatch"),
        ("calibration", "calibration_parameters.json SHA-256 mismatch"),
    ),
)
def test_dry_run_rejects_frozen_artifact_tampering(
    tmp_path: Path,
    target: str,
    message: str,
) -> None:
    config_path = _write_frozen_fixture(tmp_path)
    paths = {
        "checkpoint": tmp_path / "run" / "best_composite" / "checkpoint.pt",
        "training_config": tmp_path / "configs" / "multitask_classification_expanded.yaml",
        "calibration": tmp_path / "run" / "validation_calibration" / "calibration_parameters.json",
    }
    with paths[target].open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(ValueError, match=message):
        run_expanded_locked_test_evaluation(
            evaluation_config=config_path,
            output_dir=tmp_path / "output",
            dry_run=True,
        )


def test_tampered_calibration_freeze_record_fails_closed(tmp_path: Path) -> None:
    config_path = _write_frozen_fixture(tmp_path)
    freeze_record = (
        tmp_path
        / "run"
        / "validation_calibration"
        / "calibration_freeze_record.json"
    )
    with freeze_record.open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(ValueError, match="Calibration freeze record SHA-256 mismatch"):
        run_expanded_locked_test_evaluation(
            evaluation_config=config_path,
            output_dir=tmp_path / "output",
            dry_run=True,
        )


def test_incorrect_configured_freeze_record_sha_fails_closed(tmp_path: Path) -> None:
    config_path = _write_frozen_fixture(tmp_path)
    _replace_config_hash(config_path, "calibration_freeze_record.json", "b" * 64)

    with pytest.raises(ValueError, match="Calibration freeze record SHA-256 mismatch"):
        run_expanded_locked_test_evaluation(
            evaluation_config=config_path,
            output_dir=tmp_path / "output",
            dry_run=True,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("checkpoint", "selected_model.checkpoint_sha256 mismatch"),
        ("endpoint_order", "endpoint order mismatch"),
    ),
)
def test_freeze_record_internal_identity_mismatch_fails_closed(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    config_path = _write_frozen_fixture(tmp_path)
    freeze_record = (
        tmp_path
        / "run"
        / "validation_calibration"
        / "calibration_freeze_record.json"
    )
    payload = json.loads(freeze_record.read_text(encoding="utf-8"))
    if mutation == "checkpoint":
        payload["selected_model"]["checkpoint_sha256"] = "c" * 64
    else:
        payload["endpoint_order"][0:2] = reversed(payload["endpoint_order"][0:2])
    freeze_record.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    _replace_config_hash(
        config_path,
        "calibration_freeze_record.json",
        _digest(freeze_record),
    )

    with pytest.raises(ValueError, match=message):
        run_expanded_locked_test_evaluation(
            evaluation_config=config_path,
            output_dir=tmp_path / "output",
            dry_run=True,
        )


def test_freeze_record_embedded_artifact_hash_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    config_path = _write_frozen_fixture(tmp_path)
    metrics_summary = (
        tmp_path
        / "run"
        / "validation_calibration"
        / "calibration_metrics_summary.csv"
    )
    with metrics_summary.open("a", encoding="utf-8") as handle:
        handle.write("ames,0.2\n")

    with pytest.raises(
        ValueError,
        match="calibration_metrics_summary.csv SHA-256 mismatch",
    ):
        run_expanded_locked_test_evaluation(
            evaluation_config=config_path,
            output_dir=tmp_path / "output",
            dry_run=True,
        )


def test_endpoint_order_mismatch_is_rejected(tmp_path: Path) -> None:
    config_path = _write_frozen_fixture(tmp_path)
    text = config_path.read_text(encoding="utf-8")
    text = text.replace("  - hia_hou\n  - pgp_broccatelli", "  - pgp_broccatelli\n  - hia_hou", 1)
    config_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="endpoint order"):
        load_expanded_evaluation_config(config_path)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("random_seed", 7, "random_seed"),
        ("global_step", 2999, "global step"),
        ("task_sampling", "proportional", "task_sampling"),
        ("task_sampling_alpha", 1.0, "task_sampling_alpha"),
    ),
)
def test_checkpoint_frozen_training_metadata_mismatch_is_rejected(
    tmp_path: Path,
    field: str,
    replacement: object,
    message: str,
) -> None:
    config_path = _write_frozen_fixture(tmp_path)
    checkpoint_path = tmp_path / "run" / "best_composite" / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if field == "global_step":
        checkpoint[field] = replacement
    else:
        checkpoint["training_config"][field] = replacement
    torch.save(checkpoint, checkpoint_path)
    _replace_config_hash(config_path, "checkpoint.pt", _digest(checkpoint_path))

    with pytest.raises(ValueError, match=message):
        run_expanded_locked_test_evaluation(
            evaluation_config=config_path,
            output_dir=tmp_path / "output",
            dry_run=True,
        )


def test_multiple_checkpoint_configuration_is_rejected(tmp_path: Path) -> None:
    config_path = _write_frozen_fixture(tmp_path)
    with config_path.open("a", encoding="utf-8") as handle:
        handle.write("checkpoints: []\n")

    with pytest.raises(ValueError, match="exactly one checkpoint"):
        load_expanded_evaluation_config(config_path)


def test_frozen_platt_transformation_and_exact_hia_preservation() -> None:
    logits = np.asarray([-2.0, -0.5, 0.5, 2.0])
    probabilities = sigmoid(logits)
    predictions = pd.DataFrame(
        {
            "molecule_id": [f"mol-{index}" for index in range(4)],
            "canonical_smiles": ["CCO"] * 4,
            "target": [0, 0, 1, 1],
            "raw_logit": logits,
            "probability": probabilities,
            "prediction": (probabilities >= 0.5).astype(int),
        }
    )
    parameters = {
        "fit_status": "fitted",
        "coefficient_a": 1.5,
        "intercept_b": -0.25,
    }
    calibrated = apply_frozen_calibration(
        predictions, parameters, endpoint="bbb_martins"
    )
    hia = apply_frozen_calibration(
        predictions,
        {
            "fit_status": "insufficient_class_support",
            "coefficient_a": None,
            "intercept_b": None,
        },
        endpoint="hia_hou",
    )

    assert np.allclose(
        calibrated["calibrated_probability"], sigmoid(1.5 * logits - 0.25)
    )
    assert np.array_equal(hia["calibrated_probability"], probabilities)


def test_platt_parameters_remain_endpoint_specific_and_threshold_is_fixed() -> None:
    logits = np.asarray([-1.0, 1.0])
    probabilities = sigmoid(logits)
    predictions = pd.DataFrame(
        {
            "molecule_id": ["a", "b"],
            "canonical_smiles": ["CCO", "CCN"],
            "target": [0, 1],
            "raw_logit": logits,
            "probability": probabilities,
            "prediction": [0, 1],
        }
    )
    first = apply_frozen_calibration(
        predictions,
        {"fit_status": "fitted", "coefficient_a": 2.0, "intercept_b": 0.5},
        endpoint="bbb_martins",
    )
    second = apply_frozen_calibration(
        predictions,
        {"fit_status": "fitted", "coefficient_a": 0.5, "intercept_b": -0.5},
        endpoint="ames",
    )
    metrics = expanded_binary_metrics([0, 1], [0.49, 0.5])

    assert np.allclose(first["calibrated_probability"], sigmoid(2.0 * logits + 0.5))
    assert np.allclose(second["calibrated_probability"], sigmoid(0.5 * logits - 0.5))
    assert not np.array_equal(
        first["calibrated_probability"], second["calibrated_probability"]
    )
    assert metrics["threshold"] == 0.5
    assert metrics["confusion_matrix"] == {"tn": 1, "fp": 0, "fn": 0, "tp": 1}


def test_stratified_bootstrap_is_deterministic_and_reports_counts() -> None:
    labels = np.asarray([0] * 12 + [1] * 12)
    probabilities = np.linspace(0.05, 0.95, len(labels))
    kwargs = {
        "endpoint": "ames",
        "probability_type": "calibrated",
        "replicates": 50,
        "seed": 91,
        "confidence_level": 0.95,
    }
    first = stratified_bootstrap_confidence_intervals(labels, probabilities, **kwargs)
    second = stratified_bootstrap_confidence_intervals(labels, probabilities, **kwargs)

    assert first == second
    assert {row["metric"] for row in first} == {
        "roc_auc",
        "average_precision",
        "brier_score",
        "sensitivity",
        "specificity",
    }
    assert all(row["successful_replicates"] == 50 for row in first)
    assert all(row["rejected_replicates"] == 0 for row in first)


def test_nonempty_output_is_rejected_before_test_hashing_or_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_frozen_fixture(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    (output / "existing.txt").write_text("keep", encoding="utf-8")

    def reject_training(*args, **kwargs):
        raise AssertionError("Evaluation attempted training.")

    def reject_test_hashes(*args, **kwargs):
        raise AssertionError("Overwrite rejection occurred after test hashing.")

    monkeypatch.setattr(MultiTaskTrainer, "train_step", reject_training)
    monkeypatch.setattr(
        "admet_platform.training.expanded_locked_test_evaluation.verify_locked_test_hashes",
        reject_test_hashes,
    )
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_expanded_locked_test_evaluation(
            evaluation_config=config_path,
            output_dir=output,
            device="cpu",
            dry_run=False,
        )


def test_partial_failure_never_creates_final_output_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_frozen_fixture(tmp_path)
    output = tmp_path / "final-output"
    logits = np.asarray([-1.0, 1.0])
    probabilities = sigmoid(logits)
    frame = pd.DataFrame(
        {
            "molecule_id": ["a", "b"],
            "canonical_smiles": ["CCO", "CCN"],
            "target": [0, 1],
            "raw_logit": logits,
            "probability": probabilities,
            "prediction": [0, 1],
        }
    )
    monkeypatch.setattr(
        expanded_evaluation,
        "_load_and_validate_test_manifest_contracts",
        lambda config: _synthetic_manifest_contract(),
    )
    monkeypatch.setattr(
        expanded_evaluation,
        "_hash_locked_test_files",
        lambda config, expected: {},
    )
    monkeypatch.setattr(
        expanded_evaluation,
        "_generate_test_predictions",
        lambda config, device: (
            {endpoint: frame.copy() for endpoint in EXPECTED_ENDPOINT_ORDER},
            3000,
        ),
    )

    def fail_after_first_raw_artifact(*args, **kwargs):
        raise RuntimeError("synthetic endpoint failure")

    monkeypatch.setattr(
        expanded_evaluation,
        "apply_frozen_calibration",
        fail_after_first_raw_artifact,
    )
    with pytest.raises(RuntimeError, match="synthetic endpoint failure"):
        run_expanded_locked_test_evaluation(
            evaluation_config=config_path,
            output_dir=output,
            device="cpu",
            dry_run=False,
        )

    assert not output.exists()
    incomplete = list(tmp_path.glob(".final-output.incomplete-*"))
    assert len(incomplete) == 1
    assert not (incomplete[0] / "test_evaluation_manifest.json").exists()


def test_completed_manifest_references_actual_calibration_freeze_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_frozen_fixture(tmp_path)
    output = tmp_path / "completed-output"
    logits = np.asarray([-1.0, 1.0])
    probabilities = sigmoid(logits)
    frame = pd.DataFrame(
        {
            "molecule_id": ["a", "b"],
            "canonical_smiles": ["CCO", "CCN"],
            "target": [0, 1],
            "raw_logit": logits,
            "probability": probabilities,
            "prediction": [0, 1],
        }
    )
    monkeypatch.setattr(
        expanded_evaluation,
        "_load_and_validate_test_manifest_contracts",
        lambda config: _synthetic_manifest_contract(),
    )
    monkeypatch.setattr(
        expanded_evaluation,
        "_hash_locked_test_files",
        lambda config, expected: {},
    )
    monkeypatch.setattr(
        expanded_evaluation,
        "_generate_test_predictions",
        lambda config, device: (
            {endpoint: frame.copy() for endpoint in EXPECTED_ENDPOINT_ORDER},
            3000,
        ),
    )

    def synthetic_intervals(
        labels,
        scores,
        *,
        endpoint,
        probability_type,
        replicates,
        seed,
        confidence_level,
    ):
        metrics = expanded_binary_metrics(labels, scores)
        return [
            {
                "endpoint": endpoint,
                "probability_type": probability_type,
                "metric": metric,
                "point_estimate": metrics[metric],
                "confidence_level": confidence_level,
                "ci_lower": metrics[metric],
                "ci_upper": metrics[metric],
                "requested_replicates": replicates,
                "successful_replicates": 1,
                "rejected_replicates": 0,
                "bootstrap_seed": seed,
            }
            for metric in expanded_evaluation.CI_METRICS
        ]

    monkeypatch.setattr(
        expanded_evaluation,
        "stratified_bootstrap_confidence_intervals",
        synthetic_intervals,
    )
    run_expanded_locked_test_evaluation(
        evaluation_config=config_path,
        output_dir=output,
        device="cpu",
        dry_run=False,
    )

    manifest = json.loads(
        (output / "test_evaluation_manifest.json").read_text(encoding="utf-8")
    )
    freeze_identity = manifest["calibration_freeze_record"]
    assert Path(freeze_identity["path"]).name == "calibration_freeze_record.json"
    assert freeze_identity["sha256"] == _digest(
        tmp_path
        / "run"
        / "validation_calibration"
        / "calibration_freeze_record.json"
    )
    assert Path(
        manifest["calibration_artifacts"]["calibration_artifact_hashes.txt"]["path"]
    ).name == "calibration_artifact_hashes.txt"
    assert freeze_identity != manifest["calibration_artifacts"][
        "calibration_artifact_hashes.txt"
    ]
    assert manifest["coordinated_split_manifest"] == {
        "path": "synthetic-coordinated-manifest.json",
        "sha256": "a" * 64,
        "split_manifest_id": "synthetic-expanded-manifest",
    }
    assert manifest["test_hash_provenance"]["expected_hash_source"] == (
        "frozen_coordinated_split_manifest"
    )


def test_training_manifest_contract_is_checked_without_test_path_access(
    tmp_path: Path,
) -> None:
    config_path = _write_frozen_fixture(tmp_path)
    config = load_expanded_evaluation_config(config_path)
    contract = expanded_evaluation._load_and_validate_test_manifest_contracts(config)
    assert set(contract["expected_test_hashes"]) == set(EXPECTED_ENDPOINT_ORDER)
    assert contract["test_hash_provenance"] == {
        "expected_hash_source": "frozen_coordinated_split_manifest",
        "training_run_authenticates_manifest_sha256": True,
        "training_run_contains_endpoint_test_hashes": False,
    }

    run_manifest_path = tmp_path / "run" / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    assert not any(key.endswith("/test") for key in run_manifest["input_hashes"])
    run_manifest["seed"] = 7
    run_manifest_path.write_text(json.dumps(run_manifest) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="random seed"):
        expanded_evaluation._load_and_validate_test_manifest_contracts(config)


def test_missing_coordinated_manifest_test_hash_fails_closed(tmp_path: Path) -> None:
    config_path = _write_frozen_fixture(tmp_path)
    coordinated_path = tmp_path / "prepared" / "coordinated_split_manifest.json"
    coordinated = json.loads(coordinated_path.read_text(encoding="utf-8"))
    del coordinated["endpoints"]["hia_hou"]["splits"]["test"]
    coordinated_path.write_text(json.dumps(coordinated) + "\n", encoding="utf-8")
    _refresh_run_manifest_coordinated_sha(tmp_path)
    config = load_expanded_evaluation_config(config_path)

    with pytest.raises(ValueError, match="missing a frozen hash for hia_hou/test"):
        expanded_evaluation._load_and_validate_test_manifest_contracts(config)


def test_tampered_coordinated_manifest_sha_fails_before_test_access(
    tmp_path: Path,
) -> None:
    config_path = _write_frozen_fixture(tmp_path)
    coordinated_path = tmp_path / "prepared" / "coordinated_split_manifest.json"
    with coordinated_path.open("a", encoding="utf-8") as handle:
        handle.write(" \n")
    config = load_expanded_evaluation_config(config_path)

    with pytest.raises(ValueError, match="Coordinated split manifest SHA-256 mismatch"):
        expanded_evaluation._load_and_validate_test_manifest_contracts(config)


def test_coordinated_manifest_id_mismatch_fails_before_test_access(
    tmp_path: Path,
) -> None:
    config_path = _write_frozen_fixture(tmp_path)
    coordinated_path = tmp_path / "prepared" / "coordinated_split_manifest.json"
    coordinated = json.loads(coordinated_path.read_text(encoding="utf-8"))
    coordinated["split_manifest_id"] = "wrong-manifest-id"
    coordinated_path.write_text(json.dumps(coordinated) + "\n", encoding="utf-8")
    _refresh_run_manifest_coordinated_sha(tmp_path)
    config = load_expanded_evaluation_config(config_path)

    with pytest.raises(ValueError, match="Coordinated split manifest identifier mismatch"):
        expanded_evaluation._load_and_validate_test_manifest_contracts(config)


def test_actual_synthetic_test_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    config_path = _write_frozen_fixture(tmp_path)
    config = load_expanded_evaluation_config(config_path)
    contract = expanded_evaluation._load_and_validate_test_manifest_contracts(config)
    synthetic_test = tmp_path / "prepared" / "hia_hou" / "test.csv"
    synthetic_test.parent.mkdir()
    synthetic_test.write_text("molecule_id,target\na,0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Coordinated manifest hash mismatch"):
        expanded_evaluation._hash_locked_test_files(config, contract)


def test_inference_path_is_entered_only_after_all_hash_checks_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_frozen_fixture(tmp_path)
    events: list[str] = []

    def hashes_succeeded(config, contract):
        events.append("all_test_hashes_verified")
        return {}

    def enter_inference(config, device):
        assert events == ["all_test_hashes_verified"]
        events.append("datasets_loaders_and_inference")
        raise RuntimeError("stop after ordering assertion")

    monkeypatch.setattr(
        expanded_evaluation,
        "_hash_locked_test_files",
        hashes_succeeded,
    )
    monkeypatch.setattr(
        expanded_evaluation,
        "_generate_test_predictions",
        enter_inference,
    )
    with pytest.raises(RuntimeError, match="ordering assertion"):
        run_expanded_locked_test_evaluation(
            evaluation_config=config_path,
            output_dir=tmp_path / "output",
            device="cpu",
            dry_run=False,
        )
    assert events == ["all_test_hashes_verified", "datasets_loaders_and_inference"]


def test_manuscript_table_uses_final_metrics_bootstrap_cis_and_calibration_status() -> None:
    endpoint = EXPECTED_ENDPOINT_ORDER[0]
    metric_values = {
        "roc_auc": 0.81,
        "average_precision": 0.79,
        "brier_score": 0.18,
        "sensitivity": 0.75,
        "specificity": 0.72,
        "class_support": {"class_0": 40, "class_1": 60, "total": 100},
    }
    metrics = {
        endpoint: {
            "calibration_status": "insufficient_class_support",
            "calibrated": metric_values,
        }
    }
    bootstrap = pd.DataFrame(
        [
            {
                "endpoint": endpoint,
                "probability_type": "calibrated",
                "metric": metric,
                "ci_lower": value - 0.05,
                "ci_upper": value + 0.05,
            }
            for metric, value in metric_values.items()
            if metric != "class_support"
        ]
    )

    table = expanded_evaluation._build_manuscript_table(
        (endpoint,), metrics, bootstrap
    )
    row = table.iloc[0]
    assert row["n"] == 100
    assert row["class_0"] == 40
    assert row["class_1"] == 60
    assert row["calibration_status"] == "insufficient_class_support"
    for metric in expanded_evaluation.CI_METRICS:
        assert row[metric] == metric_values[metric]
        assert row[f"{metric}_ci_lower"] == pytest.approx(metric_values[metric] - 0.05)
        assert row[f"{metric}_ci_upper"] == pytest.approx(metric_values[metric] + 0.05)


def _write_frozen_fixture(root: Path) -> Path:
    configs = root / "configs"
    configs.mkdir()
    training_config = configs / "multitask_classification_expanded.yaml"
    task_entries = "\n".join(
        f"""  {endpoint}:
    endpoint_id: {endpoint}
    tdc_name: {endpoint}
    task_group: ADME
    task_type: binary_classification
    primary_metric: roc_auc"""
        for endpoint in EXPECTED_ENDPOINT_ORDER
    )
    training_config.write_text(
        f'''schema_version: "1.0.0"
run_name: expanded-test-fixture
split_track: coordinated_multitask
prepared_root: prepared
tasks:
{task_entries}
split_files:
  train: train.csv
  validation: valid.csv
  test: test.csv
audit:
  enforce_exact_smiles_exclusion: true
  enforce_scaffold_exclusion: true
training:
  random_seed: 42
  task_sampling: temperature
  task_sampling_alpha: 0.5
  class_weighted_loss: false
  max_steps: 3000
''',
        encoding="utf-8",
    )
    run_root = root / "run"
    checkpoint = run_root / "best_composite" / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save(
        {
            "checkpoint_version": 1,
            "global_step": 3000,
            "model_config": {"tasks": list(EXPECTED_ENDPOINT_ORDER)},
            "training_config": {
                "random_seed": 42,
                "task_sampling": "temperature",
                "task_sampling_alpha": 0.5,
            },
        },
        checkpoint,
    )
    coordinated = root / "prepared" / "coordinated_split_manifest.json"
    coordinated.parent.mkdir()
    frozen_test_hash = "a" * 64
    coordinated.write_text(
        json.dumps(
            {
                "split_manifest_id": "synthetic-expanded-manifest",
                "endpoints": {
                    endpoint: {
                        "splits": {
                            "test": {"output_csv_sha256": frozen_test_hash}
                        }
                    }
                    for endpoint in EXPECTED_ENDPOINT_ORDER
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    coordinated_digest = _digest(coordinated)
    calibration_root = run_root / "validation_calibration"
    calibration_root.mkdir()
    endpoint_parameters = {}
    for endpoint in EXPECTED_ENDPOINT_ORDER:
        fitted = endpoint != "hia_hou"
        endpoint_parameters[endpoint] = {
            "endpoint": endpoint,
            "calibration_method": "platt_scaling",
            "fit_status": "fitted" if fitted else "insufficient_class_support",
            "coefficient_a": 1.0 if fitted else None,
            "intercept_b": 0.0 if fitted else None,
        }
    checkpoint_digest = _digest(checkpoint)
    config_digest = _digest(training_config)
    shared_identity = {
        "checkpoint": {"sha256": checkpoint_digest},
        "config": {"sha256": config_digest},
        "coordinated_split_manifest": {"identifier": "synthetic-expanded-manifest"},
    }
    (calibration_root / "calibration_parameters.json").write_text(
        json.dumps(
            {
                "source_split": "validation",
                "endpoint_order": list(EXPECTED_ENDPOINT_ORDER),
                "endpoints": endpoint_parameters,
                **shared_identity,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (calibration_root / "calibration_manifest.json").write_text(
        json.dumps(
            {
                "source_split": "validation",
                "test_data_accessed": False,
                "checkpoint_selection_performed": False,
                "training_performed": False,
                "threshold_optimization_performed": False,
                "endpoint_order": list(EXPECTED_ENDPOINT_ORDER),
                **shared_identity,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (calibration_root / "calibration_metrics_by_endpoint.json").write_text(
        json.dumps({"endpoint_order": list(EXPECTED_ENDPOINT_ORDER)}) + "\n",
        encoding="utf-8",
    )
    (calibration_root / "calibration_metrics_summary.csv").write_text(
        "endpoint,brier_score\nhia_hou,0.1\n",
        encoding="utf-8",
    )
    (calibration_root / "calibration_artifact_hashes.txt").write_text(
        "synthetic frozen hash inventory\n", encoding="utf-8"
    )
    (run_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "seed": 42,
                "endpoint_order": list(EXPECTED_ENDPOINT_ORDER),
                "task_sampling": {"strategy": "temperature", "alpha": 0.5},
                "prepared_split_manifest": {
                    "split_manifest_id": "synthetic-expanded-manifest",
                    "sha256": coordinated_digest,
                },
                "output_checkpoint": {"sha256": checkpoint_digest},
                "input_hashes": {
                    f"{endpoint}/{split}": frozen_test_hash
                    for endpoint in EXPECTED_ENDPOINT_ORDER
                    for split in ("train", "validation")
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    freeze_artifact_hashes = {
        name: _digest(calibration_root / name)
        for name in (
            "calibration_parameters.json",
            "calibration_manifest.json",
            "calibration_metrics_by_endpoint.json",
            "calibration_metrics_summary.csv",
        )
    }
    freeze_record_path = calibration_root / "calibration_freeze_record.json"
    freeze_record_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "status": "frozen_for_locked_test_evaluation",
                "source_split": "validation",
                "selected_model": {
                    "checkpoint_sha256": checkpoint_digest,
                    "random_seed": 42,
                    "checkpoint_step": 3000,
                    "sampling_policy": "temperature",
                    "sampling_alpha": 0.5,
                },
                "endpoint_order": list(EXPECTED_ENDPOINT_ORDER),
                "calibration_policy": {
                    "descriptive_threshold": 0.5,
                    "threshold_optimization_performed": False,
                },
                "test_data_accessed": False,
                "training_performed": False,
                "checkpoint_selection_performed": False,
                "artifact_hashes": freeze_artifact_hashes,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    hashes = {
        name: _digest(calibration_root / name)
        for name in (
            "calibration_parameters.json",
            "calibration_manifest.json",
            "calibration_artifact_hashes.txt",
        )
    }
    evaluation_config = configs / "final_test_evaluation_expanded.yaml"
    endpoint_yaml = "\n".join(f"  - {name}" for name in EXPECTED_ENDPOINT_ORDER)
    evaluation_config.write_text(
        f'''schema_version: "1.0.0"
source_split: test
evaluation_only: true
endpoint_order:
{endpoint_yaml}
training_config:
  path: configs/multitask_classification_expanded.yaml
  sha256: {config_digest}
checkpoint:
  path: run/best_composite/checkpoint.pt
  sha256: {checkpoint_digest}
  global_step: 3000
  random_seed: 42
  task_sampling: temperature
  task_sampling_alpha: 0.5
prepared_root: prepared
coordinated_manifest:
  path: prepared/coordinated_split_manifest.json
  split_manifest_id: synthetic-expanded-manifest
training_run_manifest: run/run_manifest.json
calibration:
  root: run/validation_calibration
  method: platt_scaling
  descriptive_threshold: 0.5
  preserve_uncalibrated_endpoints:
    - hia_hou
  freeze_record:
    path: calibration_freeze_record.json
    sha256: {_digest(freeze_record_path)}
  artifact_hashes:
    calibration_parameters.json: {hashes["calibration_parameters.json"]}
    calibration_manifest.json: {hashes["calibration_manifest.json"]}
    calibration_artifact_hashes.txt: {hashes["calibration_artifact_hashes.txt"]}
bootstrap:
  replicates: 2000
  seed: 42
  confidence_level: 0.95
  strategy: endpoint_stratified
output_artifacts:
  - test_predictions_raw_<endpoint>.csv
  - test_predictions_calibrated_<endpoint>.csv
  - test_metrics_by_endpoint.json
  - test_metrics_summary.csv
  - test_bootstrap_confidence_intervals.csv
  - test_evaluation_manifest.json
  - manuscript_test_results_table.csv
''',
        encoding="utf-8",
    )
    return evaluation_config


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_manifest_contract() -> dict:
    return {
        "coordinated_split_manifest": {
            "path": "synthetic-coordinated-manifest.json",
            "sha256": "a" * 64,
            "split_manifest_id": "synthetic-expanded-manifest",
        },
        "expected_test_hashes": {
            endpoint: "a" * 64 for endpoint in EXPECTED_ENDPOINT_ORDER
        },
        "test_hash_provenance": {
            "expected_hash_source": "frozen_coordinated_split_manifest",
            "training_run_authenticates_manifest_sha256": True,
            "training_run_contains_endpoint_test_hashes": False,
        },
    }


def _refresh_run_manifest_coordinated_sha(root: Path) -> None:
    coordinated_path = root / "prepared" / "coordinated_split_manifest.json"
    run_manifest_path = root / "run" / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    run_manifest["prepared_split_manifest"]["sha256"] = _digest(coordinated_path)
    run_manifest_path.write_text(json.dumps(run_manifest) + "\n", encoding="utf-8")


def _replace_config_hash(config_path: Path, path_suffix: str, digest: str) -> None:
    lines = config_path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if line.strip().endswith(path_suffix):
            existing = lines[index + 1]
            indentation = existing[: len(existing) - len(existing.lstrip())]
            lines[index + 1] = f"{indentation}sha256: {digest}"
            break
    else:
        raise AssertionError(f"Could not locate {path_suffix} in fixture config.")
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
