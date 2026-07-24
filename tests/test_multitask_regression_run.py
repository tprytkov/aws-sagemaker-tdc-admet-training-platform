import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from admet_platform.data.multitask_regression import (
    load_multitask_regression_config,
)
from admet_platform.models.multitask_regression_chemberta import (
    DEFAULT_REGRESSION_ENDPOINTS,
)
from admet_platform.training.multitask_regression_run import (
    run_multitask_regression_training,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_CONFIG = PROJECT_ROOT / "configs" / "multitask_regression.yaml"


def test_production_config_freezes_milestone_two_transforms() -> None:
    config = load_multitask_regression_config(PRODUCTION_CONFIG)

    assert tuple(config.tasks) == DEFAULT_REGRESSION_ENDPOINTS
    assert {
        task: endpoint.target_transform for task, endpoint in config.tasks.items()
    } == {
        "caco2_wang": "identity",
        "lipophilicity_astrazeneca": "identity",
        "solubility_aqsoldb": "identity",
        "ppbr_az": "identity",
        "vdss_lombardo": "log10",
    }
    assert "no additional logarithmic transform" in (
        config.tasks["caco2_wang"].provenance_note
    )
    assert "1614" in config.tasks["ppbr_az"].provenance_note
    assert config.training.loss == "huber"
    assert config.training.mixed_precision == "bf16"


def test_five_task_offline_cpu_smoke_never_opens_test_split(
    tiny_model_tokenizer_dir: Path, tmp_path: Path
) -> None:
    prepared = tmp_path / "prepared"
    config_path = _fixture_config(tmp_path, prepared)
    output = tmp_path / "run"

    result = run_multitask_regression_training(
        config_path=config_path,
        prepared_root=prepared,
        output_dir=output,
        checkpoint=str(tiny_model_tokenizer_dir),
        max_steps=5,
        limit_validation_samples_per_task=2,
        device="cpu",
        offline=True,
        deterministic_algorithms=True,
        mixed_precision="no",
    )

    assert result["global_step"] == 5
    assert result["test_data_used"] is False
    assert result["task_contributions"]["batch_counts"] == {
        task: 1 for task in DEFAULT_REGRESSION_ENDPOINTS
    }
    assert (output / "best_composite" / "checkpoint.pt").is_file()
    assert (output / "latest" / "checkpoint.pt").is_file()
    checkpoint = torch.load(
        output / "latest" / "checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["checkpoint_type"] == "multitask_regression"
    assert all(
        f"heads.{task}.weight" in checkpoint["model_state"]
        for task in DEFAULT_REGRESSION_ENDPOINTS
    )
    manifest = json.loads(
        (output / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["loaded_splits"] == ["train", "validation"]
    assert manifest["test_data_used"] is False
    assert not list(prepared.rglob("test.csv"))

    selection = json.loads(
        (output / "checkpoint_selection.json").read_text(encoding="utf-8")
    )
    assert selection["source_split"] == "validation"
    assert selection["shared_checkpoint_only"] is True
    assert "normalized RMSE" in selection["primary"]

    transforms = json.loads(
        (output / "target_transforms.json").read_text(encoding="utf-8")
    )
    assert transforms["fit_split"] == "train"
    assert transforms["test_statistics_used"] is False
    vdss_metadata = transforms["endpoints"]["vdss_lombardo"]
    vdss_predictions = pd.read_csv(
        output / "validation_predictions_vdss_lombardo.csv"
    )
    expected_original = 10 ** (
        vdss_predictions["prediction_normalized"]
        * vdss_metadata["transformed_train_std"]
        + vdss_metadata["transformed_train_mean"]
    )
    assert vdss_predictions["prediction_original"].to_numpy() == pytest.approx(
        expected_original.to_numpy()
    )
    required = {
        "molecule_id",
        "canonical_smiles",
        "target_original",
        "prediction_original",
        "residual_original",
        "target_normalized",
        "prediction_normalized",
    }
    for task in DEFAULT_REGRESSION_ENDPOINTS:
        predictions = pd.read_csv(output / f"validation_predictions_{task}.csv")
        assert set(predictions.columns) == required
        assert len(predictions) == 2
    summary = json.loads(
        (output / "final_run_summary.json").read_text(encoding="utf-8")
    )
    for metrics in summary["last_validation"]["endpoints"].values():
        assert metrics["row_count"] == 2
        assert metrics["normalized_rmse"] >= 0
        assert metrics["normalized_mae"] >= 0
        assert metrics["validation_loss"] >= 0


def _fixture_config(tmp_path: Path, prepared: Path) -> Path:
    definitions = {
        "caco2_wang": ("Caco2_Wang", "log cm/s", "identity", [-6.0, -5.5, -5.0, -4.5]),
        "lipophilicity_astrazeneca": (
            "Lipophilicity_AstraZeneca",
            "log-ratio",
            "identity",
            [0.5, 1.0, 2.0, 3.0],
        ),
        "solubility_aqsoldb": (
            "Solubility_AqSolDB",
            "log mol/L",
            "identity",
            [-4.0, -3.0, -2.0, -1.0],
        ),
        "ppbr_az": ("PPBR_AZ", "percent bound", "identity", [20.0, 40.0, 70.0, 90.0]),
        "vdss_lombardo": ("VDss_Lombardo", "L/kg", "log10", [0.1, 1.0, 10.0, 100.0]),
    }
    smiles = ["CCO", "CCN", "CCC", "COC", "CCCO", "CCCN"]
    task_yaml = []
    weight_yaml = []
    for task, (tdc_name, units, transform, values) in definitions.items():
        endpoint = prepared / task
        endpoint.mkdir(parents=True)
        train = pd.DataFrame(
            [
                {
                    "molecule_id": f"{task}-train-{index}",
                    "canonical_smiles": smiles[index],
                    "target_original": value,
                    "split": "train",
                }
                for index, value in enumerate(values)
            ]
        )
        validation_values = [values[1], values[2]]
        validation = pd.DataFrame(
            [
                {
                    "molecule_id": f"{task}-validation-{index}",
                    "canonical_smiles": smiles[index + 4],
                    "target_original": value,
                    "split": "validation",
                }
                for index, value in enumerate(validation_values)
            ]
        )
        train.to_csv(endpoint / "train.csv", index=False)
        validation.to_csv(endpoint / "valid.csv", index=False)
        task_yaml.append(
            f"""  {task}:
    endpoint_id: {task}
    tdc_name: {tdc_name}
    task_type: regression
    target_definition: Synthetic {task}
    units: {units}
    target_transform: {transform}"""
        )
        weight_yaml.append(f"    {task}: 1.0")
    path = tmp_path / "regression.yaml"
    path.write_text(
        f"""schema_version: "1.0.0"
run_name: offline-smoke
split_track: coordinated_multitask_regression
prepared_root: prepared
tasks:
{chr(10).join(task_yaml)}
split_files:
  train: train.csv
  validation: valid.csv
  test: test.csv
training:
  random_seed: 42
  encoder_learning_rate: 0.0002
  head_learning_rate: 0.001
  weight_decay: 0.0
  gradient_clip_norm: 1.0
  task_sampling: round_robin
  loss: huber
  huber_delta: 1.0
  task_loss_weights:
{chr(10).join(weight_yaml)}
  model_name_or_path: unused
  pooling: masked_mean
  dropout: 0.0
  train_batch_size: 2
  evaluation_batch_size: 2
  max_sequence_length: 16
  max_steps: 5
  evaluation_interval_steps: 5
  checkpoint_interval_steps: 5
  warmup_steps: 0
  scheduler: linear_warmup_decay
  early_stopping_patience_evaluations: 0
  minimum_training_steps_before_stopping: 0
  mixed_precision: "no"
""",
        encoding="utf-8",
    )
    return path
