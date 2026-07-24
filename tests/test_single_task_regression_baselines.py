import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest

from admet_platform.data.multitask_regression import (
    load_multitask_regression_config,
)
from admet_platform.models.multitask_regression_chemberta import (
    DEFAULT_REGRESSION_ENDPOINTS,
)
from admet_platform.training.single_task_regression_run import (
    BASELINE_SUMMARY_COLUMNS,
    build_single_task_regression_comparison,
    run_single_task_regression_baseline,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MULTITASK_CONFIG = PROJECT_ROOT / "configs" / "multitask_regression.yaml"
CONFIGS = {
    endpoint: PROJECT_ROOT
    / "configs"
    / f"single_task_regression_{endpoint}.yaml"
    for endpoint in DEFAULT_REGRESSION_ENDPOINTS
}


def test_single_task_configs_are_parity_locked_to_multitask_regression() -> None:
    multitask = load_multitask_regression_config(MULTITASK_CONFIG)
    run_names = set()
    parity_fields = (
        "random_seed",
        "encoder_learning_rate",
        "head_learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "task_sampling",
        "loss",
        "huber_delta",
        "train_batch_size",
        "evaluation_batch_size",
        "max_sequence_length",
        "model_name_or_path",
        "model_revision",
        "pooling",
        "dropout",
        "evaluation_interval_steps",
        "checkpoint_interval_steps",
        "warmup_steps",
        "warmup_ratio",
        "scheduler",
        "early_stopping_patience_evaluations",
        "minimum_training_steps_before_stopping",
        "mixed_precision",
    )

    for endpoint, path in CONFIGS.items():
        config = load_multitask_regression_config(path)
        assert tuple(config.tasks) == (endpoint,)
        assert config.tasks[endpoint] == multitask.tasks[endpoint]
        assert config.prepared_root == multitask.prepared_root
        assert config.split_files == multitask.split_files
        assert config.training.max_steps == 1000
        assert config.training.task_loss_weights == {endpoint: 1.0}
        for field in parity_fields:
            assert getattr(config.training, field) == getattr(
                multitask.training, field
            )
        run_names.add(config.run_name)
    assert len(run_names) == len(DEFAULT_REGRESSION_ENDPOINTS)


def test_wrapper_rejects_a_multitask_config_before_training(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must contain one endpoint"):
        run_single_task_regression_baseline(
            config_path=MULTITASK_CONFIG,
            prepared_root=None,
            output_dir=tmp_path / "not-created",
        )
    assert not (tmp_path / "not-created").exists()


def test_single_task_offline_smoke_writes_selected_validation_artifacts(
    tiny_model_tokenizer_dir: Path, tmp_path: Path
) -> None:
    prepared = tmp_path / "prepared"
    config_path = _single_task_fixture(tmp_path, prepared)
    output = tmp_path / "baseline"

    result = run_single_task_regression_baseline(
        config_path=config_path,
        prepared_root=prepared,
        output_dir=output,
        checkpoint=str(tiny_model_tokenizer_dir),
        max_steps=2,
        limit_samples_per_task=4,
        limit_validation_samples_per_task=3,
        device="cpu",
        offline=True,
        deterministic_algorithms=True,
        mixed_precision="no",
        evaluation_interval_steps=2,
        checkpoint_interval_steps=2,
    )

    assert result["global_step"] == 2
    assert result["endpoint"] == "vdss_lombardo"
    assert result["selected_step"] == 2
    assert result["test_data_used"] is False
    assert result["task_contributions"]["batch_counts"] == {"vdss_lombardo": 2}
    assert (output / "best_composite" / "checkpoint.pt").is_file()
    assert (output / "validation_history.json").is_file()
    assert (output / "validation_history.jsonl").is_file()
    assert (output / "single_task_baseline_summary.csv").is_file()

    summary = json.loads(
        (output / "single_task_baseline_summary.json").read_text(encoding="utf-8")
    )
    row = summary["row"]
    assert summary["source_split"] == "validation"
    assert summary["selection_primary"] == "lowest validation normalized RMSE"
    assert row["test_data_used"] is False
    assert row["scientific_transform"] == "log10"
    assert row["row_count"] == 3
    assert row["train_split_sha256"]
    assert row["validation_split_sha256"]
    assert row["initial_model_state_hash"]
    assert set(pd.read_csv(output / "single_task_baseline_summary.csv").columns) == set(
        BASELINE_SUMMARY_COLUMNS
    )
    manifest = json.loads(
        (output / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["loaded_splits"] == ["train", "validation"]
    assert manifest["test_data_used"] is False
    assert not list(prepared.rglob("test.csv"))


def test_comparison_summary_requires_and_combines_all_five_endpoints(
    tmp_path: Path,
) -> None:
    runs = {}
    for index, endpoint in enumerate(DEFAULT_REGRESSION_ENDPOINTS):
        run = tmp_path / endpoint
        run.mkdir()
        row = {
            column: False if column == "test_data_used" else index
            for column in BASELINE_SUMMARY_COLUMNS
        }
        row["endpoint"] = endpoint
        row["run_name"] = f"run-{endpoint}"
        row["selected_checkpoint"] = f"{endpoint}/best_composite/checkpoint.pt"
        row["initial_model_state_hash"] = f"hash-{endpoint}"
        row["train_split_sha256"] = f"train-{endpoint}"
        row["validation_split_sha256"] = f"validation-{endpoint}"
        row["scientific_transform"] = (
            "log10" if endpoint == "vdss_lombardo" else "identity"
        )
        (run / "single_task_baseline_summary.json").write_text(
            json.dumps(
                {
                    "source_split": "validation",
                    "row": row,
                }
            ),
            encoding="utf-8",
        )
        runs[endpoint] = run

    frame = build_single_task_regression_comparison(
        runs,
        output_csv=tmp_path / "comparison.csv",
        output_json=tmp_path / "comparison.json",
    )

    assert frame["endpoint"].tolist() == list(DEFAULT_REGRESSION_ENDPOINTS)
    payload = json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))
    assert payload["source_split"] == "validation"
    assert payload["test_data_used"] is False
    assert len(payload["rows"]) == 5


def _single_task_fixture(tmp_path: Path, prepared: Path) -> Path:
    endpoint = prepared / "vdss_lombardo"
    endpoint.mkdir(parents=True)
    train = pd.DataFrame(
        {
            "molecule_id": [f"train-{index}" for index in range(4)],
            "canonical_smiles": ["CCO", "CCN", "CCC", "COC"],
            "target_original": [0.1, 1.0, 10.0, 100.0],
            "split": ["train"] * 4,
        }
    )
    validation = pd.DataFrame(
        {
            "molecule_id": [f"validation-{index}" for index in range(3)],
            "canonical_smiles": ["CCCO", "CCCN", "CCCOC"],
            "target_original": [0.5, 2.0, 20.0],
            "split": ["validation"] * 3,
        }
    )
    train.to_csv(endpoint / "train.csv", index=False)
    validation.to_csv(endpoint / "valid.csv", index=False)
    base = load_multitask_regression_config(CONFIGS["vdss_lombardo"])
    training = asdict(base.training)
    training["mixed_precision"] = "no"
    config_path = tmp_path / "single-vdss.yaml"
    config_path.write_text(
        """schema_version: "1.0.0"
run_name: single-vdss-smoke
split_track: coordinated_multitask_regression
prepared_root: prepared
tasks:
  vdss_lombardo:
    endpoint_id: vdss_lombardo
    tdc_name: VDss_Lombardo
    task_type: regression
    target_definition: Synthetic VDss
    units: L/kg
    target_transform: log10
split_files: {train: train.csv, validation: valid.csv, test: test.csv}
training:
  random_seed: 42
  encoder_learning_rate: 0.0002
  head_learning_rate: 0.001
  weight_decay: 0.0
  gradient_clip_norm: 1.0
  task_sampling: round_robin
  loss: huber
  huber_delta: 1.0
  task_loss_weights: {vdss_lombardo: 1.0}
  model_name_or_path: unused
  pooling: masked_mean
  dropout: 0.0
  train_batch_size: 2
  evaluation_batch_size: 3
  max_sequence_length: 16
  max_steps: 2
  evaluation_interval_steps: 2
  checkpoint_interval_steps: 2
  warmup_steps: 0
  scheduler: linear_warmup_decay
  early_stopping_patience_evaluations: 0
  minimum_training_steps_before_stopping: 0
  mixed_precision: "no"
""",
        encoding="utf-8",
    )
    assert training["max_steps"] == 1000
    return config_path
