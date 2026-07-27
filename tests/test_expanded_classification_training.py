import json
import math
from pathlib import Path

import pandas as pd
import pytest
import torch

from admet_platform.data.expanded_classification_splits import EXPECTED_ENDPOINTS
from admet_platform.models.multitask_chemberta import (
    DEFAULT_MULTITASK_ENDPOINTS,
    MultiTaskChemBERTa,
    MultiTaskChemBERTaConfig,
)
from admet_platform.models.multitask_regression_chemberta import (
    DEFAULT_REGRESSION_ENDPOINTS,
    MultiTaskRegressionChemBERTa,
    MultiTaskRegressionChemBERTaConfig,
)
from admet_platform.training.multitask_control import (
    classification_metrics,
    update_checkpoint_selection,
)
from admet_platform.training.multitask_losses import (
    MultiTaskBinaryLoss,
    calculate_binary_class_statistics,
)
from admet_platform.training.multitask_run import run_multitask_training


TEN_TASKS = tuple(EXPECTED_ENDPOINTS)


def test_dynamic_ten_head_and_unchanged_three_head_construction(
    tiny_encoder_dir: Path,
) -> None:
    ten = MultiTaskChemBERTa(
        MultiTaskChemBERTaConfig(
            model_name_or_path=str(tiny_encoder_dir),
            tasks=TEN_TASKS,
            dropout=0.0,
            local_files_only=True,
        )
    )
    three = MultiTaskChemBERTa(
        MultiTaskChemBERTaConfig(
            model_name_or_path=str(tiny_encoder_dir),
            dropout=0.0,
            local_files_only=True,
        )
    )

    assert tuple(ten.heads) == TEN_TASKS
    assert all(head.out_features == 1 for head in ten.heads.values())
    assert tuple(three.heads) == DEFAULT_MULTITASK_ENDPOINTS


def test_active_head_routing_and_active_task_only_loss(tiny_encoder_dir: Path) -> None:
    model = MultiTaskChemBERTa(
        MultiTaskChemBERTaConfig(
            model_name_or_path=str(tiny_encoder_dir),
            tasks=TEN_TASKS,
            dropout=0.0,
            local_files_only=True,
        )
    )
    active = TEN_TASKS[3]
    logits = model(
        input_ids=torch.tensor([[1, 2, 3], [3, 2, 1]]),
        attention_mask=torch.ones((2, 3), dtype=torch.long),
        task_name=active,
    )
    output = MultiTaskBinaryLoss({task: 1.0 for task in TEN_TASKS})(
        active,
        logits,
        torch.tensor([0.0, 1.0]),
    )
    output.combined_loss.backward()

    assert logits.shape == (2,)
    assert set(output.raw_losses) == {active}
    assert any(parameter.grad is not None for parameter in model.encoder.parameters())
    assert all(parameter.grad is not None for parameter in model.heads[active].parameters())
    for task in TEN_TASKS:
        if task != active:
            assert all(parameter.grad is None for parameter in model.heads[task].parameters())


def test_five_head_regression_model_remains_unchanged(tiny_encoder_dir: Path) -> None:
    model = MultiTaskRegressionChemBERTa(
        MultiTaskRegressionChemBERTaConfig(
            model_name_or_path=str(tiny_encoder_dir),
            local_files_only=True,
        )
    )
    output = model(
        input_ids=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.ones((1, 3), dtype=torch.long),
        task_name="vdss_lombardo",
    )

    assert tuple(model.heads) == DEFAULT_REGRESSION_ENDPOINTS
    assert output.shape == (1,)


def test_train_only_class_statistics_and_unweighted_loss_metadata() -> None:
    statistics = calculate_binary_class_statistics(
        {
            "first": [0, 0, 0, 1],
            "second": [0, 1, 1, 1],
        }
    )

    assert statistics["first"] == {
        "row_count": 4,
        "class_0_count": 3,
        "class_1_count": 1,
        "class_0_fraction": 0.75,
        "class_1_fraction": 0.25,
        "calculated_positive_class_weight": 3.0,
    }
    assert statistics["second"]["calculated_positive_class_weight"] == pytest.approx(
        1 / 3
    )
    unweighted = MultiTaskBinaryLoss({"first": 1.0, "second": 1.0})
    assert getattr(unweighted, "_pos_weight_first").item() == 1.0


def test_validation_metrics_and_validation_only_selection() -> None:
    metrics = classification_metrics(
        labels=torch.tensor([0, 0, 1, 1]).numpy(),
        probabilities=torch.tensor([0.1, 0.8, 0.4, 0.9]).numpy(),
    )
    assert {
        "roc_auc",
        "average_precision",
        "accuracy",
        "balanced_accuracy",
        "mcc",
        "sensitivity",
        "specificity",
        "confusion_matrix",
        "class_support",
    } <= set(metrics)

    state = {
        "best_composite": None,
        "best_mean_pr_auc": None,
        "best_endpoints": {TEN_TASKS[0]: None},
        "evaluations_without_improvement": 0,
        "evaluation_count": 0,
        "selection_events": [],
    }
    test_evaluation = {
        "global_step": 1,
        "split": "test",
        "endpoints": {TEN_TASKS[0]: {"roc_auc": 1.0, "pr_auc": 1.0}},
        "all_endpoint_roc_auc_valid": True,
        "mean_roc_auc": 1.0,
        "mean_pr_auc": 1.0,
    }
    with pytest.raises(ValueError, match="validation results only"):
        update_checkpoint_selection(state, test_evaluation, {}, (TEN_TASKS[0],))
    assert state["best_composite"] is None


def test_ten_task_cpu_diagnostic_never_opens_test_csv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tiny_model_tokenizer_dir: Path,
) -> None:
    config_path, prepared_root = _write_ten_task_fixture(tmp_path)
    real_read_csv = pd.read_csv
    opened: list[str] = []

    def guarded_read_csv(path, *args, **kwargs):
        source = Path(path)
        opened.append(source.name)
        if source.name == "test.csv":
            raise AssertionError("Training attempted to open test.csv")
        return real_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", guarded_read_csv)
    output = tmp_path / "diagnostic"
    result = run_multitask_training(
        config_path=config_path,
        prepared_root=prepared_root,
        output_dir=output,
        checkpoint=str(tiny_model_tokenizer_dir),
        max_steps=25,
        limit_samples_per_task=4,
        limit_validation_samples_per_task=4,
        seed=42,
        device="cpu",
        offline=True,
        deterministic_algorithms=True,
    )

    assert "test.csv" not in opened
    assert all(count > 0 for count in result["task_contributions"]["batch_counts"].values())
    assert all(
        math.isfinite(record["combined_loss"])
        for record in (
            json.loads(line)
            for line in (output / "training_history.jsonl").read_text().splitlines()
        )
    )
    assert set(result["validation"]) == set(TEN_TASKS)
    assert all(
        (output / f"validation_predictions_{task}.csv").is_file()
        for task in TEN_TASKS
    )
    assert (output / "checkpoint.pt").is_file()
    run_manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    dataset_manifest = json.loads(
        (output / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    selection = json.loads(
        (output / "checkpoint_selection.json").read_text(encoding="utf-8")
    )
    assert run_manifest["endpoint_order"] == list(TEN_TASKS)
    assert run_manifest["task_sampling"]["strategy"] == "temperature"
    assert run_manifest["task_sampling"]["alpha"] == 0.5
    assert run_manifest["loss_settings"]["class_weighted_loss"] is False
    assert all(
        weight == 1.0
        for weight in run_manifest["loss_settings"][
            "applied_positive_class_weights"
        ].values()
    )
    assert len(dataset_manifest["input_hashes"]) == 20
    assert not any(key.endswith("/test") for key in dataset_manifest["input_hashes"])
    assert selection["source_split"] == "validation"


def _write_ten_task_fixture(tmp_path: Path) -> tuple[Path, Path]:
    prepared_root = tmp_path / "prepared"
    rows = [
        ("CCO", 0),
        ("CCN", 1),
        ("CCC", 0),
        ("COC", 1),
    ]
    for task in TEN_TASKS:
        endpoint = prepared_root / task
        endpoint.mkdir(parents=True)
        for split, filename in (
            ("train", "train.csv"),
            ("validation", "valid.csv"),
            ("test", "test.csv"),
        ):
            pd.DataFrame(
                {
                    "molecule_id": [
                        f"{task}-{split}-{index}" for index in range(len(rows))
                    ],
                    "smiles": [smiles for smiles, _ in rows],
                    "canonical_smiles": [smiles for smiles, _ in rows],
                    "target": [target for _, target in rows],
                    "split": [split] * len(rows),
                }
            ).to_csv(endpoint / filename, index=False)

    tasks_yaml = "\n".join(
        f"""  {task}:
    endpoint_id: {task}
    tdc_name: {tdc_name}
    task_group: ADME
    task_type: binary_classification
    primary_metric: roc_auc"""
        for task, tdc_name in EXPECTED_ENDPOINTS.items()
    )
    config_path = tmp_path / "ten-task.yaml"
    config_path.write_text(
        f'''schema_version: "1.0.0"
run_name: ten-task-diagnostic
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
  task_sampling: temperature
  task_sampling_alpha: 0.5
  class_weighted_loss: false
  train_batch_size: 2
  evaluation_batch_size: 2
  max_sequence_length: 16
  max_steps: 25
  evaluation_interval_steps: 25
  checkpoint_interval_steps: 25
  dropout: 0.0
''',
        encoding="utf-8",
    )
    return config_path, prepared_root
