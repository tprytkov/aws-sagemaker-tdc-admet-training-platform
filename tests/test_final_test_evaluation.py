import copy
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from transformers import AutoTokenizer

from admet_platform.data.multitask import (
    EndpointDatasetSplits,
    MultiTaskEndpointConfig,
    MultiTaskTrainingConfig,
    build_task_dataloaders,
)
from admet_platform.models.multitask_chemberta import (
    MultiTaskChemBERTa,
    MultiTaskChemBERTaConfig,
)
from admet_platform.training.multitask_control import evaluate_split
from admet_platform.training.multitask_final_evaluation import (
    FinalTestExperiment,
    build_final_test_comparison,
    load_final_test_evaluation_config,
    verify_test_dataset_hashes,
)
from admet_platform.training.multitask_losses import MultiTaskBinaryLoss
from admet_platform.training.multitask_trainer import MultiTaskTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_CONFIG = PROJECT_ROOT / "configs" / "final_test_evaluation.yaml"
EXPECTED_CHECKPOINTS = {
    "bbb_single_task": (
        "outputs/local/multitask/baselines/"
        "bbb_martins_run1/best_composite/checkpoint.pt"
    ),
    "herg_single_task": (
        "outputs/local/multitask/baselines/"
        "herg_karim_run1/best_composite/checkpoint.pt"
    ),
    "ames_single_task": (
        "outputs/local/multitask/baselines/ames_run1/best_composite/checkpoint.pt"
    ),
    "multitask": (
        "outputs/local/multitask/"
        "multitask_run2_3000/best_composite/checkpoint.pt"
    ),
}


def test_selected_checkpoint_paths_are_explicit() -> None:
    config = load_final_test_evaluation_config(FINAL_CONFIG)

    assert set(config.experiments) == set(EXPECTED_CHECKPOINTS)
    for name, relative_path in EXPECTED_CHECKPOINTS.items():
        assert config.experiments[name].checkpoint == (PROJECT_ROOT / relative_path).resolve()
        assert config.experiments[name].checkpoint.name == "checkpoint.pt"
        assert config.experiments[name].checkpoint.parent.name == "best_composite"


def _frame(task: str, split: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "molecule_id": [f"{task}-{split}-0", f"{task}-{split}-1"],
            "canonical_smiles": ["CCO", "CCN"],
            "model_smiles": ["CCO", "CCN"],
            "target": [0, 1],
            "split": [split, split],
        }
    )


def _datasets(tmp_path: Path, task: str) -> dict[str, EndpointDatasetSplits]:
    endpoint = MultiTaskEndpointConfig(
        endpoint_id=task,
        tdc_name="BBB_Martins",
        task_group="ADME",
        task_type="binary_classification",
        primary_metric="roc_auc",
    )
    return {
        task: EndpointDatasetSplits(
            endpoint=endpoint,
            train=_frame(task, "train"),
            validation=_frame(task, "validation"),
            test=_frame(task, "test"),
            paths={
                "train": tmp_path / "train.csv",
                "validation": tmp_path / "valid.csv",
                "test": tmp_path / "test.csv",
            },
        )
    }


def _batch(task: str) -> dict:
    return {
        "input_ids": torch.tensor([[1, 5, 6, 0], [1, 7, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
        "labels": torch.tensor([0.0, 1.0]),
        "task_name": task,
        "molecule_id": [f"{task}-train-0", f"{task}-train-1"],
        "canonical_smiles": ["CCO", "CCN"],
    }


def test_evaluation_only_path_uses_test_loader_without_optimizer_or_selection_changes(
    tmp_path: Path,
    tiny_encoder_dir: Path,
    tiny_model_tokenizer_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = "bbb_martins"
    model_config = MultiTaskChemBERTaConfig(
        model_name_or_path=str(tiny_encoder_dir),
        tasks=(task,),
        dropout=0.0,
        local_files_only=True,
    )
    training_config = MultiTaskTrainingConfig(
        task_loss_weights={task: 1.0}, mixed_precision="no"
    )
    loss = MultiTaskBinaryLoss({task: 1.0}, {task: 1.0})
    checkpoint_trainer = MultiTaskTrainer(
        MultiTaskChemBERTa(model_config),
        {task: [_batch(task)]},
        loss,
        training_config,
    )
    checkpoint_trainer.control_state["best_composite"] = 0.99
    checkpoint = checkpoint_trainer.save_checkpoint(tmp_path / "checkpoint.pt")

    def reject_optimizer(*args, **kwargs):
        raise AssertionError("Evaluation must not construct an optimizer.")

    monkeypatch.setattr(torch.optim, "AdamW", reject_optimizer)
    evaluator = MultiTaskTrainer(
        MultiTaskChemBERTa(model_config),
        None,
        MultiTaskBinaryLoss({task: 1.0}, {task: 1.0}),
        training_config,
        evaluation_only=True,
    )
    selection_before = copy.deepcopy(evaluator.control_state)
    evaluator.load_checkpoint_for_evaluation(checkpoint)
    assert evaluator.control_state == selection_before
    assert evaluator.optimizer is None
    assert evaluator.scheduler is None
    assert evaluator.scaler is None
    with pytest.raises(RuntimeError, match="disabled"):
        evaluator.train_step()

    tokenizer = AutoTokenizer.from_pretrained(
        tiny_model_tokenizer_dir, local_files_only=True
    )
    loaders = build_task_dataloaders(
        _datasets(tmp_path, task),
        tokenizer,
        seed=42,
        train_batch_size=2,
        evaluation_batch_size=2,
        max_length=16,
        splits=("test",),
    )
    assert set(loaders[task]) == {"test"}
    evaluation = evaluate_split(
        evaluator,
        {task: loaders[task]["test"]},
        tmp_path / "evaluation",
        evaluator.global_step,
        split="test",
    )

    predictions = pd.read_csv(
        tmp_path / "evaluation" / f"test_predictions_{task}.csv"
    )
    assert evaluation["split"] == "test"
    assert evaluation["endpoints"][task]["row_count"] == 2
    assert {
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
        "f1",
        "mcc",
        "sensitivity",
        "specificity",
        "confusion_matrix",
        "row_count",
    } <= set(evaluation["endpoints"][task])
    assert predictions["molecule_id"].str.contains("-test-").all()
    assert not predictions["molecule_id"].str.contains("-train-|-validation-").any()
    assert evaluator.control_state == selection_before


def test_test_hashes_must_match_coordinated_and_training_manifests(
    tmp_path: Path,
) -> None:
    prepared_root = tmp_path / "coordinated"
    test_path = prepared_root / "bbb_martins" / "test.csv"
    test_path.parent.mkdir(parents=True)
    test_path.write_text("molecule_id,target\na,0\n", encoding="utf-8")
    digest = hashlib.sha256(test_path.read_bytes()).hexdigest()
    coordinated_manifest = prepared_root / "coordinated_split_manifest.json"
    coordinated_manifest.write_text(
        json.dumps(
            {
                "split_track": "coordinated_multitask",
                "output_file_sha256": {"bbb_martins/test": digest},
            }
        ),
        encoding="utf-8",
    )
    run_root = tmp_path / "run"
    checkpoint = run_root / "best_composite" / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    (run_root / "dataset_manifest.json").write_text(
        json.dumps({"input_hashes": {"bbb_martins/test": digest}}),
        encoding="utf-8",
    )
    experiment = FinalTestExperiment(
        name="bbb_single_task",
        role="single_task",
        endpoint="bbb_martins",
        training_config=PROJECT_ROOT / "configs" / "single_task_bbb_martins.yaml",
        checkpoint=checkpoint,
    )

    verified = verify_test_dataset_hashes(
        experiment=experiment,
        prepared_root=prepared_root,
        coordinated_manifest=coordinated_manifest,
    )
    assert verified["bbb_martins"]["sha256"] == digest

    test_path.write_text("molecule_id,target\na,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Coordinated manifest hash mismatch"):
        verify_test_dataset_hashes(
            experiment=experiment,
            prepared_root=prepared_root,
            coordinated_manifest=coordinated_manifest,
        )


def test_comparison_reports_multitask_minus_single_task_roc_auc() -> None:
    results: dict[str, dict] = {"multitask": {"metrics": {}}}
    names = {
        "bbb_martins": "bbb_single_task",
        "herg_karim": "herg_single_task",
        "ames": "ames_single_task",
    }
    for index, (task, experiment_name) in enumerate(names.items()):
        single_roc = 0.70 + index * 0.01
        metric = {
            "roc_auc": single_roc,
            "pr_auc": 0.6,
            "balanced_accuracy": 0.5,
            "f1": 0.4,
            "mcc": 0.1,
            "sensitivity": 0.8,
            "specificity": 0.2,
            "confusion_matrix": {"tn": 1, "fp": 2, "fn": 3, "tp": 4},
            "row_count": 10,
        }
        results[experiment_name] = {"metrics": {task: metric}}
        results["multitask"]["metrics"][task] = {**metric, "roc_auc": single_roc + 0.05}

    comparison = build_final_test_comparison(results).set_index("endpoint")

    assert comparison.loc["bbb_martins", "single_task_roc_auc"] == pytest.approx(0.70)
    assert comparison.loc["bbb_martins", "multitask_roc_auc"] == pytest.approx(0.75)
    assert comparison.loc[
        "bbb_martins", "delta_roc_auc_multitask_minus_single_task"
    ] == pytest.approx(0.05)
