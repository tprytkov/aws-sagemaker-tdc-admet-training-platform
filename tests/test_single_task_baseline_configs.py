from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from admet_platform.data.multitask import load_multitask_config
from admet_platform.models.multitask_chemberta import (
    MultiTaskChemBERTa,
    MultiTaskChemBERTaConfig,
)
from admet_platform.training.multitask_losses import MultiTaskBinaryLoss
from admet_platform.training.multitask_trainer import MultiTaskTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
REFERENCE_CONFIG = CONFIG_DIR / "multitask_classification.yaml"
SINGLE_TASK_CONFIGS = {
    "bbb_martins": CONFIG_DIR / "single_task_bbb_martins.yaml",
    "herg_karim": CONFIG_DIR / "single_task_herg_karim.yaml",
    "ames": CONFIG_DIR / "single_task_ames.yaml",
}
SHARED_TRAINING_FIELDS = (
    "random_seed",
    "encoder_learning_rate",
    "head_learning_rate",
    "weight_decay",
    "gradient_clip_norm",
    "task_sampling",
    "model_name_or_path",
    "model_revision",
    "pooling",
    "dropout",
    "train_batch_size",
    "evaluation_batch_size",
    "max_sequence_length",
    "allow_smiles_fallback",
    "scheduler",
)


@pytest.mark.parametrize(("task", "config_path"), SINGLE_TASK_CONFIGS.items())
def test_single_task_baseline_config_contract(task: str, config_path: Path) -> None:
    reference = load_multitask_config(REFERENCE_CONFIG)
    config = load_multitask_config(config_path)

    assert tuple(config.tasks) == (task,)
    assert config.tasks[task] == reference.tasks[task]
    assert config.split_track == "coordinated_multitask"
    assert config.prepared_root == reference.prepared_root
    assert config.split_files == reference.split_files
    for field in SHARED_TRAINING_FIELDS:
        assert getattr(config.training, field) == getattr(reference.training, field)

    assert config.training.mixed_precision == "bf16"
    assert config.training.max_steps == 1000
    assert config.training.evaluation_interval_steps == 100
    assert config.training.checkpoint_interval_steps == 100
    assert config.training.warmup_ratio == 0.1
    assert config.training.early_stopping_patience_evaluations == 5
    assert config.training.minimum_training_steps_before_stopping == 500

    configured_tasks = set(config.tasks)
    assert config.training.task_loss_weights == {task: 1.0}
    assert set(config.training.task_loss_weights or {}) <= configured_tasks
    assert set(config.training.negative_transfer_tolerance or {}) <= configured_tasks
    assert set(config.training.endpoint_minimum_roc_auc or {}) <= configured_tasks


def _batch(task: str) -> dict[str, Any]:
    return {
        "input_ids": torch.tensor([[1, 5, 6, 0], [1, 7, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
        "labels": torch.tensor([0.0, 1.0]),
        "task_name": task,
    }


@pytest.mark.parametrize(("task", "config_path"), SINGLE_TASK_CONFIGS.items())
def test_shared_trainer_accepts_each_single_task_config(
    task: str, config_path: Path, tiny_encoder_dir: Path
) -> None:
    config = load_multitask_config(config_path)
    model = MultiTaskChemBERTa(
        MultiTaskChemBERTaConfig(
            model_name_or_path=str(tiny_encoder_dir),
            tasks=tuple(config.tasks),
            pooling=config.training.pooling,
            dropout=config.training.dropout,
            local_files_only=True,
        )
    )
    cpu_config = replace(config.training, mixed_precision="no")
    trainer = MultiTaskTrainer(
        model,
        {task: [_batch(task)]},
        MultiTaskBinaryLoss({task: 1.0}, cpu_config.task_loss_weights),
        cpu_config,
    )

    assert trainer.model.task_names == (task,)
    assert trainer.sampler.task_names == (task,)
    assert set(trainer.train_loaders) == {task}
