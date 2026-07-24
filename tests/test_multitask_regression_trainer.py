from pathlib import Path
from typing import Any

import torch

from admet_platform.data.multitask_regression import (
    MultiTaskRegressionTrainingConfig,
)
from admet_platform.models.multitask_regression_chemberta import (
    DEFAULT_REGRESSION_ENDPOINTS,
    MultiTaskRegressionChemBERTa,
    MultiTaskRegressionChemBERTaConfig,
)
from admet_platform.training.multitask_regression_losses import (
    MultiTaskRegressionLoss,
)
from admet_platform.training.multitask_regression_trainer import (
    MultiTaskRegressionTrainer,
)
from admet_platform.training.reproducibility import seed_everything


def _batch(task: str, offset: int = 0) -> dict[str, Any]:
    return {
        "input_ids": torch.tensor(
            [[1, 5 + offset, 6, 0], [1, 7 + offset, 0, 0]]
        ),
        "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
        "labels": torch.tensor([-0.5, 0.5]),
        "task_name": task,
        "molecule_id": [f"{task}-a", f"{task}-b"],
    }


def _trainer(path: Path) -> MultiTaskRegressionTrainer:
    seed_everything(19)
    model = MultiTaskRegressionChemBERTa(
        MultiTaskRegressionChemBERTaConfig(
            model_name_or_path=str(path),
            dropout=0.0,
            local_files_only=True,
        )
    )
    config = MultiTaskRegressionTrainingConfig(
        random_seed=19,
        encoder_learning_rate=2e-4,
        head_learning_rate=1e-3,
        weight_decay=0.0,
        mixed_precision="no",
        max_steps=10,
        warmup_ratio=None,
        task_loss_weights={task: 1.0 for task in DEFAULT_REGRESSION_ENDPOINTS},
    )
    return MultiTaskRegressionTrainer(
        model,
        {task: [_batch(task)] for task in DEFAULT_REGRESSION_ENDPOINTS},
        MultiTaskRegressionLoss(
            DEFAULT_REGRESSION_ENDPOINTS,
            task_loss_weights=config.task_loss_weights,
        ),
        config,
    )


def test_round_robin_all_heads_contribute_and_losses_are_finite(
    tiny_encoder_dir: Path,
) -> None:
    trainer = _trainer(tiny_encoder_dir)

    records = [trainer.train_step() for _ in range(5)]

    assert [record["task_name"] for record in records] == list(
        DEFAULT_REGRESSION_ENDPOINTS
    )
    assert trainer.sampler.batch_counts == {
        task: 1 for task in DEFAULT_REGRESSION_ENDPOINTS
    }
    assert all(torch.isfinite(torch.tensor(record["combined_loss"])) for record in records)


def test_regression_checkpoint_save_and_load_restores_shared_state(
    tiny_encoder_dir: Path, tmp_path: Path
) -> None:
    trainer = _trainer(tiny_encoder_dir)
    for _ in range(5):
        trainer.train_step()
    path = trainer.save_checkpoint(tmp_path / "checkpoint.pt")
    expected = {
        name: value.detach().clone() for name, value in trainer.model.state_dict().items()
    }

    restored = _trainer(tiny_encoder_dir)
    restored.load_checkpoint(path)

    assert restored.global_step == 5
    assert restored.sampler.state_dict() == trainer.sampler.state_dict()
    assert restored.control_state == trainer.control_state
    for name, value in restored.model.state_dict().items():
        torch.testing.assert_close(value, expected[name])
