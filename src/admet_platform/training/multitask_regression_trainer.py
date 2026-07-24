"""Separate trainer for normalized multi-task regression."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from admet_platform.data.multitask_regression import (
    MultiTaskRegressionTrainingConfig,
)
from admet_platform.models.multitask_regression_chemberta import (
    MultiTaskRegressionChemBERTa,
)
from admet_platform.training.multitask_regression_control import (
    initial_regression_control_state,
)
from admet_platform.training.multitask_regression_losses import (
    MultiTaskRegressionLoss,
)
from admet_platform.training.multitask_trainer import MultiTaskTrainer
from admet_platform.training.task_sampler import RoundRobinTaskSampler


class MultiTaskRegressionTrainer(MultiTaskTrainer):
    """Reuse the deterministic optimizer engine with regression checkpoint metadata."""

    def __init__(
        self,
        model: MultiTaskRegressionChemBERTa,
        train_loaders: Mapping[str, Iterable[Mapping[str, Any]]] | None,
        loss_module: MultiTaskRegressionLoss,
        config: MultiTaskRegressionTrainingConfig,
        device: str | torch.device = "cpu",
        sampler: RoundRobinTaskSampler | None = None,
        evaluation_only: bool = False,
    ) -> None:
        super().__init__(
            model=model,  # type: ignore[arg-type]
            train_loaders=train_loaders,
            loss_module=loss_module,  # type: ignore[arg-type]
            config=config,  # type: ignore[arg-type]
            device=device,
            sampler=sampler,
            evaluation_only=evaluation_only,
        )
        self.control_state = initial_regression_control_state(model.task_names)

    def save_checkpoint(self, path: str | Path) -> Path:
        if self.evaluation_only:
            raise RuntimeError("Checkpoint saving is disabled for evaluation-only trainers.")
        assert self.optimizer is not None
        assert self.scheduler is not None
        assert self.scaler is not None
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "checkpoint_version": 1,
            "checkpoint_type": "multitask_regression",
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "sampler_state": self.sampler.state_dict(),
            "loader_states": self._loader_state_dict(),
            "global_step": self.global_step,
            "logical_pass": self.sampler.logical_pass,
            "rng_state": self._rng_state_dict(),
            "training_config": asdict(self.config),
            "model_config": self.model.multitask_config.to_dict(),
            "loss_metadata": {
                "loss": self.loss_module.loss_name,
                "huber_delta": self.loss_module.huber_delta,
                "task_loss_weights": dict(self.loss_module.task_loss_weights),
            },
            "history": self.history,
            "initial_model_state_hash": self.initial_model_state_hash,
            "initial_task_head_hashes": self.initial_task_head_hashes,
            "control_state": self.control_state,
        }
        torch.save(checkpoint, destination)
        return destination

    def _validate_checkpoint_model(self, checkpoint: Mapping[str, Any]) -> None:
        if checkpoint.get("checkpoint_type") != "multitask_regression":
            raise ValueError("Checkpoint is not a multi-task regression checkpoint.")
        if checkpoint.get("model_config") != self.model.multitask_config.to_dict():
            raise ValueError("Regression checkpoint model configuration is incompatible.")
        expected_loss = {
            "loss": self.loss_module.loss_name,
            "huber_delta": self.loss_module.huber_delta,
            "task_loss_weights": dict(self.loss_module.task_loss_weights),
        }
        if checkpoint.get("loss_metadata") != expected_loss:
            raise ValueError("Regression checkpoint loss configuration is incompatible.")


__all__ = ["MultiTaskRegressionTrainer"]
