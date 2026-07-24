"""Endpoint-weighted losses for normalized multi-task regression targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn


@dataclass(frozen=True)
class RegressionTaskLossOutput:
    raw_losses: Mapping[str, torch.Tensor]
    combined_loss: torch.Tensor


class MultiTaskRegressionLoss(nn.Module):
    def __init__(
        self,
        task_names: tuple[str, ...],
        *,
        loss: str = "huber",
        huber_delta: float = 1.0,
        task_loss_weights: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__()
        if not task_names or len(task_names) != len(set(task_names)):
            raise ValueError("task_names must contain unique regression endpoints.")
        if loss not in {"huber", "mse"}:
            raise ValueError("Regression loss must be 'huber' or 'mse'.")
        if not huber_delta > 0:
            raise ValueError("huber_delta must be positive.")
        weights = dict(task_loss_weights or {})
        if set(weights) - set(task_names):
            raise ValueError("task_loss_weights contains unknown tasks.")
        self.task_names = task_names
        self.loss_name = loss
        self.huber_delta = float(huber_delta)
        self.task_loss_weights = {
            task: float(weights.get(task, 1.0)) for task in task_names
        }
        if any(not weight > 0 for weight in self.task_loss_weights.values()):
            raise ValueError("Every task loss weight must be positive.")

    def forward(
        self, task_name: str, predictions: torch.Tensor, labels: torch.Tensor
    ) -> RegressionTaskLossOutput:
        if task_name not in self.task_names:
            raise ValueError(f"Unknown regression loss task '{task_name}'.")
        labels = labels.to(device=predictions.device, dtype=predictions.dtype).reshape(
            predictions.shape
        )
        if self.loss_name == "huber":
            raw = nn.functional.huber_loss(
                predictions, labels, reduction="mean", delta=self.huber_delta
            )
        else:
            raw = nn.functional.mse_loss(predictions, labels, reduction="mean")
        combined = raw * self.task_loss_weights[task_name]
        if not torch.isfinite(raw) or not torch.isfinite(combined):
            raise FloatingPointError(
                f"Non-finite regression loss for task '{task_name}'."
            )
        return RegressionTaskLossOutput(
            raw_losses={task_name: raw}, combined_loss=combined
        )


__all__ = ["MultiTaskRegressionLoss", "RegressionTaskLossOutput"]
