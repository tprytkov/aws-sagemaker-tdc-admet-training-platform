"""Endpoint-specific weighted binary losses for multi-task classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
from torch import nn


@dataclass(frozen=True)
class TaskLossOutput:
    """Raw endpoint loss and its configured contribution to optimization."""

    raw_losses: Mapping[str, torch.Tensor]
    combined_loss: torch.Tensor


def calculate_positive_class_weights(
    training_labels: Mapping[str, Iterable[float] | torch.Tensor],
) -> dict[str, float]:
    """Calculate negative/positive ratios using training labels only."""

    weights: dict[str, float] = {}
    for task_name, values in training_labels.items():
        labels = torch.as_tensor(list(values) if not isinstance(values, torch.Tensor) else values).reshape(-1)
        if labels.numel() == 0:
            raise ValueError(f"Training labels for task '{task_name}' are empty.")
        if not torch.isfinite(labels.float()).all() or not torch.isin(labels, torch.tensor([0, 1], device=labels.device)).all():
            raise ValueError(f"Training labels for task '{task_name}' must contain only finite binary values 0 and 1.")
        positives = int((labels == 1).sum().item())
        negatives = int((labels == 0).sum().item())
        if positives == 0 or negatives == 0:
            raise ValueError(
                f"Training labels for task '{task_name}' must contain both classes; "
                f"found negatives={negatives}, positives={positives}."
            )
        weights[task_name] = negatives / positives
    if not weights:
        raise ValueError("At least one task is required to calculate class weights.")
    return weights


class MultiTaskBinaryLoss(nn.Module):
    """Route logits through one weighted BCEWithLogitsLoss per endpoint."""

    def __init__(
        self,
        positive_class_weights: Mapping[str, float],
        task_loss_weights: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__()
        if not positive_class_weights:
            raise ValueError("positive_class_weights must define at least one task.")
        self.task_names = tuple(positive_class_weights)
        task_weights = dict(task_loss_weights or {})
        unknown = sorted(set(task_weights) - set(self.task_names))
        if unknown:
            raise ValueError(f"Unknown task loss weight(s): {', '.join(unknown)}.")
        self.task_loss_weights = {task: float(task_weights.get(task, 1.0)) for task in self.task_names}
        if any(not weight > 0 for weight in self.task_loss_weights.values()):
            raise ValueError("Every task loss weight must be positive.")
        for task, weight in positive_class_weights.items():
            value = float(weight)
            if not torch.isfinite(torch.tensor(value)) or value <= 0:
                raise ValueError(f"Positive-class weight for task '{task}' must be finite and positive.")
            self.register_buffer(f"_pos_weight_{task}", torch.tensor([value], dtype=torch.float32))

    def forward(self, task_name: str, logits: torch.Tensor, labels: torch.Tensor) -> TaskLossOutput:
        if task_name not in self.task_names:
            raise ValueError(f"Unknown loss task '{task_name}'.")
        labels = labels.to(device=logits.device, dtype=logits.dtype).reshape(logits.shape)
        criterion = nn.BCEWithLogitsLoss(pos_weight=getattr(self, f"_pos_weight_{task_name}"))
        raw_loss = criterion(logits, labels)
        combined_loss = raw_loss * self.task_loss_weights[task_name]
        if not torch.isfinite(raw_loss) or not torch.isfinite(combined_loss):
            raise FloatingPointError(f"Non-finite loss detected for task '{task_name}'.")
        return TaskLossOutput(raw_losses={task_name: raw_loss}, combined_loss=combined_loss)


__all__ = ["MultiTaskBinaryLoss", "TaskLossOutput", "calculate_positive_class_weights"]
