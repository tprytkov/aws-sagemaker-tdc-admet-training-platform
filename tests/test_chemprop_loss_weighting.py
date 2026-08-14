import numpy as np
import pytest
import torch
from chemprop import nn

from admet_platform.chemprop.losses import (
    equal_task_mean_standardized_mae_metric,
    inverse_count_equal_endpoint_weights,
)


def test_inverse_count_weights_equalize_aggregate_gradient_with_unequal_labels() -> None:
    counts = [8, 2, 1]
    weights = inverse_count_equal_endpoint_weights(counts)
    predictions = torch.zeros((8, 3), requires_grad=True)
    targets = torch.ones((8, 3))
    mask = torch.zeros((8, 3), dtype=torch.bool)
    for task, count in enumerate(counts):
        mask[:count, task] = True

    loss = nn.MSE(task_weights=torch.tensor(weights, dtype=torch.float32))(
        predictions, targets, mask
    )
    loss.backward()

    aggregate_gradient = predictions.grad.abs().sum(dim=0).numpy()
    assert aggregate_gradient == pytest.approx([aggregate_gradient[0]] * 3)
    assert predictions.grad[~mask].abs().sum().item() == 0.0
    assert weights * np.asarray(counts) == pytest.approx([weights[0] * counts[0]] * 3)


def test_checkpoint_metric_is_equal_mean_of_per_task_standardized_mae() -> None:
    metric = equal_task_mean_standardized_mae_metric(3)
    predictions = torch.zeros((8, 3))
    targets = torch.tensor([[1.0, 2.0, 3.0]]).repeat(8, 1)
    mask = torch.zeros((8, 3), dtype=torch.bool)
    mask[:8, 0] = True
    mask[:2, 1] = True
    mask[:1, 2] = True
    metric.update(predictions, targets, mask)
    assert metric.compute().item() == pytest.approx((1.0 + 2.0 + 3.0) / 3.0)
