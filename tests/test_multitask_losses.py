import pytest
import torch

from admet_platform.training.multitask_losses import (
    MultiTaskBinaryLoss,
    calculate_positive_class_weights,
)


def test_endpoint_specific_positive_class_weights_use_training_labels() -> None:
    weights = calculate_positive_class_weights(
        {
            "bbb_martins": [0, 0, 0, 1],
            "herg_karim": [0, 1, 1],
            "ames": torch.tensor([0, 0, 1, 1]),
        }
    )

    assert weights == {"bbb_martins": 3.0, "herg_karim": 0.5, "ames": 1.0}


def test_positive_class_weight_rejects_one_class_task() -> None:
    with pytest.raises(ValueError, match="ames.*both classes"):
        calculate_positive_class_weights({"ames": [1, 1, 1]})


def test_loss_returns_raw_and_explicitly_weighted_combined_loss() -> None:
    losses = MultiTaskBinaryLoss(
        {"bbb_martins": 2.0, "herg_karim": 1.0, "ames": 1.0},
        {"bbb_martins": 0.25, "herg_karim": 1.0, "ames": 2.0},
    )
    output = losses("bbb_martins", torch.tensor([0.0, 0.0]), torch.tensor([0.0, 1.0]))

    assert set(output.raw_losses) == {"bbb_martins"}
    torch.testing.assert_close(output.combined_loss, output.raw_losses["bbb_martins"] * 0.25)


def test_nonfinite_loss_is_rejected_clearly() -> None:
    losses = MultiTaskBinaryLoss({"bbb_martins": 1.0})

    with pytest.raises(FloatingPointError, match="Non-finite loss.*bbb_martins"):
        losses("bbb_martins", torch.tensor([float("nan")]), torch.tensor([1.0]))
