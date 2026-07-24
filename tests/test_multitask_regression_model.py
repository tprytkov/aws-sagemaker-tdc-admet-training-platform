from pathlib import Path

import pytest
import torch

from admet_platform.models.multitask_regression_chemberta import (
    DEFAULT_REGRESSION_ENDPOINTS,
    MultiTaskRegressionChemBERTa,
    MultiTaskRegressionChemBERTaConfig,
)
from admet_platform.training.multitask_regression_losses import (
    MultiTaskRegressionLoss,
)
from admet_platform.training.reproducibility import (
    seed_everything,
    tensor_mapping_hash,
)


def _model(path: Path) -> MultiTaskRegressionChemBERTa:
    return MultiTaskRegressionChemBERTa(
        MultiTaskRegressionChemBERTaConfig(
            model_name_or_path=str(path),
            dropout=0.0,
            local_files_only=True,
        )
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([[1, 5, 6, 0], [1, 7, 0, 0]]),
        torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
    )


@pytest.mark.parametrize("task", DEFAULT_REGRESSION_ENDPOINTS)
def test_every_regression_head_outputs_one_unsquashed_scalar(
    tiny_encoder_dir: Path, task: str
) -> None:
    model = _model(tiny_encoder_dir)
    with torch.no_grad():
        model.heads[task].weight.zero_()
        model.heads[task].bias.fill_(2.5)

    output = model(*_inputs(), task)

    assert output.shape == (2,)
    torch.testing.assert_close(output, torch.tensor([2.5, 2.5]))
    assert torch.all(output > 1.0), "A sigmoid must not be applied to regression output."


def test_offline_local_model_save_reload_and_deterministic_hash(
    tiny_encoder_dir: Path, tmp_path: Path
) -> None:
    seed_everything(42)
    first = _model(tiny_encoder_dir)
    first_hash = tensor_mapping_hash(first.state_dict())
    seed_everything(42)
    second = _model(tiny_encoder_dir)

    assert tensor_mapping_hash(second.state_dict()) == first_hash
    expected = first(*_inputs(), "caco2_wang")
    first.save_model(tmp_path / "saved")
    restored = MultiTaskRegressionChemBERTa.load_model(tmp_path / "saved")
    actual = restored(*_inputs(), "caco2_wang")
    torch.testing.assert_close(actual, expected)


def test_huber_and_mse_losses_use_normalized_scalar_predictions() -> None:
    predictions = torch.tensor([0.0, 2.0])
    labels = torch.tensor([0.0, 0.0])
    huber = MultiTaskRegressionLoss(
        ("task",),
        loss="huber",
        huber_delta=1.0,
        task_loss_weights={"task": 2.0},
    )
    mse = MultiTaskRegressionLoss(("task",), loss="mse")

    huber_output = huber("task", predictions, labels)
    mse_output = mse("task", predictions, labels)

    assert huber_output.raw_losses["task"].item() == pytest.approx(0.75)
    assert huber_output.combined_loss.item() == pytest.approx(1.5)
    assert mse_output.raw_losses["task"].item() == pytest.approx(2.0)
