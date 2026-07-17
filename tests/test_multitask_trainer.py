import json
from pathlib import Path
from typing import Any

import pytest
import torch

from admet_platform.data.multitask import MultiTaskTrainingConfig
from admet_platform.models.multitask_chemberta import (
    DEFAULT_MULTITASK_ENDPOINTS,
    MultiTaskChemBERTa,
    MultiTaskChemBERTaConfig,
)
from admet_platform.training.multitask_losses import MultiTaskBinaryLoss
from admet_platform.training.multitask_trainer import MultiTaskTrainer


def _batch(task: str, offset: int = 0) -> dict[str, Any]:
    return {
        "input_ids": torch.tensor([[1, 4 + offset, 5, 0], [1, 6 + offset, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
        "labels": torch.tensor([0.0, 1.0]),
        "task_name": task,
    }


def _make_trainer(
    tiny_encoder_dir: Path,
    *,
    gradient_clip_norm: float = 1.0,
    dropout: float = 0.2,
    max_steps: int = 100,
    warmup_steps: int = 0,
    mixed_precision: str = "no",
) -> MultiTaskTrainer:
    model = MultiTaskChemBERTa(
        MultiTaskChemBERTaConfig(
            model_name_or_path=str(tiny_encoder_dir),
            dropout=dropout,
            local_files_only=True,
        )
    )
    loaders = {
        task: [_batch(task, 0), _batch(task, 1)]
        for task in DEFAULT_MULTITASK_ENDPOINTS
    }
    config = MultiTaskTrainingConfig(
        random_seed=19,
        encoder_learning_rate=2e-4,
        head_learning_rate=1e-3,
        weight_decay=0.0,
        gradient_clip_norm=gradient_clip_norm,
        task_loss_weights={task: 1.0 for task in DEFAULT_MULTITASK_ENDPOINTS},
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        mixed_precision=mixed_precision,
    )
    return MultiTaskTrainer(
        model,
        loaders,
        MultiTaskBinaryLoss(
            {task: 1.0 for task in DEFAULT_MULTITASK_ENDPOINTS},
            config.task_loss_weights,
        ),
        config,
    )


def _clone_parameters(parameters: Any) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in parameters]


def test_selected_head_and_encoder_update_while_other_heads_remain_unchanged(tiny_encoder_dir: Path) -> None:
    trainer = _make_trainer(tiny_encoder_dir)
    encoder_before = _clone_parameters(trainer.model.encoder_parameters())
    selected_before = _clone_parameters(trainer.model.task_head_parameters("bbb_martins"))
    unselected_before = {
        task: _clone_parameters(trainer.model.task_head_parameters(task))
        for task in ("herg_karim", "ames")
    }

    trainer.train_step("bbb_martins", _batch("bbb_martins"))

    assert any(parameter.grad is not None for parameter in trainer.model.encoder_parameters())
    assert all(parameter.grad is not None for parameter in trainer.model.task_head_parameters("bbb_martins"))
    assert all(parameter.grad is None for parameter in trainer.model.task_head_parameters("herg_karim"))
    assert all(parameter.grad is None for parameter in trainer.model.task_head_parameters("ames"))
    assert any(not torch.equal(before, after) for before, after in zip(encoder_before, trainer.model.encoder_parameters()))
    assert any(not torch.equal(before, after) for before, after in zip(selected_before, trainer.model.task_head_parameters("bbb_martins")))
    for task, before_values in unselected_before.items():
        assert all(torch.equal(before, after) for before, after in zip(before_values, trainer.model.task_head_parameters(task)))


def test_optimizer_uses_separate_encoder_and_head_learning_rates(tiny_encoder_dir: Path) -> None:
    trainer = _make_trainer(tiny_encoder_dir)

    assert [group["name"] for group in trainer.optimizer.param_groups] == ["encoder", "task_heads"]
    assert [group["lr"] for group in trainer.optimizer.param_groups] == [2e-4, 1e-3]


def test_gradient_clipping_is_applied(monkeypatch: pytest.MonkeyPatch, tiny_encoder_dir: Path) -> None:
    trainer = _make_trainer(tiny_encoder_dir, gradient_clip_norm=0.05)
    original = torch.nn.utils.clip_grad_norm_
    calls: list[float] = []

    def recording_clip(parameters: Any, max_norm: float, **kwargs: Any) -> torch.Tensor:
        calls.append(max_norm)
        return original(parameters, max_norm, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", recording_clip)

    record = trainer.train_step("bbb_martins", _batch("bbb_martins"))

    assert calls == [0.05]
    assert record["gradient_clip_norm"] == 0.05


def test_checkpoint_save_and_resume_matches_uninterrupted_next_step(tiny_encoder_dir: Path, tmp_path: Path) -> None:
    uninterrupted = _make_trainer(tiny_encoder_dir)
    for _ in range(4):
        uninterrupted.train_step()
    checkpoint = uninterrupted.save_checkpoint(tmp_path / "checkpoint.pt")

    expected_record = uninterrupted.train_step()
    expected_state = _clone_parameters(uninterrupted.model.parameters())

    resumed = _make_trainer(tiny_encoder_dir)
    resumed.load_checkpoint(checkpoint)
    actual_record = resumed.train_step()

    assert resumed.global_step == 5
    assert actual_record["task_name"] == expected_record["task_name"] == "herg_karim"
    assert actual_record["combined_loss"] == pytest.approx(expected_record["combined_loss"], abs=1e-8)
    for expected, actual in zip(expected_state, resumed.model.parameters()):
        torch.testing.assert_close(actual, expected)
    assert resumed.sampler.state_dict() == uninterrupted.sampler.state_dict()
    assert resumed.scheduler.state_dict() == uninterrupted.scheduler.state_dict()


def test_cpu_synthetic_smoke_and_structured_metrics(tiny_encoder_dir: Path, tmp_path: Path) -> None:
    trainer = _make_trainer(tiny_encoder_dir, dropout=0.0)

    records = [trainer.train_step() for _ in range(6)]
    evaluation = trainer.evaluation_step("ames", _batch("ames"))
    metrics_path = trainer.write_metrics_json(tmp_path / "training_metrics.json")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert trainer.device.type == "cpu"
    assert all(torch.isfinite(torch.tensor(record["combined_loss"])) for record in records)
    assert evaluation["phase"] == "evaluation"
    assert metrics["global_step"] == 6
    assert metrics["batch_counts"] == {task: 2 for task in DEFAULT_MULTITASK_ENDPOINTS}
    assert len(metrics["history"]) == 6


def test_evaluation_restores_model_mode_rng_sampler_and_loader_state(tiny_encoder_dir: Path) -> None:
    trainer = _make_trainer(tiny_encoder_dir, dropout=0.2)
    trainer.model.train()
    rng_before = trainer._rng_state_dict()
    sampler_before = trainer.sampler.state_dict()
    loader_before = trainer._loader_state_dict()

    trainer.evaluation_step("ames", _batch("ames"))

    assert trainer.model.training is True
    rng_after = trainer._rng_state_dict()
    assert rng_before["python"] == rng_after["python"]
    assert rng_before["numpy"][0] == rng_after["numpy"][0]
    assert (rng_before["numpy"][1] == rng_after["numpy"][1]).all()
    assert torch.equal(rng_before["torch_cpu"], rng_after["torch_cpu"])
    assert sampler_before == trainer.sampler.state_dict()
    assert loader_before == trainer._loader_state_dict()


def test_linear_scheduler_warmup_and_decay(tiny_encoder_dir: Path) -> None:
    trainer = _make_trainer(tiny_encoder_dir, dropout=0.0, max_steps=6, warmup_steps=2)
    encoder_base = trainer.config.encoder_learning_rate
    factors = [trainer.optimizer.param_groups[0]["lr"] / encoder_base]
    for _ in range(6):
        trainer.train_step()
        factors.append(trainer.optimizer.param_groups[0]["lr"] / encoder_base)
    assert factors == pytest.approx([0.0, 0.5, 1.0, 0.75, 0.5, 0.25, 0.0])


def test_mixed_precision_is_rejected_on_cpu(tiny_encoder_dir: Path) -> None:
    with pytest.raises(ValueError, match="only on a CUDA device"):
        _make_trainer(tiny_encoder_dir, mixed_precision="fp16")
