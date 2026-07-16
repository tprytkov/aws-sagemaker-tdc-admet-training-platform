from pathlib import Path

import pytest
import torch
from transformers import BertConfig, BertModel

from admet_platform.models.multitask_chemberta import (
    DEFAULT_MULTITASK_ENDPOINTS,
    MultiTaskChemBERTa,
    MultiTaskChemBERTaConfig,
)


@pytest.fixture()
def tiny_encoder_dir(tmp_path: Path) -> Path:
    torch.manual_seed(7)
    config = BertConfig(
        vocab_size=31,
        hidden_size=12,
        num_hidden_layers=1,
        num_attention_heads=3,
        intermediate_size=16,
        max_position_embeddings=32,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    encoder = BertModel(config)
    path = tmp_path / "tiny-local-encoder"
    encoder.save_pretrained(path)
    return path


@pytest.fixture()
def model(tiny_encoder_dir: Path) -> MultiTaskChemBERTa:
    torch.manual_seed(11)
    return MultiTaskChemBERTa(
        MultiTaskChemBERTaConfig(
            model_name_or_path=str(tiny_encoder_dir),
            pooling="masked_mean",
            dropout=0.2,
            local_files_only=True,
        )
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([[1, 5, 6, 0], [1, 7, 0, 0]], dtype=torch.long),
        torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.long),
    )


@pytest.mark.parametrize("task_name", DEFAULT_MULTITASK_ENDPOINTS)
def test_output_shape_for_each_task(model: MultiTaskChemBERTa, task_name: str) -> None:
    input_ids, attention_mask = _inputs()

    logits = model(input_ids, attention_mask, task_name)

    assert logits.shape == (2,)
    assert logits.dtype.is_floating_point


def test_deterministic_behavior_in_evaluation_mode(model: MultiTaskChemBERTa) -> None:
    input_ids, attention_mask = _inputs()
    model.eval()

    with torch.no_grad():
        first = model(input_ids, attention_mask, "bbb_martins")
        second = model(input_ids, attention_mask, "bbb_martins")

    torch.testing.assert_close(first, second)


def test_unknown_task_is_rejected(model: MultiTaskChemBERTa) -> None:
    input_ids, attention_mask = _inputs()

    with pytest.raises(ValueError, match="Unknown task 'caco2_wang'.*bbb_martins"):
        model(input_ids, attention_mask, "caco2_wang")


def test_attention_mask_aware_mean_pooling(model: MultiTaskChemBERTa) -> None:
    hidden = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]],
            [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])

    pooled = model.pool_hidden_state(hidden, mask)

    torch.testing.assert_close(pooled, torch.tensor([[2.0, 3.0], [2.0, 4.0]]))


def test_all_tasks_share_one_encoder_and_heads_are_independent(model: MultiTaskChemBERTa) -> None:
    assert isinstance(model.encoder, BertModel)
    assert list(model.heads) == list(DEFAULT_MULTITASK_ENDPOINTS)
    head_weight_ids = {id(model.heads[task].weight) for task in DEFAULT_MULTITASK_ENDPOINTS}
    head_bias_ids = {id(model.heads[task].bias) for task in DEFAULT_MULTITASK_ENDPOINTS}

    assert len(head_weight_ids) == 3
    assert len(head_bias_ids) == 3
    assert not any(
        head_parameter is encoder_parameter
        for head_parameter in model.task_head_parameters()
        for encoder_parameter in model.encoder_parameters()
    )


def test_gradient_routing_updates_encoder_and_selected_head_only(model: MultiTaskChemBERTa) -> None:
    input_ids, attention_mask = _inputs()
    model.zero_grad(set_to_none=True)

    logits = model(input_ids, attention_mask, "herg_karim")
    logits.sum().backward()

    assert any(parameter.grad is not None for parameter in model.encoder_parameters())
    assert all(parameter.grad is not None for parameter in model.task_head_parameters("herg_karim"))
    assert all(parameter.grad is None for parameter in model.task_head_parameters("bbb_martins"))
    assert all(parameter.grad is None for parameter in model.task_head_parameters("ames"))


def test_parameter_groups_expose_encoder_and_heads(model: MultiTaskChemBERTa) -> None:
    groups = model.parameter_groups(encoder_learning_rate=2e-5, head_learning_rate=1e-4)

    assert [group["name"] for group in groups] == ["encoder", "task_heads"]
    assert [group["lr"] for group in groups] == [2e-5, 1e-4]
    assert list(groups[0]["params"])
    assert list(groups[1]["params"])


def test_save_and_reload_reproduces_outputs(model: MultiTaskChemBERTa, tmp_path: Path) -> None:
    input_ids, attention_mask = _inputs()
    model.eval()
    with torch.no_grad():
        expected = {
            task: model(input_ids, attention_mask, task)
            for task in DEFAULT_MULTITASK_ENDPOINTS
        }

    paths = model.save_model(tmp_path / "saved-model")
    reloaded = MultiTaskChemBERTa.load_model(tmp_path / "saved-model")
    reloaded.eval()

    assert all(Path(path).exists() for path in paths.values())
    assert not (tmp_path / "saved-model" / "encoder_config" / "model.safetensors").exists()
    assert not (tmp_path / "saved-model" / "encoder_config" / "pytorch_model.bin").exists()
    assert reloaded.multitask_config.to_dict() == model.multitask_config.to_dict()
    with torch.no_grad():
        for task, expected_logits in expected.items():
            torch.testing.assert_close(
                reloaded(input_ids, attention_mask, task),
                expected_logits,
            )


def test_configuration_validation_and_json_round_trip(tiny_encoder_dir: Path, tmp_path: Path) -> None:
    config = MultiTaskChemBERTaConfig(
        model_name_or_path=str(tiny_encoder_dir),
        pooling="cls",
        dropout=0.0,
        local_files_only=True,
    )
    path = config.save(tmp_path / "model-config.json")

    assert MultiTaskChemBERTaConfig.load(path) == config
    payload = MultiTaskChemBERTaConfig.load(path)
    assert payload.tasks == DEFAULT_MULTITASK_ENDPOINTS
    assert payload.head_type == "linear"
    assert payload.head_output_size == 1
    assert payload.pooling == "cls"
    assert payload.dropout == 0.0
    with pytest.raises(ValueError, match="pooling"):
        MultiTaskChemBERTaConfig(model_name_or_path="local", pooling="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exactly"):
        MultiTaskChemBERTaConfig(model_name_or_path="local", tasks=("bbb_martins",))


def test_masked_mean_rejects_missing_or_empty_masks(model: MultiTaskChemBERTa) -> None:
    hidden = torch.ones((1, 2, 3))

    with pytest.raises(ValueError, match="attention_mask is required"):
        model.pool_hidden_state(hidden, None)
    with pytest.raises(ValueError, match="at least one unmasked token"):
        model.pool_hidden_state(hidden, torch.zeros((1, 2), dtype=torch.long))
