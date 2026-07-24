import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch
import yaml
from transformers import RobertaConfig, RobertaModel

from admet_platform.models import encoder_compatibility
from admet_platform.models.encoder_compatibility import (
    EncoderCompatibilityError,
    verify_encoder_compatibility,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
MODEL_NAME = "DeepChem/ChemBERTa-100M-MLM"
MODEL_REVISION = "f5c45f44d3061f0346888f5c09db17ec1146d29d"
CONFIG_PAIRS = (
    ("multitask_classification.yaml", "chemberta_100m_multitask_classification.yaml"),
    ("single_task_bbb_martins.yaml", "chemberta_100m_single_task_bbb_martins.yaml"),
    ("single_task_herg_karim.yaml", "chemberta_100m_single_task_herg_karim.yaml"),
    ("single_task_ames.yaml", "chemberta_100m_single_task_ames.yaml"),
)


def _yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


@pytest.mark.parametrize(("reference_name", "candidate_name"), CONFIG_PAIRS)
def test_candidate_config_differs_only_in_allowed_identity_fields(
    reference_name: str, candidate_name: str
) -> None:
    reference = _yaml(CONFIG_DIR / reference_name)
    candidate = _yaml(CONFIG_DIR / candidate_name)
    expected = copy.deepcopy(reference)
    expected["run_name"] = candidate["run_name"]
    expected["training"]["model_name_or_path"] = MODEL_NAME
    expected["training"]["model_revision"] = MODEL_REVISION

    assert candidate == expected
    assert candidate["training"]["model_revision"] == MODEL_REVISION


def test_candidate_run_names_are_unique() -> None:
    candidates = [_yaml(CONFIG_DIR / candidate) for _, candidate in CONFIG_PAIRS]
    references = [_yaml(CONFIG_DIR / reference) for reference, _ in CONFIG_PAIRS]
    run_names = [config["run_name"] for config in candidates + references]

    assert len(run_names) == len(set(run_names))


class MockTokenizer:
    name_or_path = "mock-roberta-tokenizer"
    model_max_length = 32
    pad_token_id = 1
    unk_token_id = 3
    cls_token_id = 0
    sep_token_id = 2
    mask_token_id = 4
    bos_token_id = 0
    eos_token_id = 2

    def __len__(self) -> int:
        return 32

    def __call__(self, smiles: list[str], **kwargs: Any) -> dict[str, torch.Tensor]:
        assert smiles == ["CCO", "c1ccccc1", "CC(=O)O"]
        assert kwargs["return_tensors"] == "pt"
        return {
            "input_ids": torch.tensor(
                [[0, 5, 6, 2, 1], [0, 7, 8, 2, 1], [0, 5, 9, 6, 2]],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [[1, 1, 1, 1, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1]],
                dtype=torch.long,
            ),
        }


def _mock_encoder() -> RobertaModel:
    torch.manual_seed(23)
    return RobertaModel(
        RobertaConfig(
            vocab_size=32,
            hidden_size=12,
            num_hidden_layers=1,
            num_attention_heads=3,
            intermediate_size=16,
            max_position_embeddings=32,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        ),
        add_pooling_layer=False,
    )


def _install_compatible_mocks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    missing_keys: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    tokenizer_call: dict[str, Any] = {}
    model_call: dict[str, Any] = {}

    def tokenizer_from_pretrained(name: str, **kwargs: Any) -> MockTokenizer:
        tokenizer_call.update({"name": name, **kwargs})
        return MockTokenizer()

    def model_from_pretrained(name: str, **kwargs: Any):
        model_call.update({"name": name, **kwargs})
        return _mock_encoder(), {
            "missing_keys": list(missing_keys or []),
            "unexpected_keys": [
                "lm_head.bias",
                "lm_head.dense.weight",
                "lm_head.layer_norm.weight",
            ],
            "mismatched_keys": [],
            "error_msgs": [],
        }

    monkeypatch.setattr(
        encoder_compatibility.AutoTokenizer, "from_pretrained", tokenizer_from_pretrained
    )
    monkeypatch.setattr(
        encoder_compatibility.AutoModel, "from_pretrained", model_from_pretrained
    )
    return tokenizer_call, model_call


def test_mocked_roberta_preflight_accepts_unused_mlm_head_and_runs_three_heads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer_call, model_call = _install_compatible_mocks(monkeypatch)
    monkeypatch.setattr(
        pd,
        "read_csv",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Preflight must not read prepared data.")
        ),
    )
    output = tmp_path / "compatibility.json"

    report = verify_encoder_compatibility(
        model_name_or_path=MODEL_NAME,
        revision=MODEL_REVISION,
        output_json=output,
    )

    assert report["status"] == "compatible"
    assert report["architecture"] == {
        "model_type": "roberta",
        "hidden_size": 12,
        "layer_count": 1,
        "attention_heads": 3,
        "vocabulary_size": 32,
        "maximum_positions": 32,
    }
    assert set(report["task_heads"]) == {"bbb_martins", "herg_karim", "ames"}
    assert all(value["logit_shape"] == [3] for value in report["task_heads"].values())
    loading = report["loading_information"]
    assert loading["expected_unused_mlm_head_keys"] == [
        "lm_head.bias",
        "lm_head.dense.weight",
        "lm_head.layer_norm.weight",
    ]
    assert loading["shared_encoder_weights_loaded"] is True
    assert loading["compatibility_problems"] == []
    assert report["parameters"]["model_total"] > report["parameters"]["encoder_total"]
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "compatible"
    assert tokenizer_call["revision"] == MODEL_REVISION
    assert model_call["output_loading_info"] is True


def test_missing_encoder_weights_fail_and_write_failure_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing_key = "embeddings.word_embeddings.weight"
    _install_compatible_mocks(monkeypatch, missing_keys=[missing_key])
    output = tmp_path / "failed-compatibility.json"

    with pytest.raises(EncoderCompatibilityError, match="Missing encoder weights"):
        verify_encoder_compatibility(
            model_name_or_path=MODEL_NAME,
            revision=MODEL_REVISION,
            output_json=output,
        )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["loading_information"]["missing_encoder_keys"] == [missing_key]
    assert report["loading_information"]["unexpected_missing_encoder_keys"] == [missing_key]
    assert report["loading_information"]["shared_encoder_weights_loaded"] is False


def test_missing_unused_pooler_weights_are_reported_but_not_treated_as_encoder_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pooler_keys = ["pooler.dense.bias", "pooler.dense.weight"]
    _install_compatible_mocks(monkeypatch, missing_keys=pooler_keys)

    report = verify_encoder_compatibility(
        model_name_or_path=MODEL_NAME,
        revision=MODEL_REVISION,
        output_json=tmp_path / "pooler-report.json",
    )

    loading = report["loading_information"]
    assert report["status"] == "compatible"
    assert loading["expected_missing_unused_pooler_keys"] == pooler_keys
    assert loading["unexpected_missing_encoder_keys"] == []
    assert loading["shared_encoder_weights_loaded"] is True


def test_local_files_only_is_forwarded_and_enables_offline_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    tokenizer_call, model_call = _install_compatible_mocks(monkeypatch)

    report = verify_encoder_compatibility(
        model_name_or_path="models/chemberta-100m-mlm",
        local_files_only=True,
        output_json=tmp_path / "offline.json",
    )

    assert report["local_files_only"] is True
    assert tokenizer_call["local_files_only"] is True
    assert model_call["local_files_only"] is True
    assert model_call["name"] == "models/chemberta-100m-mlm"
    assert encoder_compatibility.os.environ["HF_HUB_OFFLINE"] == "1"
    assert encoder_compatibility.os.environ["TRANSFORMERS_OFFLINE"] == "1"
