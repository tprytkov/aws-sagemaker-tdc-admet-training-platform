from pathlib import Path

import pytest
import torch
from transformers import BertConfig, BertModel


@pytest.fixture()
def tiny_encoder_dir(tmp_path: Path) -> Path:
    """Create a tiny local encoder so multi-task tests never access the network."""

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
