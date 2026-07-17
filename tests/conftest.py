from pathlib import Path

import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast


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


@pytest.fixture()
def tiny_model_tokenizer_dir(tmp_path: Path) -> Path:
    """Create a complete tiny local model/tokenizer checkpoint for offline CLI tests."""
    torch.manual_seed(7)
    path = tmp_path / "tiny-local-model"
    path.mkdir()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "C", "O", "N", "(", ")", "=", "1", "c"]
    (path / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(path / "vocab.txt"), do_lower_case=False)
    tokenizer.save_pretrained(path)
    config = BertConfig(
        vocab_size=len(tokenizer), hidden_size=12, num_hidden_layers=1,
        num_attention_heads=3, intermediate_size=16, max_position_embeddings=32,
        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
    )
    BertModel(config).save_pretrained(path)
    return path
