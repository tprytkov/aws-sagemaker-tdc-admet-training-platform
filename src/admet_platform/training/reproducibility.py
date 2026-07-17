"""Central reproducibility utilities for single-device training."""

from __future__ import annotations

import hashlib
import random
from typing import Any, Mapping

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic_algorithms: bool = False) -> None:
    """Seed every supported RNG before randomized training objects are created."""
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms)


def tensor_mapping_hash(state: Mapping[str, Any]) -> str:
    """Hash tensor names, shapes, dtypes, and exact CPU bytes in stable order."""
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


__all__ = ["seed_everything", "tensor_mapping_hash"]
