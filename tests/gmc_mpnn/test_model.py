from __future__ import annotations

import importlib.metadata
from types import SimpleNamespace
from typing import Any

import pytest

from admet_platform.gmc_mpnn.model import (
    BOND_FEATURE_DIM,
    GGL_ATOM_FEATURE_DIM,
    ORDINARY_ATOM_FEATURE_DIM,
    TOTAL_ATOM_FEATURE_DIM,
    GMCMPNNArchitecture,
    build_gmc_mpnn_model,
)


class _Captured:
    def __init__(self, *args: Any, **kwargs: Any):
        self.args = args
        self.kwargs = kwargs


class _Featurizer:
    def __init__(self, *, extra_atom_fdim: int):
        self.extra_atom_fdim = extra_atom_fdim
        self.atom_fdim = ORDINARY_ATOM_FEATURE_DIM + extra_atom_fdim
        self.bond_fdim = BOND_FEATURE_DIM


def _fake_chemprop(version: str = "2.1.0") -> SimpleNamespace:
    return SimpleNamespace(
        __version__=version,
        featurizers=SimpleNamespace(SimpleMoleculeMolGraphFeaturizer=_Featurizer),
        nn=SimpleNamespace(
            BondMessagePassing=_Captured,
            NormAggregation=_Captured,
            BinaryClassificationFFN=_Captured,
        ),
        models=SimpleNamespace(MPNN=_Captured),
    )


def test_released_bbb_architecture_maps_to_exact_chemprop_arguments() -> None:
    bundle = build_gmc_mpnn_model(chemprop_module=_fake_chemprop())
    architecture = bundle.architecture
    message_passing, aggregation, predictor = bundle.model.args

    assert architecture.ordinary_atom_feature_dim == 72
    assert architecture.extra_atom_feature_dim == GGL_ATOM_FEATURE_DIM == 6
    assert architecture.atom_input_dim == 78
    assert architecture.bond_input_dim == 14
    assert bundle.featurizer.atom_fdim == TOTAL_ATOM_FEATURE_DIM
    assert bundle.featurizer.bond_fdim == BOND_FEATURE_DIM
    assert message_passing.kwargs == {
        "d_v": 78,
        "d_e": 14,
        "d_h": 300,
        "depth": 5,
        "dropout": 0.0,
        "activation": "RELU",
        "bias": False,
        "undirected": False,
    }
    assert aggregation.kwargs == {"norm": 57.0}
    assert predictor.kwargs == {
        "n_tasks": 1,
        "input_dim": 300,
        "hidden_dim": 900,
        "n_layers": 2,
        "dropout": 0.0,
        "activation": "LEAKYRELU",
    }
    assert bundle.model.kwargs == {
        "batch_norm": False,
        "warmup_epochs": 2,
        "init_lr": 1e-4,
        "max_lr": 1e-3,
        "final_lr": 1e-4,
    }
    assert architecture.loss == "binary_cross_entropy_with_logits"
    assert architecture.batch_size == 32
    assert architecture.maximum_epochs == 100
    assert architecture.checkpoint_monitor == "val_loss"
    assert architecture.checkpoint_mode == "min"
    assert architecture.checkpoint_save_top_k == 1
    assert architecture.early_stopping_patience == 10
    assert architecture.optimizer == "Adam"
    assert architecture.scheduler == "Chemprop Noam-like"


def test_model_factory_rejects_non_2_1_chemprop() -> None:
    with pytest.raises(RuntimeError, match="exactly Chemprop 2.1.0"):
        build_gmc_mpnn_model(chemprop_module=_fake_chemprop("2.3.1"))


def test_architecture_rejects_dimension_drift() -> None:
    with pytest.raises(ValueError, match="must sum"):
        GMCMPNNArchitecture(extra_atom_feature_dim=5)


def test_actual_chemprop_210_contract_if_available() -> None:
    chemprop = pytest.importorskip("chemprop")
    try:
        observed = importlib.metadata.version("chemprop")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("Chemprop package metadata unavailable")
    if observed != "2.1.0":
        pytest.skip(f"Requires Chemprop 2.1.0 compatibility environment, observed {observed}")

    bundle = build_gmc_mpnn_model(chemprop_module=chemprop)
    assert bundle.featurizer.atom_fdim == 78
    assert bundle.featurizer.bond_fdim == 14
    assert bundle.model.message_passing.depth == 5
    assert bundle.model.message_passing.W_i.in_features == 92
    assert bundle.model.message_passing.W_i.out_features == 300
    assert bundle.model.agg.norm == 57.0
    assert bundle.model.predictor.input_dim == 300
    assert bundle.model.predictor.n_tasks == 1
