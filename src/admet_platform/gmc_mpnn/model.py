"""Source-faithful Chemprop 2.1.0 GMC-MPNN BBB model construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final


MODEL_INTERFACE_VERSION: Final = "gmc-mpnn-model-interface-v1"
EXPECTED_CHEMPROP_VERSION: Final = "2.1.0"
ORDINARY_ATOM_FEATURE_DIM: Final = 72
GGL_ATOM_FEATURE_DIM: Final = 6
TOTAL_ATOM_FEATURE_DIM: Final = 78
BOND_FEATURE_DIM: Final = 14


@dataclass(frozen=True)
class GMCMPNNArchitecture:
    """Frozen MoleculeNet-BBBP architecture from the released GMC-MPNN source."""

    ordinary_atom_feature_dim: int = ORDINARY_ATOM_FEATURE_DIM
    extra_atom_feature_dim: int = GGL_ATOM_FEATURE_DIM
    atom_input_dim: int = TOTAL_ATOM_FEATURE_DIM
    bond_input_dim: int = BOND_FEATURE_DIM
    hidden_dim: int = 300
    message_passing_depth: int = 5
    message_activation: str = "RELU"
    message_dropout: float = 0.0
    aggregation: str = "normalized_sum"
    aggregation_norm: float = 57.0
    output_tasks: int = 1
    ffn_layers: int = 2
    ffn_hidden_dim: int = 900
    ffn_activation: str = "LEAKYRELU"
    ffn_dropout: float = 0.0
    batch_norm: bool = False
    loss: str = "binary_cross_entropy_with_logits"
    batch_size: int = 32
    maximum_epochs: int = 100
    checkpoint_monitor: str = "val_loss"
    checkpoint_mode: str = "min"
    checkpoint_save_top_k: int = 1
    early_stopping_patience: int = 10
    optimizer: str = "Adam"
    scheduler: str = "Chemprop Noam-like"
    warmup_epochs: int = 2
    initial_learning_rate: float = 1e-4
    maximum_learning_rate: float = 1e-3
    final_learning_rate: float = 1e-4

    def __post_init__(self) -> None:
        if self.ordinary_atom_feature_dim + self.extra_atom_feature_dim != self.atom_input_dim:
            raise ValueError(
                "Ordinary and GGL atom dimensions must sum to the model atom dimension."
            )
        if self.atom_input_dim != 78 or self.bond_input_dim != 14:
            raise ValueError("The frozen GMC-MPNN graph dimensions must be 78 atoms / 14 bonds.")


@dataclass(frozen=True)
class GMCMPNNModelBundle:
    """Chemprop objects plus their validated frozen architecture."""

    model: Any
    featurizer: Any
    architecture: GMCMPNNArchitecture


def build_gmc_mpnn_model(
    *,
    chemprop_module: Any | None = None,
    architecture: GMCMPNNArchitecture = GMCMPNNArchitecture(),
) -> GMCMPNNModelBundle:
    """Build the released BBB architecture with Chemprop 2.1.0 APIs.

    Chemprop is imported lazily so frozen feature validation remains usable in preprocessing
    environments that intentionally do not contain the model stack.
    """

    if chemprop_module is None:
        try:
            import chemprop as chemprop_module
        except ImportError as exc:  # pragma: no cover - environment boundary
            raise RuntimeError("Chemprop 2.1.0 is required to construct GMC-MPNN.") from exc
    if getattr(chemprop_module, "__version__", None) != EXPECTED_CHEMPROP_VERSION:
        raise RuntimeError(f"GMC-MPNN requires exactly Chemprop {EXPECTED_CHEMPROP_VERSION}.")

    featurizer = chemprop_module.featurizers.SimpleMoleculeMolGraphFeaturizer(
        extra_atom_fdim=architecture.extra_atom_feature_dim
    )
    if featurizer.atom_fdim - architecture.extra_atom_feature_dim != (
        architecture.ordinary_atom_feature_dim
    ):
        raise RuntimeError("Chemprop ordinary atom featurizer is not the required 72 dimensions.")
    if featurizer.atom_fdim != architecture.atom_input_dim:
        raise RuntimeError("Chemprop featurizer did not produce 78 atom input dimensions.")
    if featurizer.bond_fdim != architecture.bond_input_dim:
        raise RuntimeError("Chemprop featurizer did not produce 14 bond input dimensions.")

    message_passing = chemprop_module.nn.BondMessagePassing(
        d_v=architecture.atom_input_dim,
        d_e=architecture.bond_input_dim,
        d_h=architecture.hidden_dim,
        depth=architecture.message_passing_depth,
        dropout=architecture.message_dropout,
        activation=architecture.message_activation,
        bias=False,
        undirected=False,
    )
    aggregation = chemprop_module.nn.NormAggregation(norm=architecture.aggregation_norm)
    predictor = chemprop_module.nn.BinaryClassificationFFN(
        n_tasks=architecture.output_tasks,
        input_dim=architecture.hidden_dim,
        hidden_dim=architecture.ffn_hidden_dim,
        n_layers=architecture.ffn_layers,
        dropout=architecture.ffn_dropout,
        activation=architecture.ffn_activation,
    )
    model = chemprop_module.models.MPNN(
        message_passing,
        aggregation,
        predictor,
        batch_norm=architecture.batch_norm,
        warmup_epochs=architecture.warmup_epochs,
        init_lr=architecture.initial_learning_rate,
        max_lr=architecture.maximum_learning_rate,
        final_lr=architecture.final_learning_rate,
    )
    return GMCMPNNModelBundle(model=model, featurizer=featurizer, architecture=architecture)


__all__ = [
    "BOND_FEATURE_DIM",
    "EXPECTED_CHEMPROP_VERSION",
    "GGL_ATOM_FEATURE_DIM",
    "GMCMPNNArchitecture",
    "GMCMPNNModelBundle",
    "MODEL_INTERFACE_VERSION",
    "ORDINARY_ATOM_FEATURE_DIM",
    "TOTAL_ATOM_FEATURE_DIM",
    "build_gmc_mpnn_model",
]
