"""Read-only frozen-feature adapter for Chemprop 2.1.0 GMC-MPNN training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Final, Literal, Mapping

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase

from admet_platform.gmc_mpnn.geometry import GEOMETRY_PREPROCESSING_VERSION
from admet_platform.gmc_mpnn.ggl import GGL_FEATURE_NAMES, GGL_PREPROCESSING_VERSION
from admet_platform.gmc_mpnn.model import (
    BOND_FEATURE_DIM,
    EXPECTED_CHEMPROP_VERSION,
    GGL_ATOM_FEATURE_DIM,
    TOTAL_ATOM_FEATURE_DIM,
    GMCMPNNArchitecture,
    GMCMPNNModelBundle,
)
from admet_platform.gmc_mpnn.scaling import (
    FIT_SUMMARY_FILENAME,
    GGL_SCALER_VERSION,
    TRAINING_PREPROCESSING_VERSION,
    FrozenGGLScaler,
    _npz_content_sha256,
    _parse_integer_like,
    load_frozen_ggl_scaler,
    transform_frozen_ggl,
)
from admet_platform.gmc_mpnn.standardization import GMC_STANDARDIZATION_VERSION


VALIDATION_PREPROCESSING_VERSION: Final = "gmc-mpnn-validation-raw-scaled-ggl-v1"
SUMMARY_FILENAME: Final = "preprocessing_summary.json"
MANIFEST_FILENAME: Final = "feature_manifest.csv"
STATUS_FILENAME: Final = "molecule_status.csv"
MODEL_DATA_CONTRACT_VERSION: Final = "gmc-mpnn-frozen-model-data-v1"

TRAIN_RAW_NPZ_KEYS: Final = frozenset(
    {
        "raw_ggl_features",
        "heavy_atom_atomic_numbers",
        "heavy_atom_rdkit_indices",
        "ggl_feature_names",
        "record_key",
        "training_preprocessing_version",
        "standardization_version",
        "geometry_preprocessing_version",
        "ggl_preprocessing_version",
        "geometry_smiles",
        "geometry_fingerprint",
        "ggl_fingerprint",
        "optimization_method",
        "rdkit_version",
        "artifact_content_sha256",
    }
)
VALIDATION_SCALED_NPZ_KEYS: Final = frozenset(
    {
        "scaled_ggl_features",
        "heavy_atom_atomic_numbers",
        "heavy_atom_rdkit_indices",
        "ggl_feature_names",
        "record_key",
        "validation_preprocessing_version",
        "standardization_version",
        "geometry_preprocessing_version",
        "ggl_preprocessing_version",
        "scaler_version",
        "portable_scaler_sha256",
        "raw_artifact_content_sha256",
        "geometry_smiles",
        "geometry_fingerprint",
        "ggl_fingerprint",
        "optimization_method",
        "rdkit_version",
        "artifact_content_sha256",
    }
)
VALIDATION_RAW_NPZ_KEYS: Final = frozenset(
    (TRAIN_RAW_NPZ_KEYS - {"training_preprocessing_version"}) | {"validation_preprocessing_version"}
)
COMMON_TABLE_COLUMNS: Final = frozenset(
    {
        "record_key",
        "molecule_id",
        "canonical_smiles",
        "target",
        "split",
        "geometry_smiles",
        "status",
        "failure_category",
        "raw_ggl_path",
        "heavy_atom_count",
        "raw_ggl_rows",
        "raw_ggl_columns",
        "standardization_version",
        "geometry_fingerprint",
        "ggl_fingerprint",
        "optimization_method",
        "rdkit_version",
    }
)
STATUS_TABLE_COLUMNS: Final = frozenset(
    {
        "record_key",
        "molecule_id",
        "canonical_smiles",
        "target",
        "split",
        "status",
        "failure_category",
        "raw_ggl_path",
    }
)


class GMCModelDataError(RuntimeError):
    """A frozen model-data contract violation."""


@dataclass(frozen=True)
class FrozenSplitContract:
    """Expected immutable split counts and explicit non-success identities."""

    split: Literal["train", "validation"]
    source_rows: int
    successful_molecules: int
    heavy_atoms: int
    exclusions: tuple[str, ...] = ()
    failures: tuple[tuple[str, str], ...] = ()


TRAIN_SPLIT_CONTRACT: Final = FrozenSplitContract(
    split="train",
    source_rows=1561,
    successful_molecules=1558,
    heavy_atoms=38245,
    exclusions=("eqvalan", "sultamicillin"),
    failures=(("spiclamine", "no_conformer"),),
)
VALIDATION_SPLIT_CONTRACT: Final = FrozenSplitContract(
    split="validation",
    source_rows=196,
    successful_molecules=196,
    heavy_atoms=3755,
)


@dataclass(frozen=True)
class FrozenMoleculeFeatures:
    """One identity-bound, aligned, scaled six-column atom-feature matrix."""

    source_row_index: int
    record_key: str
    molecule_id: str
    canonical_smiles: str
    geometry_smiles: str
    molecule: Chem.Mol
    heavy_atom_atomic_numbers: np.ndarray
    heavy_atom_rdkit_indices: np.ndarray
    V_f: np.ndarray
    source_artifact_content_sha256: str


@dataclass(frozen=True)
class FrozenSupervisedSplit:
    """Frozen molecular features and a separate aligned binary-label array."""

    split: Literal["train", "validation"]
    features: tuple[FrozenMoleculeFeatures, ...]
    labels: np.ndarray
    source_row_count: int
    feature_manifest_sha256: str
    molecule_status_sha256: str
    portable_scaler_sha256: str

    def __post_init__(self) -> None:
        if self.labels.dtype != np.float64 or self.labels.shape != (len(self.features), 1):
            raise GMCModelDataError(
                "Frozen labels must be a separate float64 [n_molecules,1] array."
            )
        if not np.isfinite(self.labels).all() or not np.isin(self.labels, (0.0, 1.0)).all():
            raise GMCModelDataError("Frozen BBB labels must be finite binary values.")


@dataclass(frozen=True)
class FrozenDevelopmentFeatures:
    """Validated TRAIN/validation features tied to one frozen TRAIN scaler."""

    train: FrozenSupervisedSplit
    validation: FrozenSupervisedSplit
    scaler: FrozenGGLScaler


@dataclass(frozen=True)
class FrozenValidationFeatures:
    """Validated validation-only features tied to the frozen TRAIN scaler."""

    validation: FrozenSupervisedSplit
    scaler: FrozenGGLScaler


@dataclass(frozen=True)
class ChempropLoaders:
    """TRAIN-shuffled and validation-unshuffled Chemprop data loaders."""

    train_loader: Any
    validation_loader: Any


def load_frozen_development_features(
    train_preprocessing_dir: str | Path,
    validation_preprocessing_dir: str | Path,
    scaler_dir: str | Path,
    *,
    train_contract: FrozenSplitContract = TRAIN_SPLIT_CONTRACT,
    validation_contract: FrozenSplitContract = VALIDATION_SPLIT_CONTRACT,
) -> FrozenDevelopmentFeatures:
    """Load exactly the frozen successful TRAIN and validation rows; never inspect test data."""

    train_path = _safe_development_directory(train_preprocessing_dir, "training")
    validation_path = _safe_development_directory(validation_preprocessing_dir, "validation")
    scaler_path = _safe_development_directory(scaler_dir, "scaler")
    scaler = load_frozen_ggl_scaler(scaler_path)
    scaler_summary = _read_json(scaler_path / FIT_SUMMARY_FILENAME, "scaler fit summary")
    train = load_frozen_feature_split(
        train_path,
        split="train",
        scaler=scaler,
        scaler_summary=scaler_summary,
        contract=train_contract,
    )
    validation = load_frozen_feature_split(
        validation_path,
        split="validation",
        scaler=scaler,
        scaler_summary=scaler_summary,
        contract=validation_contract,
    )
    return FrozenDevelopmentFeatures(train=train, validation=validation, scaler=scaler)


def load_frozen_validation_features(
    validation_preprocessing_dir: str | Path,
    scaler_dir: str | Path,
    *,
    validation_contract: FrozenSplitContract = VALIDATION_SPLIT_CONTRACT,
) -> FrozenValidationFeatures:
    """Load validation and its frozen TRAIN scaler without opening TRAIN or test artifacts."""

    validation_path = _safe_development_directory(validation_preprocessing_dir, "validation")
    scaler_path = _safe_development_directory(scaler_dir, "scaler")
    scaler = load_frozen_ggl_scaler(scaler_path)
    scaler_summary = _read_json(scaler_path / FIT_SUMMARY_FILENAME, "scaler fit summary")
    validation = load_frozen_feature_split(
        validation_path,
        split="validation",
        scaler=scaler,
        scaler_summary=scaler_summary,
        contract=validation_contract,
    )
    return FrozenValidationFeatures(validation=validation, scaler=scaler)


def load_frozen_feature_split(
    preprocessing_dir: str | Path,
    *,
    split: Literal["train", "validation"] | str,
    scaler: FrozenGGLScaler,
    scaler_summary: Mapping[str, Any],
    contract: FrozenSplitContract | None = None,
) -> FrozenSupervisedSplit:
    """Join manifest identities to frozen atom artifacts without fitting or recomputation."""

    if split == "test":
        raise GMCModelDataError(
            "Locked BBB test access is prohibited by the GMC model-data adapter."
        )
    if split not in {"train", "validation"}:
        raise GMCModelDataError("GMC model data split must be 'train' or 'validation'.")
    resolved_split: Literal["train", "validation"] = split  # type: ignore[assignment]
    expected_contract = contract or (
        TRAIN_SPLIT_CONTRACT if resolved_split == "train" else VALIDATION_SPLIT_CONTRACT
    )
    if expected_contract.split != resolved_split:
        raise GMCModelDataError("Frozen split contract does not match the requested split.")

    source = _safe_development_directory(preprocessing_dir, resolved_split)
    summary = _read_json(source / SUMMARY_FILENAME, f"{resolved_split} preprocessing summary")
    manifest_path = source / MANIFEST_FILENAME
    status_path = source / STATUS_FILENAME
    manifest_sha256 = _sha256_file(manifest_path)
    status_sha256 = _sha256_file(status_path)
    _validate_summary(
        summary,
        split=resolved_split,
        contract=expected_contract,
        manifest_sha256=manifest_sha256,
        status_sha256=status_sha256,
        scaler=scaler,
        scaler_summary=scaler_summary,
    )
    manifest = _read_csv(manifest_path, "feature manifest")
    status = _read_csv(status_path, "molecule status")
    _validate_tables(manifest, status, expected_contract)

    status_labels = {
        str(row.record_key): _binary_label(row.target, str(row.record_key))
        for row in status.itertuples(index=False)
    }
    features_by_key: dict[str, FrozenMoleculeFeatures] = {}
    ordered_train_digest = hashlib.sha256()
    for source_row_index, row in enumerate(manifest.itertuples(index=False)):
        if str(row.status) != "success":
            continue
        record_key = str(row.record_key)
        if resolved_split == "train":
            feature = _load_train_record(source, row, source_row_index, scaler)
            raw_path = _safe_artifact_path(source, str(row.raw_ggl_path), "TRAIN raw GGL")
            ordered_entry = {
                "artifact_content_sha256": feature.source_artifact_content_sha256,
                "artifact_file_sha256": _sha256_file(raw_path),
                "raw_ggl_path": str(row.raw_ggl_path).replace("\\", "/"),
                "record_key": record_key,
                "source_row_index": source_row_index,
            }
            ordered_train_digest.update(
                json.dumps(
                    ordered_entry,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            )
            ordered_train_digest.update(b"\n")
        else:
            feature = _load_validation_record(source, row, source_row_index, scaler)
        if record_key in features_by_key:
            raise GMCModelDataError(f"Duplicate successful record key: {record_key}")
        features_by_key[record_key] = feature

    ordered_keys = [
        str(row.record_key) for row in manifest.itertuples(index=False) if row.status == "success"
    ]
    if set(ordered_keys) != set(features_by_key) or not set(ordered_keys).issubset(status_labels):
        raise GMCModelDataError("Stable record-key join between features and labels is incomplete.")
    features = tuple(features_by_key[key] for key in ordered_keys)
    labels = np.asarray([[status_labels[key]] for key in ordered_keys], dtype=np.float64)
    observed_atoms = sum(feature.V_f.shape[0] for feature in features)
    if len(features) != expected_contract.successful_molecules:
        raise GMCModelDataError("Successful frozen molecule count differs from the contract.")
    if observed_atoms != expected_contract.heavy_atoms:
        raise GMCModelDataError("Successful frozen heavy-atom count differs from the contract.")
    if resolved_split == "train":
        _validate_training_scaler_input_provenance(
            scaler_summary,
            expected_contract,
            manifest_sha256,
            status_sha256,
            ordered_train_digest.hexdigest(),
        )
    return FrozenSupervisedSplit(
        split=resolved_split,
        features=features,
        labels=labels,
        source_row_count=expected_contract.source_rows,
        feature_manifest_sha256=manifest_sha256,
        molecule_status_sha256=status_sha256,
        portable_scaler_sha256=scaler.portable_scaler_sha256,
    )


def build_chemprop_dataset(
    split: FrozenSupervisedSplit,
    model_bundle: GMCMPNNModelBundle,
    *,
    chemprop_module: Any | None = None,
) -> Any:
    """Convert validated float64 features to Chemprop float32 datapoints and a dataset."""

    chemprop_module = _require_chemprop_module(chemprop_module)
    if model_bundle.featurizer.atom_fdim != TOTAL_ATOM_FEATURE_DIM:
        raise GMCModelDataError("Model featurizer atom dimension is not 78.")
    if model_bundle.featurizer.bond_fdim != BOND_FEATURE_DIM:
        raise GMCModelDataError("Model featurizer bond dimension is not 14.")
    datapoints = []
    for feature, label in zip(split.features, split.labels, strict=True):
        V_f = np.asarray(feature.V_f, dtype=np.float32)
        y = np.asarray(label, dtype=np.float32)
        if V_f.dtype != np.float32 or V_f.shape != (
            feature.molecule.GetNumAtoms(),
            GGL_ATOM_FEATURE_DIM,
        ):
            raise GMCModelDataError("Model-boundary V_f must be float32 [n_atoms,6].")
        datapoints.append(
            chemprop_module.data.MoleculeDatapoint(
                mol=feature.molecule,
                y=y,
                V_f=V_f,
                name=feature.record_key,
            )
        )
    dataset = chemprop_module.data.MoleculeDataset(
        datapoints,
        featurizer=model_bundle.featurizer,
    )
    if len(dataset) != len(split.features):
        raise GMCModelDataError("Chemprop dataset silently changed the frozen molecule count.")
    return dataset


def build_chemprop_dataloaders(
    train_dataset: Any,
    validation_dataset: Any,
    *,
    chemprop_module: Any | None = None,
    architecture: GMCMPNNArchitecture = GMCMPNNArchitecture(),
    seed: int | None = None,
    num_workers: int = 0,
) -> ChempropLoaders:
    """Build the released TRAIN-shuffled/validation-unshuffled loader contract."""

    chemprop_module = _require_chemprop_module(chemprop_module)
    train_loader = chemprop_module.data.build_dataloader(
        train_dataset,
        batch_size=architecture.batch_size,
        num_workers=num_workers,
        seed=seed,
        shuffle=True,
    )
    validation_loader = build_chemprop_validation_dataloader(
        validation_dataset,
        chemprop_module=chemprop_module,
        architecture=architecture,
        num_workers=num_workers,
    )
    return ChempropLoaders(
        train_loader=train_loader,
        validation_loader=validation_loader,
    )


def build_chemprop_validation_dataloader(
    validation_dataset: Any,
    *,
    chemprop_module: Any | None = None,
    architecture: GMCMPNNArchitecture = GMCMPNNArchitecture(),
    num_workers: int = 0,
) -> Any:
    """Build an unshuffled validation-only Chemprop loader."""

    chemprop_module = _require_chemprop_module(chemprop_module)
    return chemprop_module.data.build_dataloader(
        validation_dataset,
        batch_size=architecture.batch_size,
        num_workers=num_workers,
        shuffle=False,
    )


def assert_float32_model_boundary(batch: Any) -> None:
    """Verify collated Chemprop graph tensors have the frozen dimensions and float32 dtype."""

    try:
        graph = batch[0]
        atom_features = graph.V
        bond_features = graph.E
    except (TypeError, AttributeError, IndexError) as exc:
        raise GMCModelDataError("Unexpected Chemprop training-batch structure.") from exc

    import torch

    if atom_features.dtype != torch.float32:
        raise GMCModelDataError("Model-boundary atom features must be float32.")
    if bond_features.dtype != torch.float32:
        raise GMCModelDataError("Model-boundary bond features must be float32.")
    if atom_features.ndim != 2 or atom_features.shape[1] != TOTAL_ATOM_FEATURE_DIM:
        raise GMCModelDataError("Model-boundary atom feature dimension is not 78.")
    if bond_features.ndim != 2 or bond_features.shape[1] != BOND_FEATURE_DIM:
        raise GMCModelDataError("Model-boundary bond feature dimension is not 14.")


def _require_chemprop_module(chemprop_module: Any | None) -> Any:
    if chemprop_module is None:
        try:
            import chemprop as chemprop_module
        except ImportError as exc:  # pragma: no cover - environment boundary
            raise RuntimeError(
                f"Chemprop {EXPECTED_CHEMPROP_VERSION} is required for GMC model data."
            ) from exc
    if getattr(chemprop_module, "__version__", None) != EXPECTED_CHEMPROP_VERSION:
        raise RuntimeError(
            f"GMC-MPNN model data requires exactly Chemprop {EXPECTED_CHEMPROP_VERSION}."
        )
    return chemprop_module


def _validate_training_scaler_input_provenance(
    scaler_summary: Mapping[str, Any],
    contract: FrozenSplitContract,
    manifest_sha256: str,
    status_sha256: str,
    ordered_input_artifact_sha256: str,
) -> None:
    expected = {
        "feature_order": list(GGL_FEATURE_NAMES),
        "training_molecule_count": contract.successful_molecules,
        "training_atom_count": contract.heavy_atoms,
        "source_row_count": contract.source_rows,
        "feature_manifest_sha256": manifest_sha256,
        "molecule_status_sha256": status_sha256,
        "ordered_input_artifact_sha256": ordered_input_artifact_sha256,
    }
    mismatches = {
        key: {"expected": value, "observed": scaler_summary.get(key)}
        for key, value in expected.items()
        if scaler_summary.get(key) != value
    }
    if mismatches:
        raise GMCModelDataError(
            f"Frozen TRAIN scaler input provenance is incompatible: {mismatches}"
        )


def _safe_development_directory(path: str | Path, label: str) -> Path:
    resolved = Path(path)
    prohibited = {"test", "locked", "locked-test", "locked_test", "bbb_test"}
    if prohibited.intersection(part.lower() for part in resolved.parts):
        raise GMCModelDataError(f"{label} path contains a prohibited test/locked component.")
    return resolved


def _validate_summary(
    summary: Mapping[str, Any],
    *,
    split: Literal["train", "validation"],
    contract: FrozenSplitContract,
    manifest_sha256: str,
    status_sha256: str,
    scaler: FrozenGGLScaler,
    scaler_summary: Mapping[str, Any],
) -> None:
    expected = {
        "loaded_split": split,
        "standardization_version": GMC_STANDARDIZATION_VERSION,
        "geometry_preprocessing_version": GEOMETRY_PREPROCESSING_VERSION,
        "ggl_preprocessing_version": GGL_PREPROCESSING_VERSION,
        "ggl_feature_order": list(GGL_FEATURE_NAMES),
        "rdkit_version": rdBase.rdkitVersion,
        "test_artifact_accessed": False,
    }
    if split == "train":
        expected.update(
            {
                "training_preprocessing_version": TRAINING_PREPROCESSING_VERSION,
                "validation_artifact_accessed": False,
                "ggl_scaled": False,
            }
        )
    else:
        expected.update(
            {
                "validation_preprocessing_version": VALIDATION_PREPROCESSING_VERSION,
                "validation_artifact_accessed": True,
                "ggl_scaler_version": GGL_SCALER_VERSION,
                "portable_scaler_sha256": scaler.portable_scaler_sha256,
            }
        )
    mismatches = {
        key: {"expected": value, "observed": summary.get(key)}
        for key, value in expected.items()
        if summary.get(key) != value
    }
    if mismatches:
        raise GMCModelDataError(f"Frozen {split} preprocessing is incompatible: {mismatches}")
    counts = {
        "source_row_count": contract.source_rows,
        "successful_molecule_count": contract.successful_molecules,
        "policy_excluded_molecule_count": len(contract.exclusions),
        "failed_molecule_count": len(contract.failures),
        "total_heavy_atom_count_among_successes": contract.heavy_atoms,
    }
    for key, expected_count in counts.items():
        observed = _parse_integer_like(summary.get(key), field=f"summary {key}", minimum=0)
        if observed != expected_count:
            raise GMCModelDataError(f"Frozen {split} {key} differs from the contract.")
    if summary.get("feature_manifest_sha256") != manifest_sha256:
        raise GMCModelDataError(f"Frozen {split} feature-manifest hash is invalid.")
    if summary.get("molecule_status_sha256") != status_sha256:
        raise GMCModelDataError(f"Frozen {split} molecule-status hash is invalid.")
    _validate_scaler_provenance(scaler_summary, scaler, summary, split)


def _validate_scaler_provenance(
    scaler_summary: Mapping[str, Any],
    scaler: FrozenGGLScaler,
    preprocessing_summary: Mapping[str, Any],
    split: str,
) -> None:
    expected = {
        "scaler_version": GGL_SCALER_VERSION,
        "portable_scaler_sha256": scaler.portable_scaler_sha256,
        "training_preprocessing_version": TRAINING_PREPROCESSING_VERSION,
        "standardization_version": GMC_STANDARDIZATION_VERSION,
        "geometry_preprocessing_version": GEOMETRY_PREPROCESSING_VERSION,
        "ggl_preprocessing_version": GGL_PREPROCESSING_VERSION,
        "rdkit_version": rdBase.rdkitVersion,
        "loaded_split": "train",
        "validation_artifact_accessed": False,
        "test_artifact_accessed": False,
    }
    if any(scaler_summary.get(key) != value for key, value in expected.items()):
        raise GMCModelDataError("Frozen TRAIN scaler provenance is incompatible.")
    if split == "train":
        pairs = (
            ("feature_manifest_sha256", "feature_manifest_sha256"),
            ("molecule_status_sha256", "molecule_status_sha256"),
        )
    else:
        pairs = (
            ("feature_manifest_sha256", "training_feature_manifest_sha256"),
            ("molecule_status_sha256", "training_molecule_status_sha256"),
            ("ordered_input_artifact_sha256", "training_ordered_input_artifact_sha256"),
        )
    for scaler_key, preprocessing_key in pairs:
        if scaler_summary.get(scaler_key) != preprocessing_summary.get(preprocessing_key):
            raise GMCModelDataError(
                f"Frozen scaler and {split} preprocessing disagree for {scaler_key}."
            )


def _validate_tables(
    manifest: pd.DataFrame,
    status: pd.DataFrame,
    contract: FrozenSplitContract,
) -> None:
    for label, frame, required in (
        ("manifest", manifest, COMMON_TABLE_COLUMNS),
        ("status", status, STATUS_TABLE_COLUMNS),
    ):
        missing = required.difference(frame.columns)
        if missing:
            raise GMCModelDataError(f"Frozen {label} is missing columns: {sorted(missing)}")
        if len(frame) != contract.source_rows:
            raise GMCModelDataError(f"Frozen {label} does not contain every source row.")
        if frame["record_key"].duplicated().any():
            raise GMCModelDataError(f"Frozen {label} record keys are not unique.")
        if set(frame["split"].astype(str)) != {contract.split}:
            raise GMCModelDataError(f"Frozen {label} contains a different split.")
    for column in ("record_key", "molecule_id", "canonical_smiles", "target", "split", "status"):
        if manifest[column].astype(str).tolist() != status[column].astype(str).tolist():
            raise GMCModelDataError(f"Manifest/status stable identity mismatch in {column}.")
    statuses = manifest["status"].astype(str)
    if not set(statuses).issubset({"success", "excluded", "failed"}):
        raise GMCModelDataError("Frozen manifest contains an invalid status.")
    exclusions = tuple(
        sorted(manifest.loc[statuses == "excluded", "molecule_id"].astype(str).tolist())
    )
    if exclusions != tuple(sorted(contract.exclusions)):
        raise GMCModelDataError("Frozen policy exclusions differ from the contract.")
    failures = tuple(
        sorted(
            zip(
                manifest.loc[statuses == "failed", "molecule_id"].astype(str),
                manifest.loc[statuses == "failed", "failure_category"].astype(str),
                strict=True,
            )
        )
    )
    if failures != tuple(sorted(contract.failures)):
        raise GMCModelDataError("Frozen preprocessing failures differ from the contract.")
    if contract.split == "validation":
        if "source_row_index" not in manifest.columns:
            raise GMCModelDataError("Validation manifest has no source-row indices.")
        indices = [
            _parse_integer_like(value, field="validation source_row_index", minimum=0)
            for value in manifest["source_row_index"]
        ]
        if indices != list(range(contract.source_rows)):
            raise GMCModelDataError("Validation source-row indices are incomplete or reordered.")


def _load_train_record(
    root: Path,
    row: Any,
    source_row_index: int,
    scaler: FrozenGGLScaler,
) -> FrozenMoleculeFeatures:
    path = _safe_artifact_path(root, str(row.raw_ggl_path), "TRAIN raw GGL")
    artifact = _load_npz(path, TRAIN_RAW_NPZ_KEYS, "TRAIN raw GGL")
    expected = {
        "record_key": str(row.record_key),
        "training_preprocessing_version": TRAINING_PREPROCESSING_VERSION,
        "standardization_version": GMC_STANDARDIZATION_VERSION,
        "geometry_preprocessing_version": GEOMETRY_PREPROCESSING_VERSION,
        "ggl_preprocessing_version": GGL_PREPROCESSING_VERSION,
        "geometry_smiles": str(row.geometry_smiles),
        "geometry_fingerprint": str(row.geometry_fingerprint),
        "ggl_fingerprint": str(row.ggl_fingerprint),
        "optimization_method": str(row.optimization_method),
        "rdkit_version": rdBase.rdkitVersion,
    }
    _validate_artifact_scalars(artifact, expected, path.name)
    raw = artifact["raw_ggl_features"]
    _validate_float64_matrix(raw, path.name)
    scaled = transform_frozen_ggl(raw, scaler)
    checksum = _validate_npz_checksum(artifact, path.name)
    return _aligned_feature_record(
        row,
        source_row_index,
        artifact,
        scaled,
        checksum,
    )


def _load_validation_record(
    root: Path,
    row: Any,
    source_row_index: int,
    scaler: FrozenGGLScaler,
) -> FrozenMoleculeFeatures:
    raw_path = _safe_artifact_path(root, str(row.raw_ggl_path), "validation raw GGL")
    scaled_path = _safe_artifact_path(root, str(row.scaled_ggl_path), "validation scaled GGL")
    raw_artifact = _load_npz(raw_path, VALIDATION_RAW_NPZ_KEYS, "validation raw GGL")
    scaled_artifact = _load_npz(scaled_path, VALIDATION_SCALED_NPZ_KEYS, "validation scaled GGL")
    expected = {
        "record_key": str(row.record_key),
        "validation_preprocessing_version": VALIDATION_PREPROCESSING_VERSION,
        "standardization_version": GMC_STANDARDIZATION_VERSION,
        "geometry_preprocessing_version": GEOMETRY_PREPROCESSING_VERSION,
        "ggl_preprocessing_version": GGL_PREPROCESSING_VERSION,
        "geometry_smiles": str(row.geometry_smiles),
        "geometry_fingerprint": str(row.geometry_fingerprint),
        "ggl_fingerprint": str(row.ggl_fingerprint),
        "optimization_method": str(row.optimization_method),
        "rdkit_version": rdBase.rdkitVersion,
    }
    _validate_artifact_scalars(raw_artifact, expected, raw_path.name)
    _validate_artifact_scalars(scaled_artifact, expected, scaled_path.name)
    for key in (
        "heavy_atom_atomic_numbers",
        "heavy_atom_rdkit_indices",
        "ggl_feature_names",
        "geometry_fingerprint",
        "ggl_fingerprint",
        "optimization_method",
        "rdkit_version",
    ):
        if not np.array_equal(raw_artifact[key], scaled_artifact[key]):
            raise GMCModelDataError(f"Validation raw/scaled atom provenance differs for {key}.")
    raw = raw_artifact["raw_ggl_features"]
    scaled = scaled_artifact["scaled_ggl_features"]
    _validate_float64_matrix(raw, raw_path.name)
    _validate_float64_matrix(scaled, scaled_path.name)
    raw_checksum = _validate_npz_checksum(raw_artifact, raw_path.name)
    scaled_checksum = _validate_npz_checksum(scaled_artifact, scaled_path.name)
    if _scalar_string(scaled_artifact["raw_artifact_content_sha256"]) != raw_checksum:
        raise GMCModelDataError("Validation scaled artifact does not link to its raw artifact.")
    if _scalar_string(scaled_artifact["scaler_version"]) != GGL_SCALER_VERSION:
        raise GMCModelDataError("Validation scaled artifact has the wrong scaler version.")
    if _scalar_string(scaled_artifact["portable_scaler_sha256"]) != scaler.portable_scaler_sha256:
        raise GMCModelDataError("Validation scaled artifact has the wrong frozen scaler hash.")
    if not np.array_equal(scaled, transform_frozen_ggl(raw, scaler)):
        raise GMCModelDataError("Validation scaled values differ from the frozen scaler transform.")
    if str(row.raw_artifact_content_sha256) != raw_checksum:
        raise GMCModelDataError("Validation manifest has the wrong raw artifact checksum.")
    if str(row.scaled_artifact_content_sha256) != scaled_checksum:
        raise GMCModelDataError("Validation manifest has the wrong scaled artifact checksum.")
    return _aligned_feature_record(
        row,
        source_row_index,
        scaled_artifact,
        scaled,
        scaled_checksum,
    )


def _aligned_feature_record(
    row: Any,
    source_row_index: int,
    artifact: Mapping[str, np.ndarray],
    scaled: np.ndarray,
    checksum: str,
) -> FrozenMoleculeFeatures:
    atomic_numbers = artifact["heavy_atom_atomic_numbers"]
    rdkit_indices = artifact["heavy_atom_rdkit_indices"]
    if atomic_numbers.dtype != np.int64 or rdkit_indices.dtype != np.int64:
        raise GMCModelDataError("Frozen atom identity arrays must be int64.")
    if atomic_numbers.shape != (scaled.shape[0],) or rdkit_indices.shape != (scaled.shape[0],):
        raise GMCModelDataError("Frozen atom identity arrays do not align with V_f rows.")
    if not np.array_equal(rdkit_indices, np.arange(scaled.shape[0], dtype=np.int64)):
        raise GMCModelDataError("Frozen RDKit atom indices are not ascending and complete.")
    geometry_smiles = str(row.geometry_smiles)
    molecule = Chem.MolFromSmiles(geometry_smiles)
    if molecule is None:
        raise GMCModelDataError("Frozen geometry SMILES cannot be parsed by RDKit.")
    molecule_atomic_numbers = np.asarray(
        [atom.GetAtomicNum() for atom in molecule.GetAtoms()], dtype=np.int64
    )
    if np.any(molecule_atomic_numbers <= 1):
        raise GMCModelDataError("Chemprop boundary molecule contains explicit hydrogen atoms.")
    if not np.array_equal(molecule_atomic_numbers, atomic_numbers):
        raise GMCModelDataError("RDKit molecule atom identity/order differs from frozen V_f rows.")
    expected_count = _parse_integer_like(
        row.heavy_atom_count, field=f"manifest heavy_atom_count for {row.record_key}", minimum=1
    )
    expected_rows = _parse_integer_like(
        row.raw_ggl_rows, field=f"manifest raw_ggl_rows for {row.record_key}", minimum=1
    )
    expected_columns = _parse_integer_like(
        row.raw_ggl_columns, field=f"manifest raw_ggl_columns for {row.record_key}", minimum=1
    )
    if scaled.shape != (expected_count, GGL_ATOM_FEATURE_DIM):
        raise GMCModelDataError("Frozen V_f shape differs from manifest heavy-atom count.")
    if expected_rows != expected_count or expected_columns != GGL_ATOM_FEATURE_DIM:
        raise GMCModelDataError("Frozen manifest GGL dimensions are invalid.")
    return FrozenMoleculeFeatures(
        source_row_index=source_row_index,
        record_key=str(row.record_key),
        molecule_id=str(row.molecule_id),
        canonical_smiles=str(row.canonical_smiles),
        geometry_smiles=geometry_smiles,
        molecule=molecule,
        heavy_atom_atomic_numbers=np.array(atomic_numbers, dtype=np.int64, copy=True),
        heavy_atom_rdkit_indices=np.array(rdkit_indices, dtype=np.int64, copy=True),
        V_f=np.array(scaled, dtype=np.float64, copy=True),
        source_artifact_content_sha256=checksum,
    )


def _validate_artifact_scalars(
    artifact: Mapping[str, np.ndarray], expected: Mapping[str, str], name: str
) -> None:
    for key, value in expected.items():
        if _scalar_string(artifact[key]) != value:
            raise GMCModelDataError(f"Frozen artifact {name} has invalid {key}.")
    feature_order = tuple(str(value) for value in artifact["ggl_feature_names"].tolist())
    if feature_order != tuple(GGL_FEATURE_NAMES):
        raise GMCModelDataError(f"Frozen artifact {name} has the wrong GGL feature order.")


def _validate_float64_matrix(values: np.ndarray, name: str) -> None:
    if values.dtype != np.float64:
        raise GMCModelDataError(f"Frozen feature artifact {name} is not float64.")
    if values.ndim != 2 or values.shape[1] != GGL_ATOM_FEATURE_DIM or values.shape[0] == 0:
        raise GMCModelDataError(f"Frozen feature artifact {name} must have shape [n_atoms,6].")
    if not np.isfinite(values).all():
        raise GMCModelDataError(f"Frozen feature artifact {name} contains NaN or Inf.")


def _validate_npz_checksum(artifact: Mapping[str, np.ndarray], name: str) -> str:
    stored = _scalar_string(artifact["artifact_content_sha256"])
    content = {key: artifact[key] for key in artifact if key != "artifact_content_sha256"}
    if stored != _npz_content_sha256(content):
        raise GMCModelDataError(f"Frozen feature artifact checksum failed: {name}")
    return stored


def _load_npz(path: Path, expected_keys: frozenset[str], label: str) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise GMCModelDataError(f"Missing {label} artifact: {path.name}")
    try:
        with np.load(path, allow_pickle=False) as artifact:
            if set(artifact.files) != expected_keys:
                raise GMCModelDataError(f"{label} artifact has an unexpected schema: {path.name}")
            return {key: artifact[key] for key in artifact.files}
    except GMCModelDataError:
        raise
    except (OSError, ValueError, EOFError) as exc:
        raise GMCModelDataError(f"Unreadable {label} artifact: {path.name}") from exc


def _safe_artifact_path(root: Path, relative: str, label: str) -> Path:
    if not relative.strip():
        raise GMCModelDataError(f"Successful row has no {label} path.")
    portable = PurePosixPath(relative.replace("\\", "/"))
    if portable.is_absolute() or PureWindowsPath(relative).is_absolute() or ".." in portable.parts:
        raise GMCModelDataError(f"{label} path must be relative to its preprocessing directory.")
    root_resolved = root.resolve()
    candidate = root.joinpath(*portable.parts).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise GMCModelDataError(f"{label} path escapes its preprocessing directory.") from exc
    return candidate


def _binary_label(value: object, record_key: str) -> float:
    parsed = _parse_integer_like(value, field=f"label for record {record_key}", minimum=0)
    if parsed not in {0, 1}:
        raise GMCModelDataError(f"BBB label for record {record_key} is not binary.")
    return float(parsed)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GMCModelDataError(f"Frozen {label} is unreadable.") from exc
    if not isinstance(value, dict):
        raise GMCModelDataError(f"Frozen {label} must be a JSON object.")
    return value


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, keep_default_na=False)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise GMCModelDataError(f"Frozen {label} is unreadable.") from exc


def _scalar_string(value: np.ndarray) -> str:
    if value.shape != ():
        raise GMCModelDataError("Frozen artifact provenance values must be scalar.")
    return str(value.item())


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise GMCModelDataError(f"Frozen artifact is unreadable: {path.name}") from exc


__all__ = [
    "ChempropLoaders",
    "MODEL_DATA_CONTRACT_VERSION",
    "GMCModelDataError",
    "FrozenDevelopmentFeatures",
    "FrozenMoleculeFeatures",
    "FrozenSplitContract",
    "FrozenSupervisedSplit",
    "FrozenValidationFeatures",
    "TRAIN_SPLIT_CONTRACT",
    "VALIDATION_SPLIT_CONTRACT",
    "assert_float32_model_boundary",
    "build_chemprop_dataloaders",
    "build_chemprop_dataset",
    "build_chemprop_validation_dataloader",
    "load_frozen_development_features",
    "load_frozen_feature_split",
    "load_frozen_validation_features",
]
