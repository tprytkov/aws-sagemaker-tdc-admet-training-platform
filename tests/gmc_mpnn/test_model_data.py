from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem, rdBase

from admet_platform.gmc_mpnn import geometry, ggl, scaling
from admet_platform.gmc_mpnn.geometry import GEOMETRY_PREPROCESSING_VERSION
from admet_platform.gmc_mpnn.ggl import GGL_FEATURE_NAMES, GGL_PREPROCESSING_VERSION
from admet_platform.gmc_mpnn.model import build_gmc_mpnn_model
from admet_platform.gmc_mpnn.model_data import (
    GMCModelDataError,
    TRAIN_SPLIT_CONTRACT,
    VALIDATION_SPLIT_CONTRACT,
    FrozenSplitContract,
    assert_float32_model_boundary,
    build_chemprop_dataloaders,
    build_chemprop_dataset,
    load_frozen_development_features,
    load_frozen_feature_split,
)
from admet_platform.gmc_mpnn.standardization import GMC_STANDARDIZATION_VERSION


TRAIN_CONTRACT = FrozenSplitContract(
    split="train",
    source_rows=4,
    successful_molecules=1,
    heavy_atoms=2,
    exclusions=("eqvalan", "sultamicillin"),
    failures=(("spiclamine", "no_conformer"),),
)
VALIDATION_CONTRACT = FrozenSplitContract(
    split="validation", source_rows=1, successful_molecules=1, heavy_atoms=2
)


class _Datapoint:
    def __init__(self, **kwargs: Any):
        self.__dict__.update(kwargs)


class _Dataset:
    def __init__(self, data: list[_Datapoint], *, featurizer: Any):
        self.data = data
        self.featurizer = featurizer

    def __len__(self) -> int:
        return len(self.data)


class _Featurizer:
    def __init__(self, *, extra_atom_fdim: int):
        self.atom_fdim = 72 + extra_atom_fdim
        self.bond_fdim = 14


class _Captured:
    def __init__(self, *args: Any, **kwargs: Any):
        self.args = args
        self.kwargs = kwargs


def _build_dataloader(dataset: Any, **kwargs: Any) -> _Captured:
    return _Captured(dataset, **kwargs)


def _fake_chemprop() -> SimpleNamespace:
    return SimpleNamespace(
        __version__="2.1.0",
        data=SimpleNamespace(
            MoleculeDatapoint=_Datapoint,
            MoleculeDataset=_Dataset,
            build_dataloader=_build_dataloader,
        ),
        featurizers=SimpleNamespace(SimpleMoleculeMolGraphFeaturizer=_Featurizer),
        nn=SimpleNamespace(
            BondMessagePassing=_Captured,
            NormAggregation=_Captured,
            BinaryClassificationFFN=_Captured,
        ),
        models=SimpleNamespace(MPNN=_Captured),
    )


def test_frozen_production_row_count_contracts() -> None:
    assert TRAIN_SPLIT_CONTRACT.source_rows == 1561
    assert TRAIN_SPLIT_CONTRACT.successful_molecules == 1558
    assert TRAIN_SPLIT_CONTRACT.heavy_atoms == 38245
    assert VALIDATION_SPLIT_CONTRACT.source_rows == 196
    assert VALIDATION_SPLIT_CONTRACT.successful_molecules == 196
    assert VALIDATION_SPLIT_CONTRACT.heavy_atoms == 3755


def test_stable_identity_join_alignment_and_float32_model_boundary(tmp_path: Path) -> None:
    fixture = _frozen_fixture(tmp_path)
    development = _load_fixture(fixture)

    assert [item.record_key for item in development.train.features] == ["train-success"]
    assert [item.record_key for item in development.validation.features] == ["validation-success"]
    np.testing.assert_array_equal(development.train.labels, [[1.0]])
    np.testing.assert_array_equal(development.validation.labels, [[0.0]])
    feature = development.train.features[0]
    assert feature.V_f.dtype == np.float64
    assert feature.V_f.shape == (2, 6)
    np.testing.assert_array_equal(feature.heavy_atom_rdkit_indices, [0, 1])
    np.testing.assert_array_equal(feature.heavy_atom_atomic_numbers, [6, 6])

    chemprop = _fake_chemprop()
    bundle = build_gmc_mpnn_model(chemprop_module=chemprop)
    dataset = build_chemprop_dataset(development.train, bundle, chemprop_module=chemprop)

    assert len(dataset) == 1
    assert dataset.featurizer.atom_fdim == 78
    assert dataset.featurizer.bond_fdim == 14
    assert dataset.data[0].name == "train-success"
    assert dataset.data[0].V_f.dtype == np.float32
    assert dataset.data[0].V_f.shape == (2, 6)
    assert dataset.data[0].y.dtype == np.float32


def test_dataloaders_preserve_released_shuffle_and_batch_contract() -> None:
    chemprop = _fake_chemprop()
    loaders = build_chemprop_dataloaders(
        "train-dataset",
        "validation-dataset",
        chemprop_module=chemprop,
        seed=17,
        num_workers=2,
    )

    assert loaders.train_loader.args == ("train-dataset",)
    assert loaders.train_loader.kwargs == {
        "batch_size": 32,
        "num_workers": 2,
        "seed": 17,
        "shuffle": True,
    }
    assert loaders.validation_loader.args == ("validation-dataset",)
    assert loaders.validation_loader.kwargs == {
        "batch_size": 32,
        "num_workers": 2,
        "shuffle": False,
    }


def test_collated_model_boundary_requires_float32_and_exact_dimensions() -> None:
    torch = pytest.importorskip("torch")
    good = SimpleNamespace(
        V=torch.zeros((3, 78), dtype=torch.float32),
        E=torch.zeros((4, 14), dtype=torch.float32),
    )
    assert_float32_model_boundary((good,))

    bad_dtype = SimpleNamespace(
        V=torch.zeros((3, 78), dtype=torch.float64),
        E=torch.zeros((4, 14), dtype=torch.float32),
    )
    with pytest.raises(GMCModelDataError, match="atom features must be float32"):
        assert_float32_model_boundary((bad_dtype,))

    bad_dimension = SimpleNamespace(
        V=torch.zeros((3, 77), dtype=torch.float32),
        E=torch.zeros((4, 14), dtype=torch.float32),
    )
    with pytest.raises(GMCModelDataError, match="dimension is not 78"):
        assert_float32_model_boundary((bad_dimension,))


def test_train_uses_only_frozen_scaler_transform_without_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _frozen_fixture(
        tmp_path,
        mean=np.asarray([1, 2, 3, 4, 5, 6], dtype=np.float64),
        scale=np.asarray([2, 2, 2, 2, 2, 2], dtype=np.float64),
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Frozen model data must never fit a scaler.")

    monkeypatch.setattr(scaling.StandardScaler, "fit", forbidden)
    monkeypatch.setattr(scaling.StandardScaler, "fit_transform", forbidden)
    monkeypatch.setattr(scaling.StandardScaler, "partial_fit", forbidden)
    development = _load_fixture(fixture)
    expected = (fixture["train_raw"] - np.arange(1, 7)) / 2.0

    np.testing.assert_array_equal(development.train.features[0].V_f, expected)
    np.testing.assert_array_equal(development.validation.features[0].V_f, expected)


def test_adapter_never_recomputes_geometry_or_ggl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _frozen_fixture(tmp_path)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Frozen model data must not recompute preprocessing.")

    monkeypatch.setattr(geometry, "generate_deterministic_geometry", forbidden)
    monkeypatch.setattr(ggl, "compute_ggl_features", forbidden)

    development = _load_fixture(fixture)

    assert len(development.train.features) == len(development.validation.features) == 1


def test_missing_feature_artifact_fails_instead_of_dropping_row(tmp_path: Path) -> None:
    fixture = _frozen_fixture(tmp_path)
    (fixture["train_dir"] / "raw_ggl" / "train-success.npz").unlink()

    with pytest.raises(GMCModelDataError, match="Missing TRAIN raw GGL"):
        _load_fixture(fixture)


def test_atom_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    fixture = _frozen_fixture(tmp_path)
    path = fixture["train_dir"] / "raw_ggl" / "train-success.npz"
    with np.load(path, allow_pickle=False) as artifact:
        payload = {key: artifact[key] for key in artifact.files if key != "artifact_content_sha256"}
    payload["heavy_atom_atomic_numbers"] = np.asarray([6, 7], dtype=np.int64)
    payload["artifact_content_sha256"] = np.asarray(scaling._npz_content_sha256(payload))
    scaling._write_deterministic_npz(path, payload)

    with pytest.raises(GMCModelDataError, match="atom identity/order"):
        _load_fixture(fixture)


def test_validation_scaled_raw_checksum_link_mismatch_is_rejected(tmp_path: Path) -> None:
    fixture = _frozen_fixture(tmp_path)
    path = fixture["validation_dir"] / "scaled_ggl" / "validation-success.npz"
    with np.load(path, allow_pickle=False) as artifact:
        payload = {key: artifact[key] for key in artifact.files if key != "artifact_content_sha256"}
    payload["raw_artifact_content_sha256"] = np.asarray("0" * 64)
    payload["artifact_content_sha256"] = np.asarray(scaling._npz_content_sha256(payload))
    scaling._write_deterministic_npz(path, payload)

    with pytest.raises(GMCModelDataError, match="does not link"):
        _load_fixture(fixture)


def test_float_formatted_manifest_counts_are_accepted(tmp_path: Path) -> None:
    fixture = _frozen_fixture(tmp_path, float_formatted_counts=True)

    development = _load_fixture(fixture)

    assert development.train.features[0].V_f.shape == (2, 6)


def test_wrong_split_is_rejected_before_path_access(tmp_path: Path) -> None:
    with pytest.raises(GMCModelDataError, match="test access is prohibited"):
        load_frozen_feature_split(
            tmp_path / "locked-test-must-not-exist",
            split="test",
            scaler=object(),  # type: ignore[arg-type]
            scaler_summary={},
        )


def test_locked_path_components_are_rejected_before_artifact_access(tmp_path: Path) -> None:
    for component in ("test", "locked", "locked-test", "locked_test", "bbb_test"):
        with pytest.raises(GMCModelDataError, match="prohibited"):
            load_frozen_feature_split(
                tmp_path / component / "does-not-exist",
                split="validation",
                scaler=object(),  # type: ignore[arg-type]
                scaler_summary={},
                contract=VALIDATION_CONTRACT,
            )


def test_ordered_train_input_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    fixture = _frozen_fixture(tmp_path, ordered_input_override="0" * 64)

    with pytest.raises(GMCModelDataError, match="ordered_input_artifact_sha256"):
        _load_fixture(fixture)


def test_count_mismatch_is_rejected_without_silent_drop(tmp_path: Path) -> None:
    fixture = _frozen_fixture(tmp_path)
    wrong_contract = FrozenSplitContract(
        split="train",
        source_rows=4,
        successful_molecules=2,
        heavy_atoms=2,
        exclusions=("eqvalan",),
        failures=(("spiclamine", "no_conformer"),),
    )
    scaler = scaling.load_frozen_ggl_scaler(fixture["scaler_dir"])
    scaler_summary = json.loads(
        (fixture["scaler_dir"] / scaling.FIT_SUMMARY_FILENAME).read_text(encoding="utf-8")
    )

    with pytest.raises(GMCModelDataError, match="differs from the contract"):
        load_frozen_feature_split(
            fixture["train_dir"],
            split="train",
            scaler=scaler,
            scaler_summary=scaler_summary,
            contract=wrong_contract,
        )


def _load_fixture(fixture: dict[str, Any]):
    return load_frozen_development_features(
        fixture["train_dir"],
        fixture["validation_dir"],
        fixture["scaler_dir"],
        train_contract=TRAIN_CONTRACT,
        validation_contract=VALIDATION_CONTRACT,
    )


def _frozen_fixture(
    root: Path,
    *,
    mean: np.ndarray | None = None,
    scale: np.ndarray | None = None,
    float_formatted_counts: bool = False,
    ordered_input_override: str | None = None,
) -> dict[str, Any]:
    mean = np.zeros(6, dtype=np.float64) if mean is None else mean
    scale = np.ones(6, dtype=np.float64) if scale is None else scale
    train_raw = np.arange(12, dtype=np.float64).reshape(2, 6) + 1
    train_dir = root / "train"
    validation_dir = root / "validation"
    scaler_dir = root / "scaler"
    train_dir.mkdir()
    validation_dir.mkdir()
    (train_dir / "raw_ggl").mkdir()
    (validation_dir / "raw_ggl").mkdir()
    (validation_dir / "scaled_ggl").mkdir()

    count_value: object = "2.0" if float_formatted_counts else 2
    column_value: object = "6.0" if float_formatted_counts else 6
    train_rows = [
        _manifest_row(
            "train-success",
            "train molecule",
            "CC",
            1,
            "train",
            "success",
            count_value,
            column_value,
        ),
        _manifest_row("train-eqvalan", "eqvalan", "C.C", 0, "train", "excluded", "", ""),
        _manifest_row(
            "train-sultamicillin", "sultamicillin", "C.N", 1, "train", "excluded", "", ""
        ),
        _manifest_row("train-spiclamine", "spiclamine", "N", 0, "train", "failed", "", ""),
    ]
    train_rows[1]["failure_category"] = "excluded_by_policy"
    train_rows[2]["failure_category"] = "excluded_by_policy"
    train_rows[3]["failure_category"] = "no_conformer"
    train_manifest = pd.DataFrame(train_rows)
    train_status = train_manifest[
        [
            "record_key",
            "molecule_id",
            "canonical_smiles",
            "target",
            "split",
            "status",
            "failure_category",
            "raw_ggl_path",
        ]
    ].copy()
    _write_csv(train_dir / "feature_manifest.csv", train_manifest)
    _write_csv(train_dir / "molecule_status.csv", train_status)
    train_payload = _raw_payload("train-success", "CC", train_raw, training=True)
    scaling._write_deterministic_npz(train_dir / "raw_ggl" / "train-success.npz", train_payload)
    train_summary = {
        **_summary_base("train", 4, 1, 2, 1, 2),
        "training_preprocessing_version": scaling.TRAINING_PREPROCESSING_VERSION,
        "ggl_scaled": False,
        "validation_artifact_accessed": False,
        "feature_manifest_sha256": _sha256(train_dir / "feature_manifest.csv"),
        "molecule_status_sha256": _sha256(train_dir / "molecule_status.csv"),
    }
    _write_json(train_dir / "preprocessing_summary.json", train_summary)

    train_artifact_path = train_dir / "raw_ggl" / "train-success.npz"
    ordered_entry = {
        "artifact_content_sha256": str(train_payload["artifact_content_sha256"].item()),
        "artifact_file_sha256": _sha256(train_artifact_path),
        "raw_ggl_path": "raw_ggl/train-success.npz",
        "record_key": "train-success",
        "source_row_index": 0,
    }
    ordered_input_hash = hashlib.sha256(
        (
            json.dumps(
                ordered_entry,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()

    _scaler_fixture(
        scaler_dir,
        mean,
        scale,
        train_summary["feature_manifest_sha256"],
        train_summary["molecule_status_sha256"],
        ordered_input_override or ordered_input_hash,
    )
    frozen = scaling.load_frozen_ggl_scaler(scaler_dir)
    validation_scaled = frozen.transform(train_raw)
    validation_row = _manifest_row(
        "validation-success",
        "validation molecule",
        "CC",
        0,
        "validation",
        "success",
        count_value,
        column_value,
    )
    validation_row["source_row_index"] = 0
    validation_row["scaled_ggl_path"] = "scaled_ggl/validation-success.npz"
    raw_payload = _raw_payload("validation-success", "CC", train_raw, training=False)
    raw_checksum = str(raw_payload["artifact_content_sha256"].item())
    scaled_payload = _scaled_payload(
        "validation-success", "CC", validation_scaled, frozen, raw_checksum
    )
    validation_row["raw_artifact_content_sha256"] = raw_checksum
    validation_row["scaled_artifact_content_sha256"] = str(
        scaled_payload["artifact_content_sha256"].item()
    )
    validation_manifest = pd.DataFrame([validation_row])
    validation_status = validation_manifest[
        [
            "record_key",
            "molecule_id",
            "canonical_smiles",
            "target",
            "split",
            "status",
            "failure_category",
            "raw_ggl_path",
        ]
    ].copy()
    _write_csv(validation_dir / "feature_manifest.csv", validation_manifest)
    _write_csv(validation_dir / "molecule_status.csv", validation_status)
    scaling._write_deterministic_npz(
        validation_dir / "raw_ggl" / "validation-success.npz", raw_payload
    )
    scaling._write_deterministic_npz(
        validation_dir / "scaled_ggl" / "validation-success.npz", scaled_payload
    )
    fit_summary = json.loads(
        (scaler_dir / scaling.FIT_SUMMARY_FILENAME).read_text(encoding="utf-8")
    )
    validation_summary = {
        **_summary_base("validation", 1, 1, 0, 0, 2),
        "validation_preprocessing_version": "gmc-mpnn-validation-raw-scaled-ggl-v1",
        "ggl_scaler_version": scaling.GGL_SCALER_VERSION,
        "portable_scaler_sha256": frozen.portable_scaler_sha256,
        "validation_artifact_accessed": True,
        "training_feature_manifest_sha256": fit_summary["feature_manifest_sha256"],
        "training_molecule_status_sha256": fit_summary["molecule_status_sha256"],
        "training_ordered_input_artifact_sha256": fit_summary["ordered_input_artifact_sha256"],
        "feature_manifest_sha256": _sha256(validation_dir / "feature_manifest.csv"),
        "molecule_status_sha256": _sha256(validation_dir / "molecule_status.csv"),
    }
    _write_json(validation_dir / "preprocessing_summary.json", validation_summary)
    return {
        "train_dir": train_dir,
        "validation_dir": validation_dir,
        "scaler_dir": scaler_dir,
        "train_raw": train_raw,
    }


def _manifest_row(
    record_key: str,
    molecule_id: str,
    smiles: str,
    target: int,
    split: str,
    status: str,
    count: object,
    columns: object,
) -> dict[str, Any]:
    return {
        "record_key": record_key,
        "molecule_id": molecule_id,
        "canonical_smiles": smiles,
        "target": target,
        "split": split,
        "geometry_smiles": smiles if status == "success" else "",
        "status": status,
        "failure_category": "",
        "raw_ggl_path": f"raw_ggl/{record_key}.npz" if status == "success" else "",
        "heavy_atom_count": count,
        "raw_ggl_rows": count,
        "raw_ggl_columns": columns,
        "standardization_version": GMC_STANDARDIZATION_VERSION,
        "geometry_fingerprint": f"geometry-{record_key}" if status == "success" else "",
        "ggl_fingerprint": f"ggl-{record_key}" if status == "success" else "",
        "optimization_method": "MMFF94s" if status == "success" else "",
        "rdkit_version": rdBase.rdkitVersion,
    }


def _summary_base(
    split: str,
    source: int,
    success: int,
    excluded: int,
    failed: int,
    atoms: int,
) -> dict[str, Any]:
    return {
        "loaded_split": split,
        "source_row_count": source,
        "successful_molecule_count": success,
        "policy_excluded_molecule_count": excluded,
        "failed_molecule_count": failed,
        "total_heavy_atom_count_among_successes": atoms,
        "standardization_version": GMC_STANDARDIZATION_VERSION,
        "geometry_preprocessing_version": GEOMETRY_PREPROCESSING_VERSION,
        "ggl_preprocessing_version": GGL_PREPROCESSING_VERSION,
        "ggl_feature_order": list(GGL_FEATURE_NAMES),
        "rdkit_version": rdBase.rdkitVersion,
        "test_artifact_accessed": False,
    }


def _common_payload(record_key: str, smiles: str) -> dict[str, np.ndarray]:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    return {
        "heavy_atom_atomic_numbers": np.asarray(
            [atom.GetAtomicNum() for atom in molecule.GetAtoms()], dtype=np.int64
        ),
        "heavy_atom_rdkit_indices": np.arange(molecule.GetNumAtoms(), dtype=np.int64),
        "ggl_feature_names": np.asarray(GGL_FEATURE_NAMES),
        "record_key": np.asarray(record_key),
        "standardization_version": np.asarray(GMC_STANDARDIZATION_VERSION),
        "geometry_preprocessing_version": np.asarray(GEOMETRY_PREPROCESSING_VERSION),
        "ggl_preprocessing_version": np.asarray(GGL_PREPROCESSING_VERSION),
        "geometry_smiles": np.asarray(smiles),
        "geometry_fingerprint": np.asarray(f"geometry-{record_key}"),
        "ggl_fingerprint": np.asarray(f"ggl-{record_key}"),
        "optimization_method": np.asarray("MMFF94s"),
        "rdkit_version": np.asarray(rdBase.rdkitVersion),
    }


def _raw_payload(
    record_key: str, smiles: str, values: np.ndarray, *, training: bool
) -> dict[str, np.ndarray]:
    payload = _common_payload(record_key, smiles)
    payload["raw_ggl_features"] = values
    version_key = (
        "training_preprocessing_version" if training else "validation_preprocessing_version"
    )
    version = (
        scaling.TRAINING_PREPROCESSING_VERSION
        if training
        else "gmc-mpnn-validation-raw-scaled-ggl-v1"
    )
    payload[version_key] = np.asarray(version)
    payload["artifact_content_sha256"] = np.asarray(scaling._npz_content_sha256(payload))
    return payload


def _scaled_payload(
    record_key: str,
    smiles: str,
    values: np.ndarray,
    frozen: scaling.FrozenGGLScaler,
    raw_checksum: str,
) -> dict[str, np.ndarray]:
    payload = _common_payload(record_key, smiles)
    payload.update(
        {
            "scaled_ggl_features": values,
            "validation_preprocessing_version": np.asarray("gmc-mpnn-validation-raw-scaled-ggl-v1"),
            "scaler_version": np.asarray(scaling.GGL_SCALER_VERSION),
            "portable_scaler_sha256": np.asarray(frozen.portable_scaler_sha256),
            "raw_artifact_content_sha256": np.asarray(raw_checksum),
        }
    )
    payload["artifact_content_sha256"] = np.asarray(scaling._npz_content_sha256(payload))
    return payload


def _scaler_fixture(
    path: Path,
    mean: np.ndarray,
    scale: np.ndarray,
    manifest_hash: str,
    status_hash: str,
    ordered_input_hash: str,
) -> None:
    path.mkdir()
    payload: dict[str, Any] = {
        "scaler_version": scaling.GGL_SCALER_VERSION,
        "scaler_type": "sklearn.preprocessing.StandardScaler",
        "with_mean": True,
        "with_std": True,
        "transformation": "(X - mean_) / scale_",
        "calculation_dtype": "float64",
        "feature_order": list(GGL_FEATURE_NAMES),
        "mean_": mean.tolist(),
        "var_": np.square(scale).tolist(),
        "scale_": scale.tolist(),
        "n_features_in_": 6,
        "n_samples_seen_": 2,
        "training_molecule_count": 1,
        "training_atom_count": 2,
        "source_row_count": 4,
        "feature_manifest_sha256": manifest_hash,
        "molecule_status_sha256": status_hash,
        "ordered_input_artifact_sha256": ordered_input_hash,
        "dataset": "BBB_Martins",
        "dataset_version": "synthetic",
        "loaded_split": "train",
        "training_preprocessing_version": scaling.TRAINING_PREPROCESSING_VERSION,
        "standardization_version": GMC_STANDARDIZATION_VERSION,
        "geometry_preprocessing_version": GEOMETRY_PREPROCESSING_VERSION,
        "ggl_preprocessing_version": GGL_PREPROCESSING_VERSION,
        "rdkit_version": rdBase.rdkitVersion,
        "numpy_version": np.__version__,
        "scikit_learn_version": "synthetic",
        "git_commit": "synthetic",
        "validation_artifact_accessed": False,
        "test_artifact_accessed": False,
    }
    payload["portable_scaler_sha256"] = scaling._portable_scaler_sha256(payload)
    scaler_json = path / scaling.SCALER_JSON_FILENAME
    scaling._write_json(scaler_json, payload)
    npz_payload = {
        "scaler_version": np.asarray(scaling.GGL_SCALER_VERSION),
        "feature_order": np.asarray(GGL_FEATURE_NAMES),
        "mean_": mean,
        "var_": np.square(scale),
        "scale_": scale,
        "n_features_in_": np.asarray(6, dtype=np.int64),
        "n_samples_seen_": np.asarray(2, dtype=np.int64),
        "portable_scaler_sha256": np.asarray(payload["portable_scaler_sha256"]),
    }
    scaler_npz = path / scaling.SCALER_NPZ_FILENAME
    scaling._write_deterministic_npz(scaler_npz, npz_payload)
    fit_summary = {
        **payload,
        "scaler_json_sha256": _sha256(scaler_json),
        "scaler_npz_sha256": _sha256(scaler_npz),
    }
    scaling._write_json(path / scaling.FIT_SUMMARY_FILENAME, fit_summary)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
