from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem, rdBase

from admet_platform.chemprop.config import ChempropExperimentConfig
from admet_platform.gmc_mpnn import scaling
from admet_platform.gmc_mpnn.geometry import GeometryConfig, GeometryResult
from admet_platform.gmc_mpnn.ggl import GGL_FEATURE_NAMES, GGLConfig, GGLResult
from scripts import preprocess_gmc_mpnn_validation as preprocessing


def test_validation_run_loads_frozen_scaler_without_any_fit_and_transforms_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, [_row("one", "CCO", 1)])
    scaler_dir = _scaler_fixture(
        tmp_path / "scaler",
        mean=np.asarray([1, 2, 3, 4, 5, 6], dtype=np.float64),
        scale=np.asarray([2, 4, 5, 8, 10, 20], dtype=np.float64),
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Validation preprocessing must never fit a scaler.")

    monkeypatch.setattr(scaling.StandardScaler, "fit", forbidden)
    monkeypatch.setattr(scaling.StandardScaler, "fit_transform", forbidden)
    monkeypatch.setattr(scaling.StandardScaler, "partial_fit", forbidden)
    output = tmp_path / "output"
    summary = _run(config, scaler_dir, output)

    manifest = pd.read_csv(output / preprocessing.MANIFEST_FILENAME)
    raw_path = output / str(manifest.iloc[0]["raw_ggl_path"])
    scaled_path = output / str(manifest.iloc[0]["scaled_ggl_path"])
    with (
        np.load(raw_path, allow_pickle=False) as raw,
        np.load(scaled_path, allow_pickle=False) as scaled,
    ):
        expected = (raw["raw_ggl_features"] - np.arange(1, 7)) / np.asarray([2, 4, 5, 8, 10, 20])
        np.testing.assert_array_equal(scaled["scaled_ggl_features"], expected)
        assert scaled["scaled_ggl_features"].dtype == np.float64
    assert summary["successful_molecule_count"] == 1


def test_validation_only_access_and_test_flag_are_explicit(tmp_path: Path) -> None:
    config = _config(tmp_path, [_row("one", "CC", 0)])
    scaler_dir = _scaler_fixture(tmp_path / "scaler")
    observed: list[tuple[Path, str]] = []

    def loader(path: Path, *, split: str) -> pd.DataFrame:
        observed.append((path, split))
        return preprocessing.load_bbb_development_split(path, split=split)

    summary = _run(config, scaler_dir, tmp_path / "output", validation_loader=loader)

    assert observed == [(config.prepared_root / "valid.csv", "validation")]
    assert summary["loaded_split"] == "validation"
    assert summary["validation_artifact_accessed"] is True
    assert summary["test_artifact_accessed"] is False


def test_locked_test_access_is_rejected_before_output_is_published(tmp_path: Path) -> None:
    config = _config(tmp_path, [_row("one", "CC", 0)])
    scaler_dir = _scaler_fixture(tmp_path / "scaler")
    output = tmp_path / "output"

    def malicious_loader(path: Path, *, split: str) -> pd.DataFrame:
        pd.read_csv(config.prepared_root / config.split_files["test"])
        raise AssertionError("The test read must be rejected first.")

    with pytest.raises(preprocessing.ProhibitedArtifactAccessError, match="test artifact"):
        _run(config, scaler_dir, output, validation_loader=malicious_loader)

    assert not output.exists()


def test_swallowed_test_access_attempt_still_blocks_publication(tmp_path: Path) -> None:
    config = _config(tmp_path, [_row("one", "CC", 0)])
    scaler_dir = _scaler_fixture(tmp_path / "scaler")
    output = tmp_path / "output"

    def swallowing_loader(path: Path, *, split: str) -> pd.DataFrame:
        try:
            pd.read_csv(config.prepared_root / config.split_files["test"])
        except preprocessing.ProhibitedArtifactAccessError:
            pass
        return preprocessing.load_bbb_development_split(path, split=split)

    with pytest.raises(preprocessing.ProhibitedArtifactAccessError, match="attempt"):
        _run(config, scaler_dir, output, validation_loader=swallowing_loader)

    assert not output.exists()


def test_test_access_from_geometry_callback_still_blocks_publication(tmp_path: Path) -> None:
    config = _config(tmp_path, [_row("one", "CC", 0)])
    scaler_dir = _scaler_fixture(tmp_path / "scaler")
    output = tmp_path / "output"

    def accessing_geometry(smiles: str, *, config: GeometryConfig) -> GeometryResult:
        try:
            pd.read_csv(tmp_path / "prepared" / "locked.csv")
        except preprocessing.ProhibitedArtifactAccessError:
            pass
        return _geometry_result(smiles, config)

    with pytest.raises(preprocessing.ProhibitedArtifactAccessError, match="attempt"):
        _run(config, scaler_dir, output, geometry_function=accessing_geometry)

    assert not output.exists()


def test_raw_and_scaled_artifacts_retain_identical_atom_alignment_and_provenance(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, [_row("one", "CCO", 1)])
    scaler_dir = _scaler_fixture(tmp_path / "scaler")
    output = tmp_path / "output"
    summary = _run(config, scaler_dir, output)
    manifest = pd.read_csv(output / preprocessing.MANIFEST_FILENAME, keep_default_na=False)
    row = manifest.iloc[0]

    with (
        np.load(output / row["raw_ggl_path"], allow_pickle=False) as raw,
        np.load(output / row["scaled_ggl_path"], allow_pickle=False) as scaled,
    ):
        assert set(raw.files) == preprocessing.RAW_NPZ_KEYS
        assert set(scaled.files) == preprocessing.SCALED_NPZ_KEYS
        np.testing.assert_array_equal(
            raw["heavy_atom_atomic_numbers"], scaled["heavy_atom_atomic_numbers"]
        )
        np.testing.assert_array_equal(
            raw["heavy_atom_rdkit_indices"], scaled["heavy_atom_rdkit_indices"]
        )
        assert raw["raw_ggl_features"].shape == scaled["scaled_ggl_features"].shape == (3, 6)
        assert (
            str(scaled["raw_artifact_content_sha256"].item()) == row["raw_artifact_content_sha256"]
        )
        assert str(scaled["portable_scaler_sha256"].item()) == summary["portable_scaler_sha256"]
    assert row["portable_scaler_sha256"] == summary["portable_scaler_sha256"]
    assert summary["feature_manifest_sha256"] == _sha256(output / preprocessing.MANIFEST_FILENAME)


@pytest.mark.parametrize(
    "field,value,message",
    (
        ("geometry_preprocessing_version", "old-geometry", "disagree"),
        ("ordered_input_artifact_sha256", "bad", "disagree"),
    ),
)
def test_scaler_version_or_training_hash_mismatch_is_rejected(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    config = _config(tmp_path, [_row("one", "CC", 0)])
    scaler_dir = _scaler_fixture(tmp_path / "scaler")
    fit_summary_path = scaler_dir / scaling.FIT_SUMMARY_FILENAME
    fit_summary = json.loads(fit_summary_path.read_text(encoding="utf-8"))
    fit_summary[field] = value
    _write_json(fit_summary_path, fit_summary)

    with pytest.raises(scaling.GGLScalingError, match=message):
        _run(config, scaler_dir, tmp_path / "output")


def test_scaler_train_rdkit_mismatch_is_rejected(tmp_path: Path) -> None:
    scaler_dir = _scaler_fixture(tmp_path / "scaler")
    frozen = scaling.load_frozen_ggl_scaler(scaler_dir)
    fit_summary_path = scaler_dir / scaling.FIT_SUMMARY_FILENAME
    fit_summary = json.loads(fit_summary_path.read_text(encoding="utf-8"))
    fit_summary["rdkit_version"] = "different-rdkit"
    _write_json(fit_summary_path, fit_summary)

    with pytest.raises(scaling.GGLScalingError, match="rdkit_version"):
        preprocessing._load_scaler_provenance(scaler_dir, frozen)


def test_validation_source_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path, [_row("one", "CC", 0)])
    scaler_dir = _scaler_fixture(tmp_path / "scaler")
    pd.DataFrame([_row("different", "CO", 1)]).to_csv(
        config.prepared_root / "valid.csv", index=False
    )

    with pytest.raises(preprocessing.ValidationPreprocessingError, match="split manifest"):
        _run(config, scaler_dir, tmp_path / "output")


def test_scaled_artifact_version_mismatch_is_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path, [_row("one", "CC", 0)])
    scaler_dir = _scaler_fixture(tmp_path / "scaler")
    output = tmp_path / "output"
    _run(config, scaler_dir, output)
    manifest = pd.read_csv(output / preprocessing.MANIFEST_FILENAME)
    raw_path = output / str(manifest.iloc[0]["raw_ggl_path"])
    scaled_path = output / str(manifest.iloc[0]["scaled_ggl_path"])
    with np.load(scaled_path, allow_pickle=False) as artifact:
        payload = {key: artifact[key] for key in artifact.files}
    payload["scaler_version"] = np.asarray("wrong-scaler-version")
    preprocessing._atomic_write_npz(scaled_path, payload)

    with pytest.raises(preprocessing.ValidationPreprocessingError, match="scaler version"):
        preprocessing._validate_artifact_pair(
            raw_path,
            scaled_path,
            expected_record_key=str(manifest.iloc[0]["record_key"]),
            expected_geometry_smiles=str(manifest.iloc[0]["geometry_smiles"]),
            scaler=scaling.load_frozen_ggl_scaler(scaler_dir),
        )


def test_raw_artifact_checksum_mismatch_is_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path, [_row("one", "CC", 0)])
    scaler_dir = _scaler_fixture(tmp_path / "scaler")
    output = tmp_path / "output"
    _run(config, scaler_dir, output)
    manifest = pd.read_csv(output / preprocessing.MANIFEST_FILENAME)
    raw_path = output / str(manifest.iloc[0]["raw_ggl_path"])
    scaled_path = output / str(manifest.iloc[0]["scaled_ggl_path"])
    with np.load(raw_path, allow_pickle=False) as artifact:
        payload = {key: artifact[key] for key in artifact.files}
    payload["artifact_content_sha256"] = np.asarray("0" * 64)
    preprocessing._atomic_write_npz(raw_path, payload)

    with pytest.raises(preprocessing.ValidationPreprocessingError, match="checksum"):
        preprocessing._validate_artifact_pair(
            raw_path,
            scaled_path,
            expected_record_key=str(manifest.iloc[0]["record_key"]),
            expected_geometry_smiles=str(manifest.iloc[0]["geometry_smiles"]),
            scaler=scaling.load_frozen_ggl_scaler(scaler_dir),
        )


def test_nonfinite_raw_features_are_preserved_as_explicit_failure(tmp_path: Path) -> None:
    config = _config(tmp_path, [_row("bad", "CC", 0)])
    scaler_dir = _scaler_fixture(tmp_path / "scaler")

    def nonfinite_ggl(
        coordinates: np.ndarray,
        atomic_numbers: np.ndarray,
        *,
        geometry_fingerprint: str,
        config: GGLConfig,
    ) -> GGLResult:
        result = _ggl_result(
            coordinates,
            atomic_numbers,
            geometry_fingerprint=geometry_fingerprint,
            config=config,
        )
        values = result.features.copy()
        values[0, 0] = np.nan
        return GGLResult(**{**result.__dict__, "features": values})

    output = tmp_path / "output"
    summary = _run(config, scaler_dir, output, ggl_function=nonfinite_ggl)
    manifest = pd.read_csv(output / preprocessing.MANIFEST_FILENAME, keep_default_na=False)

    assert summary["failed_molecule_count"] == 1
    assert summary["successful_molecule_count"] == 0
    assert manifest.iloc[0]["status"] == "failed"
    assert "finite" in manifest.iloc[0]["failure_message"]
    assert list((output / preprocessing.RAW_GGL_DIRECTORY).iterdir()) == []
    assert list((output / preprocessing.SCALED_GGL_DIRECTORY).iterdir()) == []


def test_failure_and_policy_exclusion_are_not_silently_dropped(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        [_row("success", "CC", 1), _row("eqvalan", "CC.C", 0), _row("failed", "N", 1)],
    )
    scaler_dir = _scaler_fixture(tmp_path / "scaler")

    def selective_geometry(smiles: str, *, config: GeometryConfig) -> GeometryResult:
        if smiles == "N":
            raise RuntimeError("synthetic geometry failure")
        return _geometry_result(smiles, config)

    output = tmp_path / "output"
    summary = _run(config, scaler_dir, output, geometry_function=selective_geometry)
    status = pd.read_csv(output / preprocessing.STATUS_FILENAME, keep_default_na=False)

    assert len(status) == 3
    assert status["status"].tolist() == ["success", "excluded", "failed"]
    assert summary["successful_molecule_count"] == 1
    assert summary["policy_excluded_molecule_count"] == 1
    assert summary["failed_molecule_count"] == 1


def test_record_keys_manifests_and_content_hashes_are_deterministic(tmp_path: Path) -> None:
    config = _config(tmp_path, [_row("one", "CCO", 1), _row("two", "CN", 0)])
    scaler_dir = _scaler_fixture(tmp_path / "scaler")
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_summary = _run(config, scaler_dir, first)
    second_summary = _run(config, scaler_dir, second)
    first_manifest = pd.read_csv(first / preprocessing.MANIFEST_FILENAME, keep_default_na=False)
    second_manifest = pd.read_csv(second / preprocessing.MANIFEST_FILENAME, keep_default_na=False)

    pd.testing.assert_frame_equal(first_manifest, second_manifest)
    for field in ("ordered_raw_artifact_sha256", "ordered_scaled_artifact_sha256"):
        assert first_summary[field] == second_summary[field]
    assert first_summary["feature_manifest_sha256"] == second_summary["feature_manifest_sha256"]
    assert first_summary["molecule_status_sha256"] == second_summary["molecule_status_sha256"]


def test_default_contract_requires_exactly_196_validation_rows(tmp_path: Path) -> None:
    config = _config(tmp_path, [_row("one", "CC", 0)])
    scaler_dir = _scaler_fixture(tmp_path / "scaler")

    with pytest.raises(preprocessing.ValidationPreprocessingError, match="expected 196"):
        preprocessing.run_validation_preprocessing(
            config.source_path,
            scaler_dir=scaler_dir,
            output_dir=tmp_path / "output",
            config_loader=lambda _: config,
        )


def _run(
    config: ChempropExperimentConfig,
    scaler_dir: Path,
    output: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    return preprocessing.run_validation_preprocessing(
        config.source_path,
        scaler_dir=scaler_dir,
        output_dir=output,
        expected_source_row_count=len(pd.read_csv(config.prepared_root / "valid.csv")),
        config_loader=lambda _: config,
        geometry_function=kwargs.pop("geometry_function", _geometry_result),
        ggl_function=kwargs.pop("ggl_function", _ggl_result),
        **kwargs,
    )


def _row(label: str, smiles: str, target: int) -> dict[str, object]:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    return {
        "molecule_id": label,
        "smiles": canonical,
        "canonical_smiles": canonical,
        "target": target,
        "split": "validation",
    }


def _config(root: Path, rows: list[dict[str, object]]) -> ChempropExperimentConfig:
    prepared = root / "prepared"
    prepared.mkdir()
    pd.DataFrame([_row("train-sentinel", "O", 0) | {"split": "train"}]).to_csv(
        prepared / "train.csv", index=False
    )
    pd.DataFrame(rows).to_csv(prepared / "valid.csv", index=False)
    (prepared / "locked.csv").write_text("must-not-be-read", encoding="utf-8")
    validation_sha256 = _sha256(prepared / "valid.csv")
    split_manifest = root / "split-manifest.json"
    split_manifest_payload = {
        "split_manifest_id": "synthetic-validation-manifest",
        "validation": validation_sha256,
    }
    _write_json(split_manifest, split_manifest_payload)
    return ChempropExperimentConfig(
        source_path=root / "config.yaml",
        raw={
            "split_manifest_id": "synthetic-validation-manifest",
            "split_manifest_sha256": _sha256(split_manifest),
        },
        endpoint="bbb_martins",
        endpoint_id="bbb_martins",
        dataset="BBB_Martins",
        dataset_version="synthetic-validation-preprocessing-fixture",
        task_type="binary_classification",
        primary_metric="auroc",
        prepared_root=prepared,
        split_manifest=split_manifest,
        split_files={"train": "train.csv", "validation": "valid.csv", "test": "locked.csv"},
        split_hash_keys={"train": "train", "validation": "validation", "test": "test"},
        tasks={},
        model={},
        training={},
    )


def _geometry_result(smiles: str, config: GeometryConfig) -> GeometryResult:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    atoms = tuple(atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() != 1)
    count = len(atoms)
    coordinates = np.zeros((count, 3), dtype=np.float64)
    coordinates[:, 0] = np.arange(count, dtype=np.float64)
    return GeometryResult(
        input_smiles=smiles,
        canonical_isomeric_smiles=smiles,
        seed=config.seed,
        embedding_seed=12345,
        etkdg_version="ETKDGv3",
        requested_conformer_count=config.num_conformers,
        generated_conformer_count=3,
        optimization_method="MMFF94s",
        selected_conformer_id=1,
        selected_energy=-2.5,
        convergence_status="converged",
        heavy_atom_count=count,
        heavy_atom_rdkit_indices=tuple(range(count)),
        heavy_atom_atomic_numbers=tuple(atom.GetAtomicNum() for atom in atoms),
        heavy_atom_formal_charges=tuple(atom.GetFormalCharge() for atom in atoms),
        coordinates=coordinates,
        rdkit_version=rdBase.rdkitVersion,
        geometry_status="success_mmff94s",
        geometry_fingerprint=f"geometry-{smiles}",
        config=config,
    )


def _ggl_result(
    coordinates: np.ndarray,
    atomic_numbers: np.ndarray,
    *,
    geometry_fingerprint: str,
    config: GGLConfig,
) -> GGLResult:
    rows = len(atomic_numbers)
    features = np.arange(rows * 6, dtype=np.float64).reshape(rows, 6)
    return GGLResult(
        features=features,
        feature_names=GGL_FEATURE_NAMES,
        geometry_fingerprint=geometry_fingerprint,
        ggl_fingerprint=f"ggl-{geometry_fingerprint}",
        ggl_status="success",
        config=config,
        preprocessing_version="test",
        element_radius_mapping_version="test",
    )


def _scaler_fixture(
    path: Path,
    *,
    mean: np.ndarray | None = None,
    scale: np.ndarray | None = None,
) -> Path:
    path.mkdir()
    mean = np.zeros(6, dtype=np.float64) if mean is None else mean
    scale = np.ones(6, dtype=np.float64) if scale is None else scale
    variance = np.square(scale)
    payload: dict[str, Any] = {
        "scaler_version": scaling.GGL_SCALER_VERSION,
        "scaler_type": "sklearn.preprocessing.StandardScaler",
        "with_mean": True,
        "with_std": True,
        "transformation": "(X - mean_) / scale_",
        "calculation_dtype": "float64",
        "feature_order": list(GGL_FEATURE_NAMES),
        "mean_": mean.tolist(),
        "var_": variance.tolist(),
        "scale_": scale.tolist(),
        "n_features_in_": 6,
        "n_samples_seen_": 100,
        "training_molecule_count": 10,
        "training_atom_count": 100,
        "source_row_count": 12,
        "feature_manifest_sha256": "1" * 64,
        "molecule_status_sha256": "2" * 64,
        "ordered_input_artifact_sha256": "3" * 64,
        "dataset": "BBB_Martins",
        "dataset_version": "synthetic",
        "loaded_split": "train",
        "training_preprocessing_version": scaling.TRAINING_PREPROCESSING_VERSION,
        "standardization_version": preprocessing.GMC_STANDARDIZATION_VERSION,
        "geometry_preprocessing_version": preprocessing.GEOMETRY_PREPROCESSING_VERSION,
        "ggl_preprocessing_version": preprocessing.GGL_PREPROCESSING_VERSION,
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
        "var_": variance,
        "scale_": scale,
        "n_features_in_": np.asarray(6, dtype=np.int64),
        "n_samples_seen_": np.asarray(100, dtype=np.int64),
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
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
