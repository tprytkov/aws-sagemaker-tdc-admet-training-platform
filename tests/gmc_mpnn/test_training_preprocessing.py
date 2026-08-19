from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem, rdBase

from admet_platform.chemprop.config import ChempropExperimentConfig
from admet_platform.gmc_mpnn.geometry import GeometryConfig, GeometryError, GeometryResult
from admet_platform.gmc_mpnn.ggl import GGLConfig, GGLResult
from scripts import preprocess_gmc_mpnn_training as preprocessing


def test_training_only_loader_and_access_summary(
    tmp_path: Path,
) -> None:
    rows = [_row("connected", "CCO", 1)]
    config = _config(tmp_path, rows)
    calls: list[tuple[Path, str]] = []

    def training_loader(path: Path, *, split: str) -> pd.DataFrame:
        calls.append((path, split))
        return pd.DataFrame(rows)

    summary = _run(
        config,
        tmp_path / "output",
        training_loader=training_loader,
    )

    assert calls == [(config.prepared_root / "train.csv", "train")]
    assert summary["loaded_split"] == "train"
    assert summary["source_row_count"] == 1
    assert summary["validation_artifact_accessed"] is False
    assert summary["test_artifact_accessed"] is False


@pytest.mark.parametrize("blocked_split", ("validation", "test"))
def test_nontraining_artifact_access_is_blocked(tmp_path: Path, blocked_split: str) -> None:
    rows = [_row("connected", "CCO", 1)]
    config = _config(tmp_path, rows)

    def forbidden_loader(path: Path, *, split: str) -> pd.DataFrame:
        assert split == "train"
        blocked = config.prepared_root / config.split_files[blocked_split]
        blocked.read_bytes()
        raise AssertionError("unreachable")

    with pytest.raises(preprocessing.NonTrainingArtifactAccessError):
        _run(
            config,
            tmp_path / "output",
            training_loader=forbidden_loader,
        )


def test_source_identity_success_parent_selection_and_exclusion_manifest(
    tmp_path: Path,
) -> None:
    rows = [
        _row("connected", "CCO", 1),
        _row("salt", "CCN.[Cl-]", 0),
        _row("eqvalan", "CCO.[Na+]", 1),
    ]
    config = _config(tmp_path, rows)
    output = tmp_path / "output"
    geometry_calls: list[str] = []
    ggl_calls = 0

    def geometry_function(smiles: str, *, config: GeometryConfig) -> GeometryResult:
        geometry_calls.append(smiles)
        return _geometry_result(smiles, config)

    def ggl_function(
        coordinates: np.ndarray,
        atomic_numbers: np.ndarray,
        *,
        geometry_fingerprint: str,
        config: GGLConfig,
    ) -> GGLResult:
        nonlocal ggl_calls
        ggl_calls += 1
        return _ggl_result(
            coordinates,
            atomic_numbers,
            geometry_fingerprint=geometry_fingerprint,
            config=config,
        )

    summary = _run(
        config,
        output,
        geometry_function=geometry_function,
        ggl_function=ggl_function,
    )
    manifest = pd.read_csv(output / preprocessing.MANIFEST_FILENAME, keep_default_na=False)
    by_id = manifest.set_index("molecule_id")

    assert len(manifest) == len(rows)
    assert by_id.loc["connected", "canonical_smiles"] == _canonical("CCO")
    assert by_id.loc["connected", "geometry_smiles"] == _canonical("CCO")
    assert by_id.loc["connected", "standardization_action"] == "unchanged"
    assert by_id.loc["connected", "status"] == "success"
    assert by_id.loc["connected", "raw_ggl_path"]

    salt_source = _canonical("CCN.[Cl-]")
    assert by_id.loc["salt", "canonical_smiles"] == salt_source
    assert by_id.loc["salt", "geometry_smiles"] == "CCN"
    assert by_id.loc["salt", "standardization_action"] == "parent_selected"
    assert json.loads(by_id.loc["salt", "removed_fragment_smiles"]) == ["[Cl-]"]
    assert by_id.loc["salt", "status"] == "success"
    assert by_id.loc["salt", "raw_ggl_path"]

    assert by_id.loc["eqvalan", "canonical_smiles"] == _canonical("CCO.[Na+]")
    assert by_id.loc["eqvalan", "status"] == "excluded"
    assert by_id.loc["eqvalan", "failure_category"] == "excluded_by_policy"
    assert by_id.loc["eqvalan", "raw_ggl_path"] == ""
    assert by_id.loc["eqvalan", "exclusion_reason"] == "ambiguous_multi_active_mixture"
    assert summary["successful_molecule_count"] == 2
    assert summary["policy_excluded_molecule_count"] == 1
    assert summary["failed_molecule_count"] == 0
    assert geometry_calls == ["CCO", "CCN"]
    assert ggl_calls == 2


def test_failure_row_is_retained_without_feature_path(tmp_path: Path) -> None:
    rows = [_row("good", "CCO", 1), _row("bad", "CCN", 0)]
    config = _config(tmp_path, rows)
    output = tmp_path / "output"

    def geometry_function(smiles: str, *, config: GeometryConfig) -> GeometryResult:
        if smiles == "CCN":
            raise GeometryError("embedding_failed", "synthetic failure")
        return _geometry_result(smiles, config)

    summary = _run(config, output, geometry_function=geometry_function)
    manifest = pd.read_csv(output / preprocessing.MANIFEST_FILENAME, keep_default_na=False)
    failed = manifest.loc[manifest["molecule_id"] == "bad"].iloc[0]

    assert len(manifest) == 2
    assert failed["status"] == "failed"
    assert failed["failure_category"] == "embedding_failed"
    assert "synthetic failure" in failed["failure_message"]
    assert failed["raw_ggl_path"] == ""
    assert summary["successful_molecule_count"] == 1
    assert summary["failed_molecule_count"] == 1


def test_record_key_is_deterministic_and_uses_original_source_identity() -> None:
    first = preprocessing.stable_record_key("id", "CCO.[Na+]", 1, "train")
    repeated = preprocessing.stable_record_key("id", "CCO.[Na+]", 1, "train")

    assert first == repeated
    assert len(first) == 64
    assert first != preprocessing.stable_record_key("id", "CCO.[K+]", 1, "train")
    assert first != preprocessing.stable_record_key("id", "CCO.[Na+]", 0, "train")
    assert first != preprocessing.stable_record_key("id", "CCO.[Na+]", 1, "validation")
    assert first != preprocessing.stable_record_key(
        "id", "CCO.[Na+]", 1, "train", preprocessing_version="future-version"
    )


def test_raw_npz_schema_alignment_finiteness_and_no_scaling(tmp_path: Path) -> None:
    rows = [_row("connected", "CCO", 1)]
    config = _config(tmp_path, rows)
    output = tmp_path / "output"

    summary = _run(config, output)
    manifest = pd.read_csv(output / preprocessing.MANIFEST_FILENAME)
    artifact_path = output / str(manifest.iloc[0]["raw_ggl_path"])

    with np.load(artifact_path, allow_pickle=False) as artifact:
        assert set(artifact.files) == preprocessing.NPZ_KEYS
        features = artifact["raw_ggl_features"]
        atomic_numbers = artifact["heavy_atom_atomic_numbers"]
        indices = artifact["heavy_atom_rdkit_indices"]
        assert features.shape == (3, 6)
        assert atomic_numbers.shape == (3,)
        assert indices.tolist() == [0, 1, 2]
        assert np.isfinite(features).all()
        assert features[0].tolist() == pytest.approx([0, 1, 2, 3, 4, 5])
        assert "mean" not in artifact.files
        assert "standard_deviation" not in artifact.files
        assert "scaled_features" not in artifact.files
    assert summary["ggl_feature_order"] == [
        "minimum",
        "maximum",
        "sum",
        "mean",
        "median",
        "population_standard_deviation",
    ]
    assert summary["ggl_scaled"] is False


def test_nonfinite_features_fail_without_writing_npz(tmp_path: Path) -> None:
    rows = [_row("nonfinite", "CCO", 1)]
    config = _config(tmp_path, rows)
    output = tmp_path / "output"

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
        features = result.features.copy()
        features[0, 0] = np.nan
        return GGLResult(
            features=features,
            feature_names=result.feature_names,
            geometry_fingerprint=result.geometry_fingerprint,
            ggl_fingerprint=result.ggl_fingerprint,
            ggl_status=result.ggl_status,
            config=result.config,
            preprocessing_version=result.preprocessing_version,
            element_radius_mapping_version=result.element_radius_mapping_version,
        )

    _run(config, output, ggl_function=nonfinite_ggl)
    manifest = pd.read_csv(output / preprocessing.MANIFEST_FILENAME, keep_default_na=False)

    assert manifest.iloc[0]["status"] == "failed"
    assert manifest.iloc[0]["failure_category"] == "nonfinite_ggl"
    assert manifest.iloc[0]["raw_ggl_path"] == ""
    assert list((output / preprocessing.RAW_GGL_DIRECTORY).glob("*.npz")) == []


def test_feature_and_status_manifests_contain_every_input_row(tmp_path: Path) -> None:
    rows = [
        _row("success", "CCO", 1),
        _row("excluded", "CCN.[Cl-]", 0, molecule_id="sultamicillin"),
        _row("failure", "CCC", 0),
    ]
    config = _config(tmp_path, rows)
    output = tmp_path / "output"

    def geometry_function(smiles: str, *, config: GeometryConfig) -> GeometryResult:
        if smiles == "CCC":
            raise GeometryError("optimization_failed", "synthetic")
        return _geometry_result(smiles, config)

    _run(config, output, geometry_function=geometry_function)
    manifest = pd.read_csv(output / preprocessing.MANIFEST_FILENAME)
    status = pd.read_csv(output / preprocessing.STATUS_FILENAME)

    assert manifest["molecule_id"].tolist() == [
        "success",
        "sultamicillin",
        "failure",
    ]
    assert status["molecule_id"].tolist() == manifest["molecule_id"].tolist()
    assert len(manifest) == len(status) == len(rows)


def test_npz_and_final_tables_use_atomic_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [_row("connected", "CCO", 1)]
    config = _config(tmp_path, rows)
    output = tmp_path / "output"
    replacements: list[tuple[Path, Path]] = []
    original_replace = os.replace

    def tracking_replace(source: str | Path, destination: str | Path) -> None:
        replacements.append((Path(source), Path(destination)))
        original_replace(source, destination)

    monkeypatch.setattr(preprocessing.os, "replace", tracking_replace)

    _run(config, output)

    destinations = {destination.name for _, destination in replacements}
    assert preprocessing.SUMMARY_FILENAME in destinations
    assert preprocessing.MANIFEST_FILENAME in destinations
    assert preprocessing.STATUS_FILENAME in destinations
    assert any(destination.suffix == ".npz" for _, destination in replacements)
    assert all(source.suffix == ".tmp" for source, _ in replacements)
    assert list(output.rglob("*.tmp")) == []


def test_resume_reuses_valid_artifact_and_keeps_manifests_deterministic(
    tmp_path: Path,
) -> None:
    rows = [_row("one", "CCO", 1), _row("two", "CCN.[Cl-]", 0)]
    config = _config(tmp_path, rows)
    output = tmp_path / "output"
    _run(config, output)
    first_manifest = (output / preprocessing.MANIFEST_FILENAME).read_bytes()
    first_status = (output / preprocessing.STATUS_FILENAME).read_bytes()

    def forbidden_geometry(*args: object, **kwargs: object) -> GeometryResult:
        raise AssertionError("geometry should have been reused")

    def forbidden_ggl(*args: object, **kwargs: object) -> GGLResult:
        raise AssertionError("GGL should have been reused")

    summary = _run(
        config,
        output,
        resume=True,
        geometry_function=forbidden_geometry,
        ggl_function=forbidden_ggl,
    )

    assert summary["reused_successful_artifact_count"] == 2
    assert (output / preprocessing.MANIFEST_FILENAME).read_bytes() == first_manifest
    assert (output / preprocessing.STATUS_FILENAME).read_bytes() == first_status


def test_resume_rejects_incompatible_summary_version(tmp_path: Path) -> None:
    rows = [_row("one", "CCO", 1)]
    config = _config(tmp_path, rows)
    output = tmp_path / "output"
    _run(config, output)
    summary_path = output / preprocessing.SUMMARY_FILENAME
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["training_preprocessing_version"] = "incompatible-version"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(preprocessing.IncompatibleResumeError, match="incompatible"):
        _run(config, output, resume=True)


def test_resume_regenerates_corrupt_artifact(tmp_path: Path) -> None:
    rows = [_row("one", "CCO", 1)]
    config = _config(tmp_path, rows)
    output = tmp_path / "output"
    _run(config, output)
    manifest = pd.read_csv(output / preprocessing.MANIFEST_FILENAME)
    artifact_path = output / str(manifest.iloc[0]["raw_ggl_path"])
    artifact_path.write_bytes(b"corrupt")
    geometry_calls = 0

    def tracking_geometry(smiles: str, *, config: GeometryConfig) -> GeometryResult:
        nonlocal geometry_calls
        geometry_calls += 1
        return _geometry_result(smiles, config)

    summary = _run(
        config,
        output,
        resume=True,
        geometry_function=tracking_geometry,
    )

    assert geometry_calls == 1
    assert summary["reused_successful_artifact_count"] == 0
    with np.load(artifact_path, allow_pickle=False) as artifact:
        assert artifact["raw_ggl_features"].shape == (3, 6)


def test_resume_regenerates_artifact_with_incompatible_internal_version(
    tmp_path: Path,
) -> None:
    rows = [_row("one", "CCO", 1)]
    config = _config(tmp_path, rows)
    output = tmp_path / "output"
    _run(config, output)
    manifest = pd.read_csv(output / preprocessing.MANIFEST_FILENAME)
    artifact_path = output / str(manifest.iloc[0]["raw_ggl_path"])
    with np.load(artifact_path, allow_pickle=False) as artifact:
        payload = {name: artifact[name] for name in artifact.files}
    payload["training_preprocessing_version"] = np.asarray("old-version")
    preprocessing._atomic_write_npz(artifact_path, payload)
    calls = 0

    def tracking_geometry(smiles: str, *, config: GeometryConfig) -> GeometryResult:
        nonlocal calls
        calls += 1
        return _geometry_result(smiles, config)

    _run(config, output, resume=True, geometry_function=tracking_geometry)

    assert calls == 1
    with np.load(artifact_path, allow_pickle=False) as artifact:
        assert str(artifact["training_preprocessing_version"].item()) == (
            preprocessing.TRAINING_PREPROCESSING_VERSION
        )


def test_failed_row_is_retryable_on_resume(tmp_path: Path) -> None:
    rows = [_row("retry", "CCO", 1)]
    config = _config(tmp_path, rows)
    output = tmp_path / "output"

    def failing_geometry(*args: object, **kwargs: object) -> GeometryResult:
        raise GeometryError("embedding_failed", "transient")

    first = _run(config, output, geometry_function=failing_geometry)
    second = _run(config, output, resume=True)
    manifest = pd.read_csv(output / preprocessing.MANIFEST_FILENAME)

    assert first["failed_molecule_count"] == 1
    assert second["successful_molecule_count"] == 1
    assert manifest.iloc[0]["status"] == "success"
    assert manifest.iloc[0]["raw_ggl_path"]


def test_default_run_refuses_existing_directory_without_resume(tmp_path: Path) -> None:
    rows = [_row("one", "CCO", 1)]
    config = _config(tmp_path, rows)
    output = tmp_path / "output"
    output.mkdir()

    with pytest.raises(FileExistsError, match="--resume"):
        _run(config, output)


def test_summary_hashes_match_final_manifest_and_status(tmp_path: Path) -> None:
    rows = [_row("one", "CCO", 1)]
    config = _config(tmp_path, rows)
    output = tmp_path / "output"

    summary = _run(config, output)

    assert summary["feature_manifest_sha256"] == preprocessing._sha256_file(
        output / preprocessing.MANIFEST_FILENAME
    )
    assert summary["molecule_status_sha256"] == preprocessing._sha256_file(
        output / preprocessing.STATUS_FILENAME
    )
    assert summary["total_heavy_atom_count_among_successes"] == 3
    assert summary["finite_ggl_molecule_count"] == 1


def _run(
    config: ChempropExperimentConfig,
    output: Path,
    *,
    resume: bool = False,
    training_loader: preprocessing.TrainingLoader = preprocessing.load_bbb_development_split,
    geometry_function: preprocessing.GeometryFunction | None = None,
    ggl_function: preprocessing.GGLFunction | None = None,
) -> dict[str, Any]:
    return preprocessing.run_training_preprocessing(
        config.source_path,
        output_dir=output,
        resume=resume,
        config_loader=lambda _: config,
        training_loader=training_loader,
        geometry_function=geometry_function or _geometry_result,
        ggl_function=ggl_function or _ggl_result,
    )


def _row(
    label: str,
    smiles: str,
    target: int,
    *,
    molecule_id: str | None = None,
) -> dict[str, object]:
    canonical = _canonical(smiles)
    return {
        "molecule_id": molecule_id or label,
        "smiles": canonical,
        "canonical_smiles": canonical,
        "target": target,
        "split": "train",
    }


def _canonical(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _config(root: Path, rows: list[dict[str, object]]) -> ChempropExperimentConfig:
    prepared = root / "prepared"
    prepared.mkdir()
    pd.DataFrame(rows).to_csv(prepared / "train.csv", index=False)
    pd.DataFrame(
        [
            {
                "molecule_id": "validation-sentinel",
                "smiles": "N",
                "canonical_smiles": "N",
                "target": 1,
                "split": "validation",
            }
        ]
    ).to_csv(prepared / "valid.csv", index=False)
    (prepared / "locked.csv").write_text("must-not-be-read", encoding="utf-8")
    return ChempropExperimentConfig(
        source_path=root / "config.yaml",
        raw={},
        endpoint="bbb_martins",
        endpoint_id="bbb_martins",
        dataset="BBB_Martins",
        dataset_version="synthetic-training-preprocessing-fixture",
        task_type="binary_classification",
        primary_metric="auroc",
        prepared_root=prepared,
        split_manifest=root / "unused-manifest.json",
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
    assert coordinates.shape == (rows, 3)
    features = np.arange(rows * 6, dtype=np.float64).reshape(rows, 6)
    return GGLResult(
        features=features,
        feature_names=(
            "minimum",
            "maximum",
            "sum",
            "mean",
            "median",
            "population_standard_deviation",
        ),
        geometry_fingerprint=geometry_fingerprint,
        ggl_fingerprint=f"ggl-{geometry_fingerprint}",
        ggl_status="success",
        config=config,
        preprocessing_version="test",
        element_radius_mapping_version="test",
    )
