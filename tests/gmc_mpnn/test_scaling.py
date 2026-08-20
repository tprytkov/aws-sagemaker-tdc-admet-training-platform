from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import sklearn
from rdkit import rdBase
from sklearn.preprocessing import StandardScaler

from admet_platform.gmc_mpnn import scaling
from admet_platform.gmc_mpnn.geometry import GEOMETRY_PREPROCESSING_VERSION
from admet_platform.gmc_mpnn.ggl import GGL_FEATURE_NAMES, GGL_PREPROCESSING_VERSION
from admet_platform.gmc_mpnn.standardization import GMC_STANDARDIZATION_VERSION


def test_fit_matches_sklearn_and_uses_manifest_ordered_successful_atoms(
    tmp_path: Path,
) -> None:
    first = np.asarray([[10, 1, 8, 3, 7, 2], [20, 2, 7, 4, 6, 3]], dtype=np.float64)
    second = np.asarray([[30, 3, 6, 5, 5, 4]], dtype=np.float64)
    source = _preprocessing_fixture(
        tmp_path / "preprocessing",
        [
            _entry("first", "success", first),
            _entry("excluded", "excluded"),
            _entry("failed", "failed"),
            _entry("second", "success", second),
        ],
    )

    pool = scaling.validate_and_pool_training_artifacts(source)
    expected_pool = np.concatenate((first, second), axis=0)
    np.testing.assert_array_equal(pool.features, expected_pool)

    output = tmp_path / "scaler"
    summary = scaling.fit_training_ggl_scaler(source, output, git_commit="synthetic-commit")
    expected = StandardScaler(with_mean=True, with_std=True).fit(expected_pool)

    np.testing.assert_array_equal(summary["mean_"], expected.mean_)
    np.testing.assert_array_equal(summary["var_"], expected.var_)
    np.testing.assert_array_equal(summary["scale_"], expected.scale_)
    assert summary["training_molecule_count"] == 2
    assert summary["training_atom_count"] == 3
    assert summary["n_features_in_"] == 6
    assert summary["n_samples_seen_"] == 3
    assert summary["validation_artifact_accessed"] is False
    assert summary["test_artifact_accessed"] is False
    assert summary["numpy_version"] == np.__version__
    assert summary["scikit_learn_version"] == sklearn.__version__


def test_zero_variance_serialization_load_and_transform_round_trip(tmp_path: Path) -> None:
    values = np.asarray(
        [[1, 5, 2, 8, 3, 13], [2, 5, 4, 8, 6, 13], [3, 5, 8, 8, 9, 13]],
        dtype=np.float64,
    )
    source = _preprocessing_fixture(tmp_path / "preprocessing", [_entry("only", "success", values)])
    output = tmp_path / "scaler"
    scaling.fit_training_ggl_scaler(source, output, git_commit="synthetic-commit")

    frozen = scaling.load_frozen_ggl_scaler(output)
    expected_scaler = StandardScaler(with_mean=True, with_std=True).fit(values)
    expected = expected_scaler.transform(values)
    observed = scaling.transform_frozen_ggl(values.copy(), frozen)

    assert frozen.scaler_version == scaling.GGL_SCALER_VERSION
    assert frozen.feature_order == GGL_FEATURE_NAMES
    assert frozen.scale_[[1, 3, 5]].tolist() == [1.0, 1.0, 1.0]
    assert observed.dtype == np.float64
    np.testing.assert_array_equal(observed, expected)
    with np.load(output / scaling.SCALER_NPZ_FILENAME, allow_pickle=False) as artifact:
        assert set(artifact.files) == scaling.SCALER_NPZ_KEYS
        np.testing.assert_array_equal(artifact["mean_"], expected_scaler.mean_)
        np.testing.assert_array_equal(artifact["var_"], expected_scaler.var_)
        np.testing.assert_array_equal(artifact["scale_"], expected_scaler.scale_)


def test_loading_and_transforming_never_call_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = np.arange(18, dtype=np.float64).reshape(3, 6)
    source = _preprocessing_fixture(tmp_path / "preprocessing", [_entry("only", "success", values)])
    output = tmp_path / "scaler"
    scaling.fit_training_ggl_scaler(source, output, git_commit="synthetic-commit")

    def forbidden_fit(*args: object, **kwargs: object) -> None:
        raise AssertionError("Frozen scaler loading/transformation must not fit.")

    monkeypatch.setattr(scaling.StandardScaler, "fit", forbidden_fit)

    frozen = scaling.load_frozen_ggl_scaler(output)
    transformed = scaling.transform_frozen_ggl(values, frozen)

    assert transformed.shape == values.shape
    assert np.isfinite(transformed).all()


@pytest.mark.parametrize(
    "values",
    (
        np.asarray([[1, 2, 3, 4, 5, np.nan]], dtype=np.float64),
        np.asarray([[1, 2, 3, 4, 5, np.inf]], dtype=np.float64),
    ),
)
def test_nonfinite_raw_artifacts_are_rejected(tmp_path: Path, values: np.ndarray) -> None:
    source = _preprocessing_fixture(tmp_path / "preprocessing", [_entry("bad", "success", values)])

    with pytest.raises(scaling.GGLScalingError, match="NaN or Inf"):
        scaling.validate_and_pool_training_artifacts(source)


@pytest.mark.parametrize(
    "values, message",
    (
        (np.ones((2, 6), dtype=np.float32), "float64"),
        (np.ones((2, 5), dtype=np.float64), "shape"),
        (np.asarray([[1, 2, 3, 4, 5, np.nan]], dtype=np.float64), "NaN or Inf"),
    ),
)
def test_transform_rejects_wrong_dtype_shape_and_nonfinite_values(
    tmp_path: Path,
    values: np.ndarray,
    message: str,
) -> None:
    training = np.arange(18, dtype=np.float64).reshape(3, 6)
    source = _preprocessing_fixture(
        tmp_path / "preprocessing", [_entry("only", "success", training)]
    )
    output = tmp_path / "scaler"
    scaling.fit_training_ggl_scaler(source, output, git_commit="synthetic-commit")
    frozen = scaling.load_frozen_ggl_scaler(output)

    with pytest.raises(scaling.GGLScalingError, match=message):
        frozen.transform(values)


@pytest.mark.parametrize(
    "field, value, message",
    (
        ("scaler_version", "future-scaler", "version"),
        ("feature_order", list(reversed(GGL_FEATURE_NAMES)), "feature order"),
    ),
)
def test_loader_rejects_wrong_scaler_version_or_feature_order(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    source = _preprocessing_fixture(
        tmp_path / "preprocessing",
        [_entry("only", "success", np.arange(12, dtype=np.float64).reshape(2, 6))],
    )
    output = tmp_path / "scaler"
    scaling.fit_training_ggl_scaler(source, output, git_commit="synthetic-commit")
    scaler_path = output / scaling.SCALER_JSON_FILENAME
    summary_path = output / scaling.FIT_SUMMARY_FILENAME
    payload = json.loads(scaler_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    payload[field] = value
    payload["portable_scaler_sha256"] = scaling._portable_scaler_sha256(payload)
    _write_json(scaler_path, payload)
    summary.update(payload)
    summary["scaler_json_sha256"] = _sha256(scaler_path)
    _write_json(summary_path, summary)

    with pytest.raises(scaling.GGLScalingError, match=message):
        scaling.load_frozen_ggl_scaler(output)


def test_bad_manifest_hash_is_rejected_before_artifacts_are_opened(tmp_path: Path) -> None:
    source = _preprocessing_fixture(
        tmp_path / "preprocessing",
        [_entry("only", "success", np.arange(12, dtype=np.float64).reshape(2, 6))],
    )
    summary_path = source / scaling.PREPROCESSING_SUMMARY_FILENAME
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["feature_manifest_sha256"] = "0" * 64
    _write_json(summary_path, summary)

    with pytest.raises(scaling.GGLScalingError, match="feature_manifest.csv SHA-256"):
        scaling.validate_and_pool_training_artifacts(source)


def test_bad_raw_artifact_checksum_is_rejected(tmp_path: Path) -> None:
    source = _preprocessing_fixture(
        tmp_path / "preprocessing",
        [_entry("only", "success", np.arange(12, dtype=np.float64).reshape(2, 6))],
    )
    artifact_path = next((source / scaling.RAW_GGL_DIRECTORY).glob("*.npz"))
    with np.load(artifact_path, allow_pickle=False) as artifact:
        payload = {key: artifact[key] for key in artifact.files}
    payload["artifact_content_sha256"] = np.asarray("0" * 64)
    np.savez(artifact_path, **payload)

    with pytest.raises(scaling.GGLScalingError, match="checksum"):
        scaling.validate_and_pool_training_artifacts(source)


def test_manifest_and_atom_order_determine_ordered_input_hash(tmp_path: Path) -> None:
    first = np.arange(12, dtype=np.float64).reshape(2, 6)
    second = np.arange(12, 24, dtype=np.float64).reshape(2, 6)
    source_a = _preprocessing_fixture(
        tmp_path / "a", [_entry("first", "success", first), _entry("second", "success", second)]
    )
    source_b = _preprocessing_fixture(
        tmp_path / "b", [_entry("second", "success", second), _entry("first", "success", first)]
    )

    pool_a = scaling.validate_and_pool_training_artifacts(source_a)
    pool_b = scaling.validate_and_pool_training_artifacts(source_b)

    np.testing.assert_array_equal(pool_a.features, np.concatenate((first, second), axis=0))
    np.testing.assert_array_equal(pool_b.features, np.concatenate((second, first), axis=0))
    assert pool_a.ordered_input_artifact_sha256 != pool_b.ordered_input_artifact_sha256


def test_nonascending_atom_indices_are_rejected(tmp_path: Path) -> None:
    source = _preprocessing_fixture(
        tmp_path / "preprocessing",
        [
            _entry(
                "bad-order",
                "success",
                np.arange(12, dtype=np.float64).reshape(2, 6),
                rdkit_indices=np.asarray([1, 0], dtype=np.int64),
            )
        ],
    )

    with pytest.raises(scaling.GGLScalingError, match="ascending RDKit atom order"):
        scaling.validate_and_pool_training_artifacts(source)


def test_scaler_outputs_and_hashes_are_deterministic(tmp_path: Path) -> None:
    source = _preprocessing_fixture(
        tmp_path / "preprocessing",
        [_entry("only", "success", np.arange(18, dtype=np.float64).reshape(3, 6))],
    )
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_summary = scaling.fit_training_ggl_scaler(source, first, git_commit="synthetic-commit")
    second_summary = scaling.fit_training_ggl_scaler(source, second, git_commit="synthetic-commit")

    assert first_summary == second_summary
    for filename in (
        scaling.SCALER_JSON_FILENAME,
        scaling.SCALER_NPZ_FILENAME,
        scaling.FIT_SUMMARY_FILENAME,
    ):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


@pytest.mark.parametrize("flag", ("validation_artifact_accessed", "test_artifact_accessed"))
def test_nontraining_access_flags_must_remain_false(tmp_path: Path, flag: str) -> None:
    source = _preprocessing_fixture(
        tmp_path / "preprocessing",
        [_entry("only", "success", np.arange(12, dtype=np.float64).reshape(2, 6))],
        access_override={flag: True},
    )

    with pytest.raises(scaling.GGLScalingError, match="incompatible"):
        scaling.validate_and_pool_training_artifacts(source)


def test_scaler_output_directory_must_be_new(tmp_path: Path) -> None:
    source = _preprocessing_fixture(
        tmp_path / "preprocessing",
        [_entry("only", "success", np.arange(12, dtype=np.float64).reshape(2, 6))],
    )
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        scaling.fit_training_ggl_scaler(source, output, git_commit="synthetic-commit")


def _entry(
    label: str,
    status: str,
    features: np.ndarray | None = None,
    *,
    rdkit_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    return {
        "label": label,
        "status": status,
        "features": features,
        "rdkit_indices": rdkit_indices,
    }


def _preprocessing_fixture(
    root: Path,
    entries: list[dict[str, Any]],
    *,
    access_override: dict[str, bool] | None = None,
) -> Path:
    raw_directory = root / scaling.RAW_GGL_DIRECTORY
    raw_directory.mkdir(parents=True)
    manifest_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    atom_count = 0
    for index, entry in enumerate(entries):
        label = str(entry["label"])
        status = str(entry["status"])
        record_key = hashlib.sha256(f"{index}:{label}".encode("utf-8")).hexdigest()
        raw_path = ""
        count: int | str = ""
        columns: int | str = ""
        geometry_smiles = "CCO"
        geometry_fingerprint = ""
        ggl_fingerprint = ""
        optimization_method = ""
        if status == "success":
            features = np.asarray(entry["features"])
            count = int(features.shape[0])
            columns = int(features.shape[1])
            atom_count += count
            raw_path = f"{scaling.RAW_GGL_DIRECTORY}/{record_key}.npz"
            geometry_fingerprint = f"geometry-{record_key}"
            ggl_fingerprint = f"ggl-{record_key}"
            optimization_method = "MMFF94s"
            indices = entry["rdkit_indices"]
            if indices is None:
                indices = np.arange(count, dtype=np.int64)
            payload = {
                "raw_ggl_features": features,
                "heavy_atom_atomic_numbers": np.full(count, 6, dtype=np.int64),
                "heavy_atom_rdkit_indices": indices,
                "ggl_feature_names": np.asarray(GGL_FEATURE_NAMES),
                "record_key": np.asarray(record_key),
                "training_preprocessing_version": np.asarray(
                    scaling.TRAINING_PREPROCESSING_VERSION
                ),
                "standardization_version": np.asarray(GMC_STANDARDIZATION_VERSION),
                "geometry_preprocessing_version": np.asarray(GEOMETRY_PREPROCESSING_VERSION),
                "ggl_preprocessing_version": np.asarray(GGL_PREPROCESSING_VERSION),
                "geometry_smiles": np.asarray(geometry_smiles),
                "geometry_fingerprint": np.asarray(geometry_fingerprint),
                "ggl_fingerprint": np.asarray(ggl_fingerprint),
                "optimization_method": np.asarray(optimization_method),
                "rdkit_version": np.asarray(rdBase.rdkitVersion),
            }
            payload["artifact_content_sha256"] = np.asarray(scaling._npz_content_sha256(payload))
            np.savez(root / raw_path, **payload)
        row = {
            "record_key": record_key,
            "molecule_id": label,
            "split": "train",
            "standardization_version": GMC_STANDARDIZATION_VERSION,
            "geometry_smiles": geometry_smiles,
            "status": status,
            "raw_ggl_path": raw_path,
            "heavy_atom_count": count,
            "raw_ggl_rows": count,
            "raw_ggl_columns": columns,
            "geometry_fingerprint": geometry_fingerprint,
            "ggl_fingerprint": ggl_fingerprint,
            "optimization_method": optimization_method,
            "rdkit_version": rdBase.rdkitVersion if status == "success" else "",
        }
        manifest_rows.append(row)
        status_rows.append({key: row[key] for key in scaling.STATUS_MATCH_COLUMNS})

    manifest_path = root / scaling.MANIFEST_FILENAME
    status_path = root / scaling.STATUS_FILENAME
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False, lineterminator="\n")
    pd.DataFrame(status_rows).to_csv(status_path, index=False, lineterminator="\n")
    statuses = [str(entry["status"]) for entry in entries]
    summary = {
        "dataset": "BBB_Martins",
        "dataset_version": "synthetic-scaler-fixture",
        "loaded_split": "train",
        "source_row_count": len(entries),
        "successful_molecule_count": statuses.count("success"),
        "failed_molecule_count": statuses.count("failed"),
        "policy_excluded_molecule_count": statuses.count("excluded"),
        "finite_ggl_molecule_count": statuses.count("success"),
        "total_heavy_atom_count_among_successes": atom_count,
        "training_preprocessing_version": scaling.TRAINING_PREPROCESSING_VERSION,
        "standardization_version": GMC_STANDARDIZATION_VERSION,
        "geometry_preprocessing_version": GEOMETRY_PREPROCESSING_VERSION,
        "ggl_preprocessing_version": GGL_PREPROCESSING_VERSION,
        "ggl_feature_order": list(GGL_FEATURE_NAMES),
        "ggl_scaled": False,
        "rdkit_version": rdBase.rdkitVersion,
        "feature_manifest_sha256": _sha256(manifest_path),
        "molecule_status_sha256": _sha256(status_path),
        "validation_artifact_accessed": False,
        "test_artifact_accessed": False,
    }
    if access_override:
        summary.update(access_override)
    _write_json(root / scaling.PREPROCESSING_SUMMARY_FILENAME, summary)
    return root


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
