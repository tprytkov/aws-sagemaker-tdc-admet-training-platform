import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from admet_platform.features import (
    DESCRIPTOR_NAMES,
    DEFAULT_MORGAN_BITS,
    DEFAULT_MORGAN_RADIUS,
    FeatureConfig,
    featurize_csv,
    featurize_dataframe,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_descriptor_names_are_explicit_and_deterministic() -> None:
    assert DESCRIPTOR_NAMES == [
        "molecular_weight",
        "logp",
        "tpsa",
        "h_bond_donors",
        "h_bond_acceptors",
        "rotatable_bonds",
        "ring_count",
        "aromatic_ring_count",
        "heavy_atom_count",
        "fraction_csp3",
    ]

    features_df, rejected_df, metadata = featurize_dataframe(
        _classification_df(),
        FeatureConfig(feature_type="descriptors"),
    )

    assert list(features_df.columns[-len(DESCRIPTOR_NAMES) :]) == DESCRIPTOR_NAMES
    assert rejected_df.empty
    assert metadata["descriptor_names"] == DESCRIPTOR_NAMES


def test_morgan_fingerprint_dimensions_and_column_names() -> None:
    features_df, rejected_df, metadata = featurize_dataframe(
        _classification_df(),
        FeatureConfig(feature_type="morgan", morgan_radius=2, morgan_bits=8),
    )

    morgan_columns = [column for column in features_df.columns if column.startswith("morgan_")]
    assert morgan_columns == [f"morgan_{index:04d}" for index in range(8)]
    assert len(morgan_columns) == 8
    assert metadata["morgan_radius"] == 2
    assert metadata["morgan_bits"] == 8
    assert metadata["n_features"] == 8
    assert rejected_df.empty


def test_morgan_fingerprint_output_is_deterministic() -> None:
    config = FeatureConfig(feature_type="morgan", morgan_radius=2, morgan_bits=32)

    first_features, first_rejected, first_metadata = featurize_dataframe(_classification_df(), config)
    second_features, second_rejected, second_metadata = featurize_dataframe(_classification_df(), config)

    pd.testing.assert_frame_equal(first_features, second_features)
    pd.testing.assert_frame_equal(first_rejected, second_rejected)
    assert first_metadata == second_metadata


def test_identifier_target_endpoint_and_split_columns_are_preserved() -> None:
    features_df, _, _ = featurize_dataframe(
        _classification_df(),
        FeatureConfig(feature_type="descriptors"),
    )

    assert features_df["molecule_id"].tolist() == ["mol_001", "mol_002", "mol_003"]
    assert features_df["target"].tolist() == [1, 0, 1]
    assert features_df["endpoint_id"].tolist() == ["bbb_martins", "bbb_martins", "bbb_martins"]
    assert features_df["split"].tolist() == ["train", "validation", "test"]


def test_classification_input_is_supported() -> None:
    features_df, rejected_df, metadata = featurize_dataframe(
        _classification_df(),
        FeatureConfig(feature_type="descriptors"),
    )

    assert len(features_df) == 3
    assert features_df["target"].tolist() == [1, 0, 1]
    assert rejected_df.empty
    assert metadata["accepted_row_count"] == 3


def test_regression_input_is_supported_and_targets_are_preserved() -> None:
    features_df, rejected_df, metadata = featurize_dataframe(
        _regression_df(),
        FeatureConfig(feature_type="descriptors"),
    )

    assert features_df["target"].tolist() == [-4.8, -5.1, -3.9]
    assert rejected_df.empty
    assert metadata["accepted_row_count"] == 3


def test_invalid_smiles_are_rejected_without_zero_vector_replacement() -> None:
    df = pd.DataFrame(
        [
            {
                "molecule_id": "mol_001",
                "canonical_smiles": "CCO",
                "target": 1,
                "split": "train",
            },
            {
                "molecule_id": "mol_bad",
                "canonical_smiles": "not_a_smiles",
                "target": 0,
                "split": "train",
            },
        ]
    )

    features_df, rejected_df, metadata = featurize_dataframe(
        df,
        FeatureConfig(feature_type="morgan", morgan_bits=16),
    )

    assert features_df["molecule_id"].tolist() == ["mol_001"]
    assert rejected_df["canonical_smiles"].tolist() == ["not_a_smiles"]
    assert rejected_df["rejection_reason"].tolist() == ["invalid_or_missing_canonical_smiles"]
    assert metadata["accepted_row_count"] == 1
    assert metadata["rejected_row_count"] == 1


def test_empty_input_raises_value_error() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        featurize_dataframe(
            pd.DataFrame(columns=["molecule_id", "canonical_smiles", "target", "split"]),
            FeatureConfig(feature_type="descriptors"),
        )


def test_feature_configuration_is_serializable() -> None:
    descriptor_config = FeatureConfig(feature_type="descriptors").to_dict()
    morgan_config = FeatureConfig(feature_type="morgan", morgan_radius=3, morgan_bits=16).to_dict()

    json.dumps(descriptor_config)
    json.dumps(morgan_config)
    assert descriptor_config["feature_type"] == "descriptors"
    assert descriptor_config["descriptor_names"] == DESCRIPTOR_NAMES
    assert morgan_config["feature_type"] == "morgan"
    assert morgan_config["morgan_radius"] == 3
    assert morgan_config["morgan_bits"] == 16
    assert morgan_config["feature_columns"] == [f"morgan_{index:04d}" for index in range(16)]


def test_featurize_dataset_cli_smoke_execution(tmp_path: Path) -> None:
    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "features.csv"
    metadata_json = tmp_path / "metadata.json"
    rejected_csv = tmp_path / "rejected.csv"
    _classification_df().to_csv(input_csv, index=False)

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "featurize_dataset.py"),
            "--input-csv",
            str(input_csv),
            "--output-csv",
            str(output_csv),
            "--feature-type",
            "morgan",
            "--metadata-json",
            str(metadata_json),
            "--rejected-csv",
            str(rejected_csv),
            "--morgan-bits",
            "16",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    features_df = pd.read_csv(output_csv)
    metadata = json.loads(metadata_json.read_text(encoding="utf-8"))
    rejected_df = pd.read_csv(rejected_csv)
    assert output_csv.exists()
    assert metadata_json.exists()
    assert rejected_csv.exists()
    assert len([column for column in features_df.columns if column.startswith("morgan_")]) == 16
    assert metadata["feature_type"] == "morgan"
    assert rejected_df.empty
    assert "Wrote feature CSV" in result.stdout


def test_featurize_csv_writes_metadata(tmp_path: Path) -> None:
    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "descriptors.csv"
    metadata_json = tmp_path / "descriptors_metadata.json"
    _classification_df().to_csv(input_csv, index=False)

    metadata = featurize_csv(
        input_csv=input_csv,
        output_csv=output_csv,
        metadata_json=metadata_json,
        config=FeatureConfig(feature_type="descriptors"),
    )

    written_metadata = json.loads(metadata_json.read_text(encoding="utf-8"))
    assert output_csv.exists()
    assert written_metadata == metadata


def _classification_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "molecule_id": "mol_001",
                "canonical_smiles": "CCO",
                "target": 1,
                "endpoint_id": "bbb_martins",
                "split": "train",
            },
            {
                "molecule_id": "mol_002",
                "canonical_smiles": "CCN",
                "target": 0,
                "endpoint_id": "bbb_martins",
                "split": "validation",
            },
            {
                "molecule_id": "mol_003",
                "canonical_smiles": "c1ccccc1",
                "target": 1,
                "endpoint_id": "bbb_martins",
                "split": "test",
            },
        ]
    )


def _regression_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "molecule_id": "mol_001",
                "canonical_smiles": "CCO",
                "target": -4.8,
                "endpoint_id": "caco2_wang",
                "split": "train",
            },
            {
                "molecule_id": "mol_002",
                "canonical_smiles": "CCN",
                "target": -5.1,
                "endpoint_id": "caco2_wang",
                "split": "validation",
            },
            {
                "molecule_id": "mol_003",
                "canonical_smiles": "c1ccccc1",
                "target": -3.9,
                "endpoint_id": "caco2_wang",
                "split": "test",
            },
        ]
    )
