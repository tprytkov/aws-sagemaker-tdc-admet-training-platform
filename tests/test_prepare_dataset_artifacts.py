import json
import random
from pathlib import Path

import pandas as pd

from admet_platform.data.prepare import prepare_dataset_artifacts


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
REQUIRED_ARTIFACTS = {
    "train.csv",
    "valid.csv",
    "test.csv",
    "data_profile.json",
    "split_metadata.json",
    "rejected_rows.csv",
    "problematic_molecules.csv",
}
REQUIRED_ACCEPTED_COLUMNS = {"molecule_id", "smiles", "canonical_smiles", "target", "split"}
REJECTED_COLUMNS = ["molecule_id", "smiles", "target", "split", "rejection_reason"]


def test_binary_classification_artifact_preparation_writes_expected_outputs(
    tmp_path: Path,
) -> None:
    input_csv = _write_input_csv(
        tmp_path,
        [
            {"molecule_id": "mol_001", "smiles": "CCO", "target": 1, "split": "train"},
            {"molecule_id": "mol_002", "smiles": "CCN", "target": 0, "split": "train"},
            {"molecule_id": "mol_003", "smiles": "C1CCCCC1", "target": 1, "split": "validation"},
            {"molecule_id": "mol_004", "smiles": "c1ccccc1", "target": 0, "split": "test"},
        ],
    )
    output_dir = tmp_path / "prepared_bbb"

    prepare_dataset_artifacts(input_csv, CONFIG_DIR / "bbb_martins.yaml", output_dir)

    _assert_required_artifacts(output_dir)
    train = pd.read_csv(output_dir / "train.csv")
    valid = pd.read_csv(output_dir / "valid.csv")
    test = pd.read_csv(output_dir / "test.csv")
    accepted = pd.concat([train, valid, test], ignore_index=True)
    profile = _read_json(output_dir / "data_profile.json")
    split_metadata = _read_json(output_dir / "split_metadata.json")

    assert REQUIRED_ACCEPTED_COLUMNS <= set(accepted.columns)
    assert accepted["canonical_smiles"].notna().all()
    assert accepted["canonical_smiles"].astype(str).str.strip().ne("").all()
    assert profile["task_type"] == "binary_classification"
    assert profile["endpoint_id"] == "bbb_martins"
    assert profile["class_counts"] == {"0": 2, "1": 2}
    assert profile["n_train"] == len(train)
    assert profile["n_validation"] == len(valid)
    assert profile["n_test"] == len(test)
    assert split_metadata["split_counts"] == {
        "train": len(train),
        "validation": len(valid),
        "test": len(test),
    }


def test_regression_artifact_preparation_preserves_numeric_targets_and_metadata(
    tmp_path: Path,
) -> None:
    input_csv = _write_input_csv(
        tmp_path,
        [
            {"molecule_id": "mol_001", "smiles": "CCO", "target": -4.8, "split": "train"},
            {"molecule_id": "mol_002", "smiles": "CCN", "target": -5.1, "split": "train"},
            {"molecule_id": "mol_003", "smiles": "C1CCCCC1", "target": -4.4, "split": "validation"},
            {"molecule_id": "mol_004", "smiles": "c1ccccc1", "target": -3.9, "split": "test"},
        ],
    )
    output_dir = tmp_path / "prepared_caco2"

    prepare_dataset_artifacts(input_csv, CONFIG_DIR / "caco2_wang.yaml", output_dir)

    _assert_required_artifacts(output_dir)
    train = pd.read_csv(output_dir / "train.csv")
    valid = pd.read_csv(output_dir / "valid.csv")
    test = pd.read_csv(output_dir / "test.csv")
    targets = pd.concat([train, valid, test], ignore_index=True)["target"].tolist()
    profile = _read_json(output_dir / "data_profile.json")
    split_metadata = _read_json(output_dir / "split_metadata.json")

    assert targets == [-4.8, -5.1, -4.4, -3.9]
    assert profile["task_type"] == "regression"
    assert profile["endpoint_id"] == "caco2_wang"
    assert profile["tdc_name"] == "Caco2_Wang"
    assert profile["target_min"] == -5.1
    assert profile["target_max"] == -3.9
    assert split_metadata["task_type"] == "regression"
    assert split_metadata["endpoint_id"] == "caco2_wang"
    assert split_metadata["tdc_name"] == "Caco2_Wang"


def test_invalid_smiles_are_rejected_and_excluded_from_accepted_splits(
    tmp_path: Path,
) -> None:
    input_csv = _write_input_csv(
        tmp_path,
        [
            {"molecule_id": "mol_001", "smiles": "CCO", "target": 1, "split": "train"},
            {"molecule_id": "mol_bad", "smiles": "not_a_smiles", "target": 0, "split": "train"},
            {"molecule_id": "mol_002", "smiles": "CCN", "target": 1, "split": "validation"},
            {"molecule_id": "mol_003", "smiles": "c1ccccc1", "target": 0, "split": "test"},
        ],
    )
    output_dir = tmp_path / "prepared_with_rejections"

    prepare_dataset_artifacts(input_csv, CONFIG_DIR / "bbb_martins.yaml", output_dir)

    accepted = pd.concat(
        [
            pd.read_csv(output_dir / "train.csv"),
            pd.read_csv(output_dir / "valid.csv"),
            pd.read_csv(output_dir / "test.csv"),
        ],
        ignore_index=True,
    )
    rejected = pd.read_csv(output_dir / "rejected_rows.csv")
    profile = _read_json(output_dir / "data_profile.json")

    assert "mol_bad" not in set(accepted["molecule_id"])
    assert rejected["molecule_id"].tolist() == ["mol_bad"]
    assert rejected["rejection_reason"].tolist() == ["invalid_smiles"]
    assert profile["n_rows"] == 4
    assert profile["n_accepted_rows"] == 3
    assert profile["n_rejected_rows"] == 1


def test_rejected_rows_file_has_headers_and_zero_data_rows_when_no_invalid_rows(
    tmp_path: Path,
) -> None:
    input_csv = _write_input_csv(
        tmp_path,
        [
            {"molecule_id": "mol_001", "smiles": "CCO", "target": 1, "split": "train"},
            {"molecule_id": "mol_002", "smiles": "CCN", "target": 0, "split": "validation"},
            {"molecule_id": "mol_003", "smiles": "c1ccccc1", "target": 1, "split": "test"},
        ],
    )
    output_dir = tmp_path / "prepared_no_rejections"

    prepare_dataset_artifacts(input_csv, CONFIG_DIR / "bbb_martins.yaml", output_dir)

    rejected_path = output_dir / "rejected_rows.csv"
    rejected = pd.read_csv(rejected_path)

    assert rejected_path.exists()
    assert list(rejected.columns) == REJECTED_COLUMNS
    assert rejected.empty


def test_artifact_preparation_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    input_csv = _write_input_csv(
        tmp_path,
        [
            {"molecule_id": "mol_001", "smiles": "CCO", "target": 1, "split": "train"},
            {"molecule_id": "mol_002", "smiles": "CCN", "target": 0, "split": "train"},
            {"molecule_id": "mol_003", "smiles": "C1CCCCC1", "target": 1, "split": "validation"},
            {"molecule_id": "mol_004", "smiles": "c1ccccc1", "target": 0, "split": "test"},
        ],
    )
    first_output_dir = tmp_path / "prepared_first"
    second_output_dir = tmp_path / "prepared_second"

    random.seed(42)
    prepare_dataset_artifacts(input_csv, CONFIG_DIR / "bbb_martins.yaml", first_output_dir)
    random.seed(42)
    prepare_dataset_artifacts(input_csv, CONFIG_DIR / "bbb_martins.yaml", second_output_dir)

    for artifact_name in ("train.csv", "valid.csv", "test.csv", "split_metadata.json"):
        assert (first_output_dir / artifact_name).read_text(encoding="utf-8") == (
            second_output_dir / artifact_name
        ).read_text(encoding="utf-8")


def _write_input_csv(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    input_csv = tmp_path / "input.csv"
    pd.DataFrame(rows).to_csv(input_csv, index=False)
    return input_csv


def _assert_required_artifacts(output_dir: Path) -> None:
    assert {path.name for path in output_dir.iterdir()} == REQUIRED_ARTIFACTS


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
