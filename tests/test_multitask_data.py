from pathlib import Path

import pandas as pd
import pytest

from admet_platform.data.multitask import load_endpoint_datasets, load_multitask_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "multitask_classification.yaml"


def test_repository_multitask_config_defines_three_binary_tasks() -> None:
    config = load_multitask_config(CONFIG_PATH)

    assert list(config.tasks) == ["bbb_martins", "herg_karim", "ames"]
    assert all(task.task_type == "binary_classification" for task in config.tasks.values())
    assert config.tasks["bbb_martins"].tdc_name == "BBB_Martins"
    assert config.tasks["herg_karim"].tdc_name == "herg"
    assert config.tasks["ames"].tdc_name == "AMES"
    assert config.split_track == "coordinated_multitask"


def test_referenced_endpoint_mismatch_is_rejected(tmp_path: Path) -> None:
    endpoint = tmp_path / "endpoint.yaml"
    endpoint.write_text((PROJECT_ROOT / "configs" / "bbb_martins.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    config_path = _write_config(tmp_path, endpoint_name=endpoint.name)
    text = config_path.read_text(encoding="utf-8").replace("tdc_name: BBB_Martins", "tdc_name: WRONG", 1)
    config_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="does not match referenced endpoint config"):
        load_multitask_config(config_path)


def test_non_binary_task_is_rejected(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    text = config_path.read_text(encoding="utf-8").replace(
        "task_type: binary_classification", "task_type: regression", 1
    )
    config_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="binary_classification"):
        load_multitask_config(config_path)


def test_separate_endpoint_splits_load_and_validate(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    root = tmp_path / "prepared"
    for endpoint in ("bbb_martins", "herg_karim", "ames"):
        _write_splits(root / endpoint)

    config = load_multitask_config(config_path)
    datasets = load_endpoint_datasets(config)

    assert set(datasets) == {"bbb_martins", "herg_karim", "ames"}
    assert len(datasets["bbb_martins"].train) == 2
    assert len(datasets["bbb_martins"].validation) == 1
    assert len(datasets["bbb_martins"].test) == 1


def test_missing_split_file_fails_clearly(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    root = tmp_path / "prepared"
    for endpoint in ("bbb_martins", "herg_karim", "ames"):
        _write_splits(root / endpoint)
    (root / "ames" / "test.csv").unlink()

    with pytest.raises(FileNotFoundError, match="Missing prepared test CSV"):
        load_endpoint_datasets(load_multitask_config(config_path))


def _write_config(tmp_path: Path, endpoint_name: str | None = None) -> Path:
    endpoint_ref = f"    endpoint_config: {endpoint_name}\n" if endpoint_name else ""
    path = tmp_path / "multitask.yaml"
    path.write_text(
        f'''schema_version: "1.0.0"
run_name: test
split_track: coordinated_multitask
prepared_root: prepared
tasks:
  bbb_martins:
{endpoint_ref}    endpoint_id: bbb_martins
    tdc_name: BBB_Martins
    task_group: ADME
    task_type: binary_classification
    primary_metric: roc_auc
  herg_karim:
    endpoint_id: herg_karim
    tdc_name: herg
    task_group: Tox
    task_type: binary_classification
    primary_metric: roc_auc
  ames:
    endpoint_id: ames
    tdc_name: AMES
    task_group: Tox
    task_type: binary_classification
    primary_metric: roc_auc
split_files:
  train: train.csv
  validation: valid.csv
  test: test.csv
audit:
  enforce_exact_smiles_exclusion: true
  enforce_scaffold_exclusion: true
  fail_on_invalid_molecules: true
  fail_on_conflicting_labels: true
  fail_on_duplicates: true
''',
        encoding="utf-8",
    )
    return path


def _write_splits(root: Path) -> None:
    root.mkdir(parents=True)
    for split, file_name, smiles in (
        ("train", "train.csv", ["CCO", "CCN"]),
        ("validation", "valid.csv", ["c1ccccc1"]),
        ("test", "test.csv", ["C1CCCCC1"]),
    ):
        pd.DataFrame(
            {
                "molecule_id": [f"{split}-{index}" for index in range(len(smiles))],
                "smiles": smiles,
                "canonical_smiles": smiles,
                "target": [index % 2 for index in range(len(smiles))],
                "split": [split] * len(smiles),
            }
        ).to_csv(root / file_name, index=False)

