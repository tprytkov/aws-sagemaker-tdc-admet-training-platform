import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from admet_platform.data.multitask import load_endpoint_datasets, load_multitask_config
from admet_platform.data.multitask_audit import (
    MultiTaskAuditError,
    audit_multitask_splits,
    require_leakage_safe,
    write_audit_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "audit_multitask_splits.py"


def test_audit_detects_exact_and_cross_task_train_test_leakage(tmp_path: Path) -> None:
    config_path, root = _fixture(tmp_path)
    _write_split(root / "bbb_martins" / "test.csv", "test", [("bbb-test", "CCO", 1)])
    _write_split(root / "herg_karim" / "train.csv", "train", [("herg-train", "OCC", 0)])

    result = _audit(config_path)

    exact = result.tables["exact_overlaps"]
    assert ((exact["cross_task"]) & (exact["train_vs_heldout"])).any()
    assert "exact_smiles_train_heldout_overlap" in set(result.tables["violations"]["violation_type"])
    with pytest.raises(MultiTaskAuditError, match="blocking violation"):
        require_leakage_safe(result)


def test_audit_detects_murcko_scaffold_overlap(tmp_path: Path) -> None:
    config_path, root = _fixture(tmp_path)
    _write_split(root / "bbb_martins" / "test.csv", "test", [("bbb-test", "Cc1ccccc1", 1)])
    _write_split(root / "ames" / "train.csv", "train", [("ames-train", "Oc1ccccc1", 0)])

    result = _audit(config_path)

    scaffolds = result.tables["scaffold_overlaps"]
    assert ((scaffolds["cross_task"]) & (scaffolds["train_vs_heldout"])).any()
    assert "scaffold_train_heldout_overlap" in set(result.tables["violations"]["violation_type"])


def test_audit_detects_duplicates_conflicts_and_invalid_molecules(tmp_path: Path) -> None:
    config_path, root = _fixture(tmp_path)
    _write_split(
        root / "bbb_martins" / "train.csv",
        "train",
        [("one", "CCO", 0), ("two", "OCC", 1), ("bad", "not-smiles", 0)],
    )

    result = _audit(config_path)

    assert len(result.tables["duplicates"]) == 1
    assert len(result.tables["conflicts"]) == 1
    assert len(result.tables["invalid"]) == 1
    assert {"duplicate_molecules", "conflicting_labels", "invalid_molecules"}.issubset(
        set(result.tables["violations"]["violation_type"])
    )


def test_clean_audit_passes_and_writes_all_machine_readable_outputs(tmp_path: Path) -> None:
    config_path, _ = _fixture(tmp_path)
    result = _audit(config_path)
    output = tmp_path / "audit"

    paths = write_audit_outputs(result, output)
    require_leakage_safe(result)

    assert result.summary["status"] == "passed"
    assert result.summary["leakage_safe_for_training"] is True
    assert all(Path(path).is_file() for path in paths.values())
    summary = json.loads((output / "audit_summary.json").read_text(encoding="utf-8"))
    assert summary["split_track"] == "coordinated_multitask"
    assert (output / "exact_smiles_overlaps.csv").read_text(encoding="utf-8").startswith("canonical_smiles,")


def test_cli_writes_artifacts_then_returns_nonzero_for_blocking_leakage(tmp_path: Path) -> None:
    config_path, root = _fixture(tmp_path)
    _write_split(root / "bbb_martins" / "test.csv", "test", [("bbb-test", "CCO", 1)])
    output = tmp_path / "audit"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(config_path),
            "--output-dir",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "leakage audit failed" in completed.stderr.lower()
    assert (output / "audit_summary.json").is_file()
    assert (output / "leakage_violations.csv").is_file()


def _audit(config_path: Path):
    config = load_multitask_config(config_path)
    return audit_multitask_splits(config, load_endpoint_datasets(config))


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "prepared"
    task_smiles = {
        "bbb_martins": {"train": "CCO", "validation": "c1ccccc1", "test": "C1CCCCC1"},
        "herg_karim": {"train": "CCN", "validation": "c1ccncc1", "test": "C1CCNCC1"},
        "ames": {"train": "CCC", "validation": "c1ccoc1", "test": "C1CCOC1"},
    }
    for task, splits in task_smiles.items():
        for split, smiles in splits.items():
            file_name = "valid.csv" if split == "validation" else f"{split}.csv"
            _write_split(root / task / file_name, split, [(f"{task}-{split}", smiles, 1)])
    config_path = tmp_path / "multitask.yaml"
    config_path.write_text(
        '''schema_version: "1.0.0"
run_name: audit-test
split_track: coordinated_multitask
prepared_root: prepared
tasks:
  bbb_martins: {endpoint_id: bbb_martins, tdc_name: BBB_Martins, task_group: ADME, task_type: binary_classification, primary_metric: roc_auc}
  herg_karim: {endpoint_id: herg_karim, tdc_name: herg, task_group: Tox, task_type: binary_classification, primary_metric: roc_auc}
  ames: {endpoint_id: ames, tdc_name: AMES, task_group: Tox, task_type: binary_classification, primary_metric: roc_auc}
split_files: {train: train.csv, validation: valid.csv, test: test.csv}
audit:
  enforce_exact_smiles_exclusion: true
  enforce_scaffold_exclusion: true
  fail_on_invalid_molecules: true
  fail_on_conflicting_labels: true
  fail_on_duplicates: true
''',
        encoding="utf-8",
    )
    return config_path, root


def _write_split(path: Path, split: str, rows: list[tuple[str, str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "molecule_id": molecule_id,
                "smiles": smiles,
                "canonical_smiles": smiles,
                "target": target,
                "split": split,
            }
            for molecule_id, smiles, target in rows
        ]
    ).to_csv(path, index=False)

