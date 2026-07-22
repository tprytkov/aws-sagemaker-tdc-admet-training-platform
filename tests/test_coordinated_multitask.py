import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from rdkit import Chem

import admet_platform.data.scaffolds as scaffold_module
from admet_platform.data.coordinated_multitask import build_coordinated_multitask_splits
from admet_platform.data.multitask import load_multitask_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "build_coordinated_multitask_splits.py"
TASKS = ("bbb_martins", "herg_karim", "ames")


def test_coordinated_outputs_are_deterministic_and_leakage_safe(tmp_path: Path) -> None:
    config_path, source = _fixture(tmp_path)
    config = load_multitask_config(config_path)
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_result = build_coordinated_multitask_splits(config, source, first, seed=42)
    second_result = build_coordinated_multitask_splits(config, source, second, seed=42)

    for task in TASKS:
        for name in ("train.csv", "valid.csv", "test.csv"):
            assert (first / task / name).read_bytes() == (second / task / name).read_bytes()
    for name in (
        "quarantined_conflicts.csv",
        "deduplication_provenance.csv",
        "global_scaffold_assignments.csv",
        "coordinated_split_manifest.json",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()

    conflicts = pd.read_csv(first / "quarantined_conflicts.csv")
    conflict = conflicts.loc[conflicts["canonical_smiles"] == "CCl"].iloc[0]
    assert conflict["endpoint_id"] == "bbb_martins"
    assert conflict["labels"] == "0|1"
    assert conflict["row_count"] == 2
    assert not _endpoint_rows(first, "bbb_martins")["canonical_smiles"].eq("CCl").any()

    deduplication = pd.read_csv(first / "deduplication_provenance.csv")
    duplicate = deduplication.loc[
        (deduplication["endpoint_id"] == "bbb_martins")
        & (deduplication["canonical_smiles"] == "CCO")
    ].iloc[0]
    assert duplicate["kept_molecule_id"] == "bbb_martins-1"
    assert duplicate["removed_row_count"] == 1
    assert _endpoint_rows(first, "bbb_martins")["canonical_smiles"].eq("CCO").sum() == 1

    bbb_shared = _endpoint_rows(first, "bbb_martins").query("canonical_smiles == 'CS'")
    herg_shared = _endpoint_rows(first, "herg_karim").query("canonical_smiles == 'CS'")
    assert len(bbb_shared) == len(herg_shared) == 1
    assert bbb_shared.iloc[0]["split"] == herg_shared.iloc[0]["split"]

    bbb_aromatic = _endpoint_rows(first, "bbb_martins").query("canonical_smiles == 'Cc1ccccc1'")
    ames_aromatic = _endpoint_rows(first, "ames").query("canonical_smiles == 'Oc1ccccc1'")
    assert bbb_aromatic.iloc[0]["split"] == ames_aromatic.iloc[0]["split"]

    assignments = pd.read_csv(first / "global_scaffold_assignments.csv")
    assert "ACYCLIC::CS" in set(assignments["scaffold_key"])
    for task in TASKS:
        rows = _endpoint_rows(first, task)
        for split in ("train", "validation", "test"):
            assert set(rows.loc[rows["split"] == split, "target"]) == {0, 1}

    counts = first_result.audit.summary["counts"]
    assert counts["invalid_molecules"] == 0
    assert counts["duplicate_groups"] == 0
    assert counts["conflicting_label_groups"] == 0
    assert counts["exact_train_vs_heldout_overlaps"] == 0
    assert counts["scaffold_train_vs_heldout_overlaps"] == 0
    assert counts["blocking_violations"] == 0
    assert first_result.audit.summary["leakage_safe_for_training"] is True
    assert first_result.manifest == second_result.manifest


def test_different_seed_changes_assignment_when_groups_are_interchangeable(tmp_path: Path) -> None:
    config_path, source = _fixture(tmp_path)
    config = load_multitask_config(config_path)
    first = tmp_path / "seed-42"
    second = tmp_path / "seed-43"

    build_coordinated_multitask_splits(config, source, first, seed=42)
    build_coordinated_multitask_splits(config, source, second, seed=43)

    first_assignments = pd.read_csv(first / "global_scaffold_assignments.csv").set_index("scaffold_key")
    second_assignments = pd.read_csv(second / "global_scaffold_assignments.csv").set_index("scaffold_key")
    assert (first_assignments["split"] != second_assignments["split"]).any()


def test_bad_stereo_fallback_preserves_canonical_and_does_not_invalidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path, source = _fixture(tmp_path)
    config = load_multitask_config(config_path)
    output = tmp_path / "fallback"
    real_scaffold = scaffold_module.MurckoScaffold.MurckoScaffoldSmiles
    calls = 0
    fallback_molecule = ""

    def fail_once(*, mol: Chem.Mol, includeChirality: bool) -> str:
        nonlocal calls, fallback_molecule
        calls += 1
        if calls == 1:
            raise RuntimeError("Pre-condition Violation: bad bond stereo")
        if calls == 2:
            fallback_molecule = Chem.MolToSmiles(mol, isomericSmiles=True)
            return ""
        return real_scaffold(mol=mol, includeChirality=includeChirality)

    monkeypatch.setattr(scaffold_module.MurckoScaffold, "MurckoScaffoldSmiles", fail_once)
    result = build_coordinated_multitask_splits(config, source, output, seed=42)

    rows = _endpoint_rows(output, "bbb_martins")
    assert "C/C=C/C" in set(rows["canonical_smiles"])
    assert fallback_molecule == "CC=CC"
    assignments = pd.read_csv(output / "global_scaffold_assignments.csv")
    assert "ACYCLIC::C/C=C/C" in set(assignments["scaffold_key"])
    assert result.manifest["invalid_molecules"] == 0
    assert result.audit.tables["invalid"].empty


def test_cli_builds_separate_coordinated_track(tmp_path: Path) -> None:
    config_path, source = _fixture(tmp_path)
    output = tmp_path / "cli-output"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(config_path),
            "--source-root",
            str(source),
            "--output-root",
            str(output),
            "--seed",
            "42",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output / "coordinated_split_manifest.json").is_file()
    assert (output / "audit" / "audit_summary.json").is_file()
    summary = json.loads((output / "audit" / "audit_summary.json").read_text(encoding="utf-8"))
    assert summary["leakage_safe_for_training"] is True


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    rows = {
        "bbb_martins": [
            ("C/C=C/C", 0), ("CCO", 0), ("CCCO", 0), ("CCCCO", 0),
            ("CCCCCO", 0), ("CCCCCCO", 0), ("CCN", 1), ("CCCN", 1),
            ("CCCCN", 1), ("CCCCCN", 1), ("CCCCCCN", 1), ("CCCCCCCN", 1),
            ("CS", 0), ("Cc1ccccc1", 1),
        ],
        "herg_karim": [
            ("COC", 0), ("CCOC", 0), ("CCCOC", 0), ("CCCCOC", 0),
            ("CCCCCOC", 0), ("CCCCCCOC", 0), ("CNC", 1), ("CCNC", 1),
            ("CCCNC", 1), ("CCCCNC", 1), ("CCCCCNC", 1), ("CCCCCCNC", 1),
            ("CS", 1),
        ],
        "ames": [
            ("CC(=O)O", 0), ("CCC(=O)O", 0), ("CCCC(=O)O", 0),
            ("CCCCC(=O)O", 0), ("CCCCCC(=O)O", 0), ("CCCCCCC(=O)O", 0),
            ("CC(=O)N", 1), ("CCC(=O)N", 1), ("CCCC(=O)N", 1),
            ("CCCCC(=O)N", 1), ("CCCCCC(=O)N", 1), ("CCCCCCC(=O)N", 1),
            ("Oc1ccccc1", 0),
        ],
    }
    for task, task_rows in rows.items():
        split_rows = {"train": [], "validation": [], "test": []}
        for index, (smiles, target) in enumerate(task_rows):
            split = ("train", "validation", "test")[index % 3]
            split_rows[split].append((f"{task}-{index}", smiles, target))
        if task == "bbb_martins":
            split_rows["validation"].append(("bbb-duplicate", "CCO", 0))
            split_rows["train"].append(("bbb-conflict-zero", "CCl", 0))
            split_rows["test"].append(("bbb-conflict-one", "CCl", 1))
        for split, split_values in split_rows.items():
            file_name = "valid.csv" if split == "validation" else f"{split}.csv"
            _write_split(source / task / file_name, split, split_values)

    config_path = tmp_path / "multitask.yaml"
    config_path.write_text(
        '''schema_version: "1.0.0"
run_name: coordinated-test
split_track: coordinated_multitask
prepared_root: coordinated
tasks:
  bbb_martins: {endpoint_id: bbb_martins, tdc_name: BBB_Martins, task_group: ADME, task_type: binary_classification, primary_metric: roc_auc}
  herg_karim: {endpoint_id: herg_karim, tdc_name: hERG_Karim, task_group: Tox, task_type: binary_classification, primary_metric: roc_auc}
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
    return config_path, source


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


def _endpoint_rows(output: Path, task: str) -> pd.DataFrame:
    return pd.concat(
        [pd.read_csv(output / task / name) for name in ("train.csv", "valid.csv", "test.csv")],
        ignore_index=True,
    )
