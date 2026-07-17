from pathlib import Path

import pandas as pd
import pytest
from rdkit import Chem

from admet_platform.data import prepare
from admet_platform.data.prepare import prepare_dataset_artifacts
from admet_platform.data.scaffolds import safe_murcko_scaffold
import admet_platform.data.scaffolds as scaffold_module


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AMES_CONFIG = PROJECT_ROOT / "configs" / "ames.yaml"


def test_normal_scaffold_generation() -> None:
    molecule = Chem.MolFromSmiles("c1ccccc1CCO")
    result = safe_murcko_scaffold(molecule)
    assert result.scaffold == "c1ccccc1"
    assert result.used_stereo_fallback is False


def test_bad_bond_stereo_removal_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    molecule = Chem.MolFromSmiles("C/C=C/C")
    original = Chem.MolToSmiles(molecule, isomericSmiles=True)
    calls = 0

    def fake_scaffold(*, mol: Chem.Mol, includeChirality: bool) -> str:
        nonlocal calls
        calls += 1
        assert includeChirality is False
        if calls == 1:
            raise RuntimeError("Pre-condition Violation: bad bond stereo")
        assert Chem.MolToSmiles(mol, isomericSmiles=True) == "CC=CC"
        return ""

    monkeypatch.setattr(scaffold_module.MurckoScaffold, "MurckoScaffoldSmiles", fake_scaffold)
    result = safe_murcko_scaffold(molecule)
    assert result.used_stereo_fallback is True
    assert Chem.MolToSmiles(molecule, isomericSmiles=True) == original


def test_unrecoverable_scaffold_is_rejected_with_machine_readable_detail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = _write_unsplit(tmp_path / "raw.csv")
    real_helper = prepare.safe_murcko_scaffold

    def fail_selected(molecule_or_smiles: Chem.Mol | str):
        molecule = (
            Chem.MolFromSmiles(molecule_or_smiles)
            if isinstance(molecule_or_smiles, str)
            else molecule_or_smiles
        )
        if Chem.MolToSmiles(molecule, canonical=True) == "CCN":
            raise RuntimeError("bad bond stereo remains invalid")
        return real_helper(molecule_or_smiles)

    monkeypatch.setattr(prepare, "safe_murcko_scaffold", fail_selected)
    output = tmp_path / "prepared"
    summary = prepare_dataset_artifacts(source, AMES_CONFIG, output)
    problems = pd.read_csv(output / "problematic_molecules.csv")
    rejected = pd.read_csv(output / "rejected_rows.csv")

    assert summary["n_accepted_rows"] == 5
    assert summary["n_rejected_rows"] == 1
    assert rejected.loc[0, "rejection_reason"] == "scaffold_generation_failed"
    assert problems.loc[0, "source_row"] == 1
    assert problems.loc[0, "molecule_id"] == "mol-1"
    assert problems.loc[0, "original_smiles"] == "CCN"
    assert problems.loc[0, "endpoint"] == "ames"
    assert problems.loc[0, "failure_stage"] == "scaffold_assignment"
    assert problems.loc[0, "exception_category"] == "RuntimeError"
    assert "bad bond stereo" in problems.loc[0, "error_message"]


def test_unsplit_scaffold_preparation_is_deterministic_and_preserves_smiles(tmp_path: Path) -> None:
    source = _write_unsplit(tmp_path / "raw.csv")
    first = tmp_path / "first"
    second = tmp_path / "second"
    prepare_dataset_artifacts(source, AMES_CONFIG, first)
    prepare_dataset_artifacts(source, AMES_CONFIG, second)

    for name in ("train.csv", "valid.csv", "test.csv", "split_metadata.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    prepared = pd.concat(
        [pd.read_csv(first / name) for name in ("train.csv", "valid.csv", "test.csv")],
        ignore_index=True,
    )
    source_frame = pd.read_csv(source)
    assert set(prepared["smiles"]) == set(source_frame["smiles"])
    assert prepared["canonical_smiles"].notna().all()
    scaffold_to_split: dict[str, set[str]] = {}
    for row in prepared.itertuples():
        molecule = Chem.MolFromSmiles(row.canonical_smiles)
        scaffold = safe_murcko_scaffold(molecule).scaffold or f"ACYCLIC::{row.canonical_smiles}"
        scaffold_to_split.setdefault(scaffold, set()).add(row.split)
    assert all(len(splits) == 1 for splits in scaffold_to_split.values())


def _write_unsplit(path: Path) -> Path:
    pd.DataFrame(
        {
            "molecule_id": [f"mol-{index}" for index in range(6)],
            "smiles": ["CCO", "CCN", "c1ccccc1", "c1ccccc1O", "C1CCCCC1", "C1CCCCC1O"],
            "target": [0, 1, 0, 1, 0, 1],
        }
    ).to_csv(path, index=False)
    return path
