from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from admet_platform.chemprop.config import ChempropExperimentConfig
from admet_platform.gmc_mpnn.data import (
    load_bbb_development_data,
    load_bbb_development_split,
)


def test_valid_train_validation_loading_and_provenance(tmp_path: Path) -> None:
    config = _write_development_fixture(tmp_path)

    result = load_bbb_development_data(config)

    assert result.train["molecule_id"].tolist() == ["train-1", "train-2"]
    assert result.validation["molecule_id"].tolist() == ["valid-1", "valid-2"]
    assert result.train["target"].tolist() == [0, 1]
    assert result.validation["split"].tolist() == ["validation", "validation"]
    assert set(result.provenance.splits) == {"train", "validation"}
    assert result.provenance.splits["train"].row_count == 2
    assert result.provenance.splits["validation"].label_counts == {"0": 1, "1": 1}
    assert result.provenance.split_manifest_id == "synthetic-development-manifest"
    assert result.leakage.exact_canonical_smiles_count == 0
    assert result.leakage.murcko_scaffold_count == 0


def test_required_column_failure(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "train.csv", _rows("train")).drop_column("molecule_id")

    with pytest.raises(ValueError, match="missing columns.*molecule_id"):
        load_bbb_development_split(path, split="train")


@pytest.mark.parametrize("target", [2, -1, 0.5])
def test_invalid_nonbinary_target(tmp_path: Path, target: float) -> None:
    rows = _rows("train")
    rows[0]["target"] = target
    path = _write_csv(tmp_path / "train.csv", rows).path

    with pytest.raises(ValueError, match="binary 0/1"):
        load_bbb_development_split(path, split="train")


def test_incorrect_split_label(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "train.csv", _rows("validation")).path

    with pytest.raises(ValueError, match="unexpected split labels"):
        load_bbb_development_split(path, split="train")


def test_exact_duplicate_detection(tmp_path: Path) -> None:
    rows = _rows("train")
    path = _write_csv(tmp_path / "train.csv", [rows[0], rows[0]]).path

    with pytest.raises(ValueError, match="exact duplicate rows"):
        load_bbb_development_split(path, split="train")


@pytest.mark.parametrize(
    ("identity", "expected"),
    [
        ("composite", "molecule_id \\+ canonical_smiles"),
        ("canonical_smiles", "canonical_smiles"),
    ],
)
def test_conflicting_label_detection(
    tmp_path: Path, identity: str, expected: str
) -> None:
    rows = _rows("train")
    conflicting = dict(rows[0])
    conflicting["target"] = 1
    if identity == "canonical_smiles":
        conflicting["molecule_id"] = "train-conflict"
    path = _write_csv(tmp_path / "train.csv", [rows[0], conflicting]).path

    with pytest.raises(ValueError, match=f"conflicting labels for {expected}"):
        load_bbb_development_split(path, split="train")


def test_nonunique_molecule_ids_with_consistent_labels_are_preserved(tmp_path: Path) -> None:
    rows = [
        _row("shared-id", "CCO", 0, "train"),
        _row("shared-id", "CCN", 1, "train"),
    ]
    path = _write_csv(tmp_path / "train.csv", rows).path

    result = load_bbb_development_split(path, split="train")

    assert result["molecule_id"].tolist() == ["shared-id", "shared-id"]
    assert result["canonical_smiles"].tolist() == ["CCO", "CCN"]


@pytest.mark.parametrize("column", ["molecule_id", "canonical_smiles"])
def test_missing_identity_failure(tmp_path: Path, column: str) -> None:
    rows = _rows("train")
    rows[0][column] = None
    path = _write_csv(tmp_path / "train.csv", rows).path

    with pytest.raises(ValueError, match=f"missing {column}"):
        load_bbb_development_split(path, split="train")


def test_exact_smiles_overlap_is_reported_without_dropping_rows(tmp_path: Path) -> None:
    config = _write_development_fixture(
        tmp_path,
        train_rows=[_row("train-1", "CCO", 0, "train")],
        validation_rows=[_row("valid-1", "CCO", 1, "validation")],
    )

    result = load_bbb_development_data(config)

    assert result.leakage.exact_canonical_smiles == ("CCO",)
    assert result.leakage.exact_canonical_smiles_count == 1
    assert len(result.train) == len(result.validation) == 1


def test_scaffold_overlap_is_reported(tmp_path: Path) -> None:
    config = _write_development_fixture(
        tmp_path,
        train_rows=[_row("train-1", "Cc1ccccc1", 0, "train")],
        validation_rows=[_row("valid-1", "Oc1ccccc1", 1, "validation")],
    )

    result = load_bbb_development_data(config)

    assert result.leakage.exact_canonical_smiles_count == 0
    assert result.leakage.murcko_scaffolds == ("c1ccccc1",)


def test_test_split_is_rejected_before_path_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    touched: list[Path] = []

    def forbidden_read(path: Path) -> bytes:
        touched.append(path)
        raise AssertionError("a path was touched")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
    with pytest.raises(ValueError, match="outside the Phase-1 development loader"):
        load_bbb_development_split(tmp_path / "locked.csv", split="test")
    assert touched == []


def test_main_loader_touches_and_reports_development_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_development_fixture(tmp_path)
    opened: list[Path] = []
    original_read_bytes = Path.read_bytes

    def guarded_read(path: Path) -> bytes:
        opened.append(path)
        if path.name == "locked.csv":
            raise AssertionError("locked data path was touched")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    result = load_bbb_development_data(config)

    assert {path.name for path in opened} == {"manifest.json", "train.csv", "valid.csv"}
    assert set(result.provenance.splits) == {"train", "validation"}
    assert all("test" not in key for key in result.provenance.splits)


class _WrittenCsv:
    def __init__(self, path: Path):
        self.path = path

    def drop_column(self, column: str) -> Path:
        frame = pd.read_csv(self.path).drop(columns=[column])
        frame.to_csv(self.path, index=False)
        return self.path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> _WrittenCsv:
    pd.DataFrame(rows).to_csv(path, index=False)
    return _WrittenCsv(path)


def _row(molecule_id: str, smiles: str, target: int, split: str) -> dict[str, object]:
    return {
        "molecule_id": molecule_id,
        "smiles": smiles,
        "canonical_smiles": smiles,
        "target": target,
        "split": split,
    }


def _rows(split: str) -> list[dict[str, object]]:
    return [
        _row(f"{split}-1", "CCO", 0, split),
        _row(f"{split}-2", "c1ccccc1", 1, split),
    ]


def _write_development_fixture(
    root: Path,
    *,
    train_rows: list[dict[str, object]] | None = None,
    validation_rows: list[dict[str, object]] | None = None,
) -> ChempropExperimentConfig:
    prepared = root / "prepared"
    prepared.mkdir()
    train_path = _write_csv(prepared / "train.csv", train_rows or _rows("train")).path
    validation_path = _write_csv(
        prepared / "valid.csv", validation_rows or [
            _row("valid-1", "CCN", 0, "validation"),
            _row("valid-2", "c1ccncc1", 1, "validation"),
        ]
    ).path
    hashes = {
        "train": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "validation": hashlib.sha256(validation_path.read_bytes()).hexdigest(),
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "split_manifest_id": "synthetic-development-manifest",
                "endpoints": {
                    "bbb_martins": {
                        "splits": {
                            "train": {"output_csv_sha256": hashes["train"]},
                            "validation": {"output_csv_sha256": hashes["validation"]},
                            "test": {"output_csv_sha256": "not-accessed"},
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    raw = {"split_manifest_id": "synthetic-development-manifest"}
    return ChempropExperimentConfig(
        source_path=root / "synthetic-config.yaml",
        raw=raw,
        endpoint="bbb_martins",
        endpoint_id="bbb_martins",
        dataset="BBB_Martins",
        dataset_version="synthetic-software-fixture",
        task_type="binary_classification",
        primary_metric="auroc",
        prepared_root=prepared,
        split_manifest=manifest_path,
        split_files={"train": "train.csv", "validation": "valid.csv", "test": "locked.csv"},
        split_hash_keys={
            "train": "endpoints.bbb_martins.splits.train.output_csv_sha256",
            "validation": "endpoints.bbb_martins.splits.validation.output_csv_sha256",
            "test": "endpoints.bbb_martins.splits.test.output_csv_sha256",
        },
        tasks={},
        model={},
        training={},
    )
