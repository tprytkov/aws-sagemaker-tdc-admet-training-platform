from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from rdkit import Chem

from admet_platform.gmc_mpnn import scaling
from scripts import create_gmc_mpnn_bbb_oof_manifest as oof


def test_nested_oof_manifest_is_deterministic_grouped_and_train_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = _synthetic_frozen()
    calls: list[tuple[Any, ...]] = []

    def fake_load(*args: Any, **kwargs: Any) -> Any:
        calls.append(args)
        assert not kwargs
        return frozen

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("OOF manifest generation must not fit a scaler.")

    monkeypatch.setattr(oof, "load_frozen_training_manifest", fake_load)
    monkeypatch.setattr(oof, "_scaffold_key", lambda record: record.synthetic_scaffold_key)
    monkeypatch.setattr(scaling.StandardScaler, "fit", forbidden)
    monkeypatch.setattr(scaling.StandardScaler, "fit_transform", forbidden)
    monkeypatch.setattr(scaling.StandardScaler, "partial_fit", forbidden)
    output = tmp_path / "oof"

    summary = oof.create_oof_manifest(
        oof.OOFConfig(training_preprocessing_dir=Path("frozen-train"), output_dir=output),
        git_commit="a" * 40,
    )

    assert calls == [(Path("frozen-train"),)]
    assert summary["train_count"] == 1558
    assert summary["outer_fold_count"] == 5
    assert summary["inner_fold_count"] == 8
    assert summary["split_seed"] == 1729
    assert summary["inner_early_stop_target_fraction"] == pytest.approx(0.125)
    assert summary["validation_artifact_accessed"] is False
    assert summary["test_artifact_accessed"] is False
    assert all(summary["checks"].values())
    assert summary["git_commit"] == "a" * 40
    assert summary["frozen_train_provenance"]["feature_manifest_sha256"] == "b" * 64
    assert summary["frozen_train_provenance"]["ordered_input_artifact_sha256"] == "d" * 64

    outer = pd.read_csv(output / oof.OUTER_FILENAME, keep_default_na=False)
    inner = pd.read_csv(output / oof.INNER_FILENAME, keep_default_na=False)
    persisted_summary = json.loads((output / oof.SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert persisted_summary == summary
    assert list(outer.columns) == [
        "record_key",
        "molecule_id",
        "canonical_smiles",
        "label",
        "scaffold_key",
        "outer_fold",
    ]
    assert list(inner.columns) == ["outer_fold", "record_key", "inner_role"]
    assert len(outer) == 1558
    assert outer["record_key"].is_unique
    assert set(outer["outer_fold"]) == {0, 1, 2, 3, 4}
    assert outer.groupby("scaffold_key")["outer_fold"].nunique().max() == 1
    assert len(inner) == 1558 * 4

    scaffold_by_key = outer.set_index("record_key")["scaffold_key"]
    for outer_fold in range(5):
        holdout_keys = set(outer.loc[outer["outer_fold"] == outer_fold, "record_key"])
        fold_inner = inner.loc[inner["outer_fold"] == outer_fold]
        assert not holdout_keys.intersection(fold_inner["record_key"])
        assert fold_inner["record_key"].is_unique
        early_keys = fold_inner.loc[
            fold_inner["inner_role"] == "inner_early_stop_validation", "record_key"
        ]
        train_keys = fold_inner.loc[fold_inner["inner_role"] == "inner_train", "record_key"]
        assert not set(scaffold_by_key.loc[early_keys]).intersection(
            scaffold_by_key.loc[train_keys]
        )
        fraction = len(early_keys) / len(fold_inner)
        assert 0.10 <= fraction <= 0.15
        assert set(outer.set_index("record_key").loc[early_keys, "label"]) == {0, 1}


def test_scaffold_key_uses_murcko_and_distinct_acyclic_fallbacks() -> None:
    ethanol = SimpleNamespace(record_key="ethanol", molecule=Chem.MolFromSmiles("CCO"))
    propane = SimpleNamespace(record_key="propane", molecule=Chem.MolFromSmiles("CCC"))
    toluene = SimpleNamespace(record_key="toluene", molecule=Chem.MolFromSmiles("Cc1ccccc1"))
    ethylbenzene = SimpleNamespace(
        record_key="ethylbenzene", molecule=Chem.MolFromSmiles("CCc1ccccc1")
    )

    ethanol_key = oof._scaffold_key(ethanol)
    propane_key = oof._scaffold_key(propane)
    assert ethanol_key.startswith("acyclic:")
    assert propane_key.startswith("acyclic:")
    assert ethanol_key != propane_key
    assert oof._scaffold_key(toluene) == oof._scaffold_key(ethylbenzene) == "murcko:c1ccccc1"


def test_existing_output_fails_before_train_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "existing"
    output.mkdir()

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("TRAIN must not load after output collision.")

    monkeypatch.setattr(oof, "load_frozen_training_manifest", forbidden)
    with pytest.raises(FileExistsError, match="already exists"):
        oof.create_oof_manifest(
            oof.OOFConfig(training_preprocessing_dir=Path("train"), output_dir=output)
        )


def test_train_count_mismatch_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frozen = _synthetic_frozen(count=1557)
    monkeypatch.setattr(oof, "load_frozen_training_manifest", lambda _: frozen)

    with pytest.raises(oof.GMCOOFSplitError, match="exactly 1,558"):
        oof.create_oof_manifest(
            oof.OOFConfig(
                training_preprocessing_dir=Path("train"),
                output_dir=tmp_path / "output",
            )
        )


def test_missing_class_in_outer_holdout_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    frozen = _synthetic_frozen()
    monkeypatch.setattr(oof, "_scaffold_key", lambda record: record.synthetic_scaffold_key)
    tables = oof._generate_split_tables(frozen.records)
    first_fold = int(tables.outer["outer_fold"].iloc[0])
    tables.outer.loc[tables.outer["outer_fold"] == first_fold, "label"] = 0

    with pytest.raises(oof.GMCOOFSplitError, match="Both classes.*outer fold"):
        oof._validate_split_tables(tables)


def test_missing_class_in_inner_early_stop_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = _synthetic_frozen()
    monkeypatch.setattr(oof, "_scaffold_key", lambda record: record.synthetic_scaffold_key)
    tables = oof._generate_split_tables(frozen.records)
    fold = 0
    labels = tables.outer.set_index("record_key")["label"]
    fold_rows = tables.inner["outer_fold"] == fold
    positive_keys = set(labels[labels == 1].index)
    early_positive = fold_rows & tables.inner["record_key"].isin(positive_keys)
    tables.inner.loc[fold_rows, "inner_role"] = "inner_train"
    tables.inner.loc[early_positive, "inner_role"] = "inner_early_stop_validation"

    with pytest.raises(oof.GMCOOFSplitError, match="Both classes.*inner early stop"):
        oof._validate_split_tables(tables)


def test_repeated_generation_difference_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = _synthetic_frozen()
    first = oof.SplitTables(
        outer=pd.DataFrame({"assignment": [0]}),
        inner=pd.DataFrame({"assignment": [0]}),
    )
    second = oof.SplitTables(
        outer=pd.DataFrame({"assignment": [1]}),
        inner=pd.DataFrame({"assignment": [0]}),
    )
    generated = iter((first, second))
    monkeypatch.setattr(oof, "load_frozen_training_manifest", lambda _: frozen)
    monkeypatch.setattr(oof, "_generate_split_tables", lambda _: next(generated))

    with pytest.raises(oof.GMCOOFSplitError, match="Repeated deterministic"):
        oof.create_oof_manifest(
            oof.OOFConfig(
                training_preprocessing_dir=Path("train"),
                output_dir=tmp_path / "output",
            )
        )


def _synthetic_frozen(count: int = 1558) -> Any:
    records = tuple(
        SimpleNamespace(
            source_row_index=index,
            record_key=f"record-{index:04d}",
            molecule_id=f"molecule-{index:04d}",
            canonical_smiles="C" if index % 2 == 0 else "CC",
            geometry_smiles="C" if index % 2 == 0 else "CC",
            label=index % 2,
            molecule=None,
            synthetic_scaffold_key=f"scaffold-{index % 100:03d}",
        )
        for index in range(count)
    )
    return SimpleNamespace(
        records=records,
        source_row_count=1561,
        feature_manifest_sha256="b" * 64,
        molecule_status_sha256="c" * 64,
        ordered_input_artifact_sha256="d" * 64,
        preprocessing_provenance={
            "training_preprocessing_version": "gmc-mpnn-training-raw-ggl-v1",
            "standardization_version": "parent-fragment-v1",
            "geometry_preprocessing_version": "rdkit-etkdgv3-mmff94s-retry2000-v2",
            "ggl_preprocessing_version": "released-pooled-six-feature-v1",
            "rdkit_version": "synthetic",
            "validation_artifact_accessed": False,
            "test_artifact_accessed": False,
        },
    )
