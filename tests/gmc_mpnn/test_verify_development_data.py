from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from admet_platform.chemprop.config import ChempropExperimentConfig
from scripts import verify_gmc_mpnn_bbb_development_data as verifier


def test_report_contains_hashes_counts_overlaps_and_membership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path)
    monkeypatch.setattr(verifier, "load_chemprop_config", lambda _: config)

    report = verifier.verify_development_data(tmp_path / "config.yaml")

    assert report["dataset_name"] == "BBB_Martins"
    assert report["split_manifest_identity"] == "synthetic-manifest"
    assert report["expected_train_hash"] == report["observed_train_hash"]
    assert report["expected_validation_hash"] == report["observed_validation_hash"]
    assert report["train_row_count"] == 2
    assert report["validation_row_count"] == 2
    assert report["train_negative_count"] == report["train_positive_count"] == 1
    assert report["validation_negative_count"] == report["validation_positive_count"] == 1
    assert report["exact_canonical_smiles_overlap_count"] == 0
    assert report["murcko_scaffold_overlap_count"] == 0
    assert report["canonical_smiles_verification_passed"] is True
    assert report["loaded_splits"] == ["train", "validation"]
    membership = report["chemprop_validation_membership"]
    assert membership == {
        "gmc_validation_row_count": 2,
        "chemprop_validation_row_count": 2,
        "exact_composite_key_match": True,
        "missing_from_gmc_count": 0,
        "extra_in_gmc_count": 0,
    }


def test_composite_membership_allows_repeated_molecule_ids() -> None:
    gmc = pd.DataFrame(
        {
            "molecule_id": ["shared", "shared"],
            "canonical_smiles": ["CCO", "CCN"],
            "target": [1, 1],
        }
    )
    chemprop = gmc.iloc[::-1].reset_index(drop=True)

    result = verifier._compare_validation_membership(gmc, chemprop)

    assert result["exact_composite_key_match"] is True
    assert result["missing_from_gmc_count"] == 0
    assert result["extra_in_gmc_count"] == 0


def test_composite_membership_reports_missing_and_extra() -> None:
    gmc = pd.DataFrame(
        {"molecule_id": ["a"], "canonical_smiles": ["CCO"], "target": [0]}
    )
    chemprop = pd.DataFrame(
        {"molecule_id": ["b"], "canonical_smiles": ["CCN"], "target": [1]}
    )

    result = verifier._compare_validation_membership(gmc, chemprop)

    assert result["exact_composite_key_match"] is False
    assert result["missing_from_gmc_count"] == 1
    assert result["extra_in_gmc_count"] == 1


def test_runtime_guard_blocks_test_resolution_and_reads(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    locked = config.prepared_root / config.split_files["test"]

    with verifier._deny_test_artifact_access(config):
        with pytest.raises(verifier.LockedTestAccessError):
            locked.resolve()
        with pytest.raises(verifier.LockedTestAccessError):
            locked.read_bytes()
        with pytest.raises(verifier.LockedTestAccessError):
            pd.read_csv(locked)
        with pytest.raises(verifier.LockedTestAccessError):
            open(locked, encoding="utf-8")


def test_development_alias_to_test_is_rejected_before_loading(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    config.split_files["validation"] = config.split_files["test"]

    with pytest.raises(verifier.LockedTestAccessError, match="aliases"):
        verifier._validate_development_paths(config)


def test_stdout_is_default_and_output_file_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    report = _minimal_report(exact_match=True)
    monkeypatch.setattr(verifier, "verify_development_data", lambda _: report)
    output = tmp_path / "report.json"

    assert verifier.main(["--config", str(tmp_path / "config.yaml")]) == 0
    assert json.loads(capsys.readouterr().out) == report
    assert not output.exists()

    assert verifier.main(
        ["--config", str(tmp_path / "config.yaml"), "--output", str(output)]
    ) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert capsys.readouterr().out == ""


def test_membership_mismatch_returns_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        verifier, "verify_development_data", lambda _: _minimal_report(exact_match=False)
    )

    assert verifier.main(["--config", str(tmp_path / "config.yaml")]) == 1
    capsys.readouterr()


def _fixture_config(root: Path) -> ChempropExperimentConfig:
    prepared = root / "prepared"
    prepared.mkdir()
    train = prepared / "train.csv"
    validation = prepared / "valid.csv"
    pd.DataFrame(
        [
            _row("shared-id", "CCO", 0, "train"),
            _row("shared-id", "c1ccccc1", 1, "train"),
        ]
    ).to_csv(train, index=False)
    pd.DataFrame(
        [
            _row("valid-1", "CCN", 0, "validation"),
            _row("valid-2", "c1ccncc1", 1, "validation"),
        ]
    ).to_csv(validation, index=False)
    hashes = {
        "train": hashlib.sha256(train.read_bytes()).hexdigest(),
        "validation": hashlib.sha256(validation.read_bytes()).hexdigest(),
    }
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "split_manifest_id": "synthetic-manifest",
                "endpoints": {
                    "bbb_martins": {
                        "splits": {
                            "train": {"output_csv_sha256": hashes["train"]},
                            "validation": {"output_csv_sha256": hashes["validation"]},
                            "test": {"output_csv_sha256": "not-read"},
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return ChempropExperimentConfig(
        source_path=root / "config.yaml",
        raw={"split_manifest_id": "synthetic-manifest"},
        endpoint="bbb_martins",
        endpoint_id="bbb_martins",
        dataset="BBB_Martins",
        dataset_version="synthetic",
        task_type="binary_classification",
        primary_metric="auroc",
        prepared_root=prepared,
        split_manifest=manifest,
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


def _row(molecule_id: str, smiles: str, target: int, split: str) -> dict[str, object]:
    return {
        "molecule_id": molecule_id,
        "smiles": smiles,
        "canonical_smiles": smiles,
        "target": target,
        "split": split,
    }


def _minimal_report(*, exact_match: bool) -> dict[str, object]:
    return {
        "dataset_name": "BBB_Martins",
        "chemprop_validation_membership": {"exact_composite_key_match": exact_match},
    }
