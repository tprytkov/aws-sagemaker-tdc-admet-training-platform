import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

from admet_platform.sagemaker import prepare_tdc_dataset as processing_entry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "bbb_martins.yaml"


def test_processing_path_resolution_from_cli_and_config_dir(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    endpoint_copy = config_dir / "endpoint.yaml"
    endpoint_copy.write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    args = Namespace(
        config_dir=str(config_dir),
        input_data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "output"),
        endpoint_config=None,
        mode="supplied_csv",
        development_row_limit=None,
    )

    paths = processing_entry.resolve_processing_paths(args, {})

    assert paths["endpoint_config"] == endpoint_copy
    assert paths["mode"] == "supplied_csv"


def test_supplied_csv_mode_writes_processing_layout(tmp_path: Path) -> None:
    input_dir = _write_input_dir(tmp_path)
    output_dir = tmp_path / "processing_output"

    manifest = processing_entry.prepare_processing_dataset(
        endpoint_config_path=CONFIG_PATH,
        processing_mode="supplied_csv",
        input_data_dir=input_dir,
        output_dir=output_dir,
        run_id="run-1",
    )

    _assert_processing_layout(output_dir)
    assert manifest["processing_mode"] == "supplied_csv"
    assert manifest["split_counts"] == {"train": 2, "validation": 1, "test": 1}


def test_tdc_download_mode_uses_loader_and_writes_layout(tmp_path: Path) -> None:
    output_dir = tmp_path / "processing_output"

    manifest = processing_entry.prepare_processing_dataset(
        endpoint_config_path=CONFIG_PATH,
        processing_mode="tdc_download",
        output_dir=output_dir,
        tdc_split_loader=lambda config: _fake_tdc_split(),
        run_id="run-tdc",
    )

    _assert_processing_layout(output_dir)
    assert manifest["processing_mode"] == "tdc_download"
    assert manifest["source_dataset"] == "BBB_Martins"


def test_missing_mode_errors(tmp_path: Path) -> None:
    args = Namespace(
        config_dir=None,
        input_data_dir=None,
        output_dir=str(tmp_path / "out"),
        endpoint_config=str(CONFIG_PATH),
        mode=None,
        development_row_limit=None,
    )
    paths = processing_entry.resolve_processing_paths(args, {})

    assert paths["mode"] is None
    assert processing_entry.main(["--endpoint-config", str(CONFIG_PATH), "--output-dir", str(tmp_path / "out")]) == 1


def test_missing_and_ambiguous_csv_errors(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    with pytest.raises(ValueError, match="No CSV"):
        processing_entry.resolve_single_input_csv(input_dir)
    (input_dir / "a.csv").write_text("x\n", encoding="utf-8")
    (input_dir / "b.csv").write_text("x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Multiple CSV"):
        processing_entry.resolve_single_input_csv(input_dir)


def test_required_column_validation(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    pd.DataFrame([{"molecule_id": "m1", "smiles": "CCO", "target": 1}]).to_csv(input_dir / "bad.csv", index=False)

    with pytest.raises(ValueError, match="missing required column"):
        processing_entry.prepare_processing_dataset(
            endpoint_config_path=CONFIG_PATH,
            processing_mode="supplied_csv",
            input_data_dir=input_dir,
            output_dir=tmp_path / "out",
        )


def test_manifest_schema_and_overlap_warning(tmp_path: Path) -> None:
    input_dir = _write_input_dir(tmp_path, overlap=True)
    output_dir = tmp_path / "out"
    manifest = processing_entry.prepare_processing_dataset(
        endpoint_config_path=CONFIG_PATH,
        processing_mode="supplied_csv",
        input_data_dir=input_dir,
        output_dir=output_dir,
    )

    assert {
        "run_id",
        "endpoint_id",
        "task_type",
        "source_dataset",
        "processing_mode",
        "source_row_count",
        "accepted_row_count",
        "rejected_row_count",
        "split_counts",
        "target_statistics",
        "duplicate_canonical_smiles_count",
        "cross_split_overlap_counts",
        "package_versions",
        "status",
        "warnings",
    } <= set(manifest)
    assert "canonical_smiles overlap detected across splits" in manifest["warnings"]


def test_failed_manifest_behavior_and_secret_redaction(tmp_path: Path) -> None:
    exc = RuntimeError("bad aws_secret_access_key=abc")
    manifest_path = processing_entry.write_failed_manifest(tmp_path / "out", exc)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["status"] == "failed"
    assert manifest["error"]["type"] == "RuntimeError"
    assert "[REDACTED]" in manifest["error"]["message"]


def test_minimal_processing_dependencies() -> None:
    requirements = (PROJECT_ROOT / "sagemaker" / "processing_requirements.txt").read_text(encoding="utf-8")
    requirement_lines = {line.strip() for line in requirements.splitlines() if line.strip()}

    assert "PyTDC==0.3.9" in requirement_lines
    assert "tdc" not in requirement_lines
    assert "torch" not in requirements.lower()
    assert "transformers" not in requirements.lower()
    assert "PyTDC" in processing_entry.PACKAGE_NAMES
    assert "tdc" not in processing_entry.PACKAGE_NAMES


def test_processing_wrapper_imports_without_repo_root_assumptions() -> None:
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "sagemaker" / "prepare_tdc_dataset.py"), "--help"],
        cwd=PROJECT_ROOT / "sagemaker",
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--mode" in result.stdout


def _write_input_dir(tmp_path: Path, overlap: bool = False) -> Path:
    input_dir = tmp_path / "input"
    input_dir.mkdir(exist_ok=True)
    valid_smiles = "CCO" if overlap else "CCN"
    pd.DataFrame(
        [
            {"molecule_id": "mol_001", "smiles": "CCO", "target": 1, "split": "train"},
            {"molecule_id": "mol_002", "smiles": "CCCl", "target": 0, "split": "train"},
            {"molecule_id": "mol_003", "smiles": valid_smiles, "target": 1, "split": "validation"},
            {"molecule_id": "mol_004", "smiles": "c1ccccc1", "target": 0, "split": "test"},
        ]
    ).to_csv(input_dir / "input.csv", index=False)
    return input_dir


def _fake_tdc_split() -> dict[str, pd.DataFrame]:
    return {
        "train": pd.DataFrame({"Drug": ["CCO", "CCCl"], "Y": [1, 0]}),
        "valid": pd.DataFrame({"Drug": ["CCN"], "Y": [1]}),
        "test": pd.DataFrame({"Drug": ["c1ccccc1"], "Y": [0]}),
    }


def _assert_processing_layout(output_dir: Path) -> None:
    assert (output_dir / "train" / "train.csv").exists()
    assert (output_dir / "validation" / "valid.csv").exists()
    assert (output_dir / "test" / "test.csv").exists()
    assert (output_dir / "metadata" / "data_profile.json").exists()
    assert (output_dir / "metadata" / "split_metadata.json").exists()
    assert (output_dir / "metadata" / "rejected_rows.csv").exists()
    assert (output_dir / "metadata" / "processing_manifest.json").exists()
