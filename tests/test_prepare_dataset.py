import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from admet_platform.data.prepare import prepare_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"


def test_prepare_dataset_writes_cleaned_csv_and_summary_json(tmp_path: Path) -> None:
    output_csv = tmp_path / "bbb_clean.csv"
    summary_json = tmp_path / "bbb_summary.json"

    summary = prepare_dataset(
        input_csv=SAMPLE_DIR / "bbb_martins_sample.csv",
        config_path=CONFIG_DIR / "bbb_martins.yaml",
        output_csv=output_csv,
        summary_json=summary_json,
    )

    cleaned_df = pd.read_csv(output_csv)
    written_summary = json.loads(summary_json.read_text(encoding="utf-8"))

    assert output_csv.exists()
    assert summary_json.exists()
    assert list(cleaned_df.columns) == ["molecule_id", "smiles", "target", "split"]
    assert summary == written_summary
    assert written_summary["endpoint_id"] == "bbb_martins"
    assert written_summary["n_rows"] == 7


def test_prepare_dataset_drops_duplicate_rows(tmp_path: Path) -> None:
    source_df = pd.read_csv(SAMPLE_DIR / "bbb_martins_sample.csv")
    duplicated_df = pd.concat([source_df, source_df.iloc[[0]]], ignore_index=True)
    input_csv = tmp_path / "bbb_with_duplicate.csv"
    output_csv = tmp_path / "bbb_clean.csv"
    summary_json = tmp_path / "bbb_summary.json"
    duplicated_df.to_csv(input_csv, index=False)

    summary = prepare_dataset(
        input_csv=input_csv,
        config_path=CONFIG_DIR / "bbb_martins.yaml",
        output_csv=output_csv,
        summary_json=summary_json,
    )

    cleaned_df = pd.read_csv(output_csv)
    assert len(cleaned_df) == len(source_df)
    assert summary["n_rows"] == len(source_df)


def test_binary_classification_summary_includes_class_counts(tmp_path: Path) -> None:
    summary = prepare_dataset(
        input_csv=SAMPLE_DIR / "herg_karim_sample.csv",
        config_path=CONFIG_DIR / "herg_karim.yaml",
        output_csv=tmp_path / "herg_clean.csv",
        summary_json=tmp_path / "herg_summary.json",
    )

    assert summary["class_counts"] == {"0": 4, "1": 3}
    assert "target_min" not in summary
    assert "target_max" not in summary


def test_regression_summary_includes_target_min_and_max(tmp_path: Path) -> None:
    summary = prepare_dataset(
        input_csv=SAMPLE_DIR / "caco2_wang_sample.csv",
        config_path=CONFIG_DIR / "caco2_wang.yaml",
        output_csv=tmp_path / "caco2_clean.csv",
        summary_json=tmp_path / "caco2_summary.json",
    )

    assert summary["target_min"] == -5.1
    assert summary["target_max"] == -3.9
    assert "class_counts" not in summary


def test_prepare_dataset_cli_works_on_sample_dataset(tmp_path: Path) -> None:
    output_csv = tmp_path / "bbb_cli_clean.csv"
    summary_json = tmp_path / "bbb_cli_summary.json"

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "prepare_dataset.py"),
            "--input-csv",
            str(SAMPLE_DIR / "bbb_martins_sample.csv"),
            "--config",
            str(CONFIG_DIR / "bbb_martins.yaml"),
            "--output-csv",
            str(output_csv),
            "--summary-json",
            str(summary_json),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert output_csv.exists()
    assert summary_json.exists()
    assert "Wrote cleaned CSV" in result.stdout
