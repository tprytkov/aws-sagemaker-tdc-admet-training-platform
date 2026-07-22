import json
from pathlib import Path

import pandas as pd
import pytest

from admet_platform.config import load_endpoint_config
from admet_platform.data import tdc_loader
from admet_platform.data.tdc_loader import (
    download_and_prepare_tdc_dataset,
    load_tdc_data,
    load_tdc_split,
    normalize_tdc_dataframe,
    normalize_tdc_raw_dataframe,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def test_adme_config_routes_to_adme_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class FakeADME:
        def __init__(self, name: str) -> None:
            calls.append(name)

        def get_data(self) -> pd.DataFrame:
            return _fake_frame()

        def get_split(self, method: str) -> dict[str, pd.DataFrame]:
            raise AssertionError("get_split must not be called")

    monkeypatch.setattr(tdc_loader, "_get_tdc_loader_class", lambda task_group: FakeADME)
    config = load_endpoint_config(CONFIG_DIR / "bbb_martins.yaml")

    data = load_tdc_split(config)

    assert calls == ["BBB_Martins"]
    assert list(data.columns) == ["Drug", "Y"]


def test_tox_config_routes_to_tox_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class FakeTox:
        def __init__(self, name: str) -> None:
            calls.append(name)

        def get_data(self) -> pd.DataFrame:
            return _fake_frame()

    monkeypatch.setattr(tdc_loader, "_get_tdc_loader_class", lambda task_group: FakeTox)
    config = load_endpoint_config(CONFIG_DIR / "herg_karim.yaml")

    data = load_tdc_data(config)

    assert calls == ["hERG_Karim"]
    assert len(data) == 2


def test_tdc_splits_normalize_to_project_split_names() -> None:
    config = load_endpoint_config(CONFIG_DIR / "bbb_martins.yaml")

    train_df = normalize_tdc_dataframe(_fake_frame(), "train", config)
    valid_df = normalize_tdc_dataframe(_fake_frame(), "valid", config)
    test_df = normalize_tdc_dataframe(_fake_frame(), "test", config)

    assert set(train_df["split"]) == {"train"}
    assert set(valid_df["split"]) == {"validation"}
    assert set(test_df["split"]) == {"test"}


def test_normalized_dataframe_has_project_columns() -> None:
    config = load_endpoint_config(CONFIG_DIR / "bbb_martins.yaml")

    normalized = normalize_tdc_dataframe(_fake_frame(), "valid", config)

    assert list(normalized.columns) == ["molecule_id", "smiles", "target", "split"]


def test_raw_normalized_dataframe_is_unsplit() -> None:
    config = load_endpoint_config(CONFIG_DIR / "ames.yaml")
    normalized = normalize_tdc_raw_dataframe(_fake_frame(), config)
    assert list(normalized.columns) == ["molecule_id", "smiles", "target"]
    assert "split" not in normalized


def test_missing_tdc_dependency_raises_clear_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "tdc.single_pred":
            raise ImportError("missing tdc")
        return original_import(name, *args, **kwargs)

    original_import = __import__
    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(RuntimeError, match="requirements-tdc-download.txt"):
        tdc_loader._get_tdc_loader_class("ADME")


def test_verified_pyt_dc_dependency_is_download_only() -> None:
    download = (PROJECT_ROOT / "requirements-tdc-download.txt").read_text(encoding="utf-8")
    root = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    training = (PROJECT_ROOT / "sagemaker" / "requirements.txt").read_text(encoding="utf-8")
    processing = (PROJECT_ROOT / "sagemaker" / "processing_requirements.txt").read_text(
        encoding="utf-8"
    )

    download_lines = {line.strip().lower() for line in download.splitlines() if line.strip()}
    root_lines = {line.strip().lower() for line in root.splitlines() if line.strip()}
    training_lines = {line.strip().lower() for line in training.splitlines() if line.strip()}
    processing_lines = {line.strip().lower() for line in processing.splitlines() if line.strip()}

    assert "pytdc==0.3.9" in download_lines
    assert "pytdc==0.3.9" in processing_lines
    assert not {"tdc", "pytdc==0.3.9"} & root_lines
    assert not {"tdc", "pytdc==0.3.9"} & training_lines


def test_download_and_prepare_writes_output_csv_and_summary_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeADME:
        def __init__(self, name: str) -> None:
            assert name == "BBB_Martins"

        def get_data(self) -> pd.DataFrame:
            return _fake_frame()

        def get_split(self, method: str) -> dict[str, pd.DataFrame]:
            raise AssertionError("get_split must not be called")

    monkeypatch.setattr(tdc_loader, "_get_tdc_loader_class", lambda task_group: FakeADME)
    output_csv = tmp_path / "bbb_tdc_clean.csv"
    summary_json = tmp_path / "bbb_tdc_summary.json"

    summary = download_and_prepare_tdc_dataset(
        config_path=CONFIG_DIR / "bbb_martins.yaml",
        output_csv=output_csv,
        summary_json=summary_json,
    )

    written_df = pd.read_csv(output_csv)
    written_summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert output_csv.exists()
    assert summary_json.exists()
    assert list(written_df.columns) == ["molecule_id", "smiles", "target"]
    assert summary == written_summary
    assert written_summary["split_status"] == "unsplit"
    assert written_summary["n_accepted_rows"] == 2
    assert written_summary["n_rejected_rows"] == 0


def _fake_split() -> dict[str, pd.DataFrame]:
    return {
        "train": _fake_frame(),
        "valid": _fake_frame(),
        "test": _fake_frame(),
    }


def _fake_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Drug": ["CCO", "CCN"],
            "Y": [0, 1],
        }
    )
