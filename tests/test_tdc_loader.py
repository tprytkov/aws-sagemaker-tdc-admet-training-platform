import json
from pathlib import Path

import pandas as pd
import pytest

from admet_platform.config import load_endpoint_config
from admet_platform.data import tdc_loader
from admet_platform.data.tdc_loader import (
    download_and_prepare_tdc_dataset,
    load_tdc_split,
    normalize_tdc_dataframe,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def test_adme_config_routes_to_adme_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class FakeADME:
        def __init__(self, name: str) -> None:
            calls.append(name)

        def get_split(self, method: str) -> dict[str, pd.DataFrame]:
            assert method == "scaffold"
            return _fake_split()

    monkeypatch.setattr(tdc_loader, "_get_tdc_loader_class", lambda task_group: FakeADME)
    config = load_endpoint_config(CONFIG_DIR / "bbb_martins.yaml")

    split = load_tdc_split(config)

    assert calls == ["BBB_Martins"]
    assert set(split) == {"train", "valid", "test"}


def test_tox_config_routes_to_tox_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class FakeTox:
        def __init__(self, name: str) -> None:
            calls.append(name)

        def get_split(self, method: str) -> dict[str, pd.DataFrame]:
            assert method == "scaffold"
            return _fake_split()

    monkeypatch.setattr(tdc_loader, "_get_tdc_loader_class", lambda task_group: FakeTox)
    config = load_endpoint_config(CONFIG_DIR / "herg_karim.yaml")

    split = load_tdc_split(config)

    assert calls == ["herg"]
    assert set(split) == {"train", "valid", "test"}


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


def test_missing_tdc_dependency_raises_clear_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "tdc.single_pred":
            raise ImportError("missing tdc")
        return original_import(name, *args, **kwargs)

    original_import = __import__
    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(RuntimeError, match="PyTDC is required"):
        tdc_loader._get_tdc_loader_class("ADME")


def test_download_and_prepare_writes_output_csv_and_summary_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeADME:
        def __init__(self, name: str) -> None:
            assert name == "BBB_Martins"

        def get_split(self, method: str) -> dict[str, pd.DataFrame]:
            assert method == "scaffold"
            return _fake_split()

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
    assert list(written_df.columns) == ["molecule_id", "smiles", "target", "split"]
    assert summary == written_summary
    assert written_summary["n_train"] == 2
    assert written_summary["n_validation"] == 2
    assert written_summary["n_test"] == 2


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
