from pathlib import Path

import pandas as pd
import pytest

from admet_platform.config import load_endpoint_config
from admet_platform.data.schema import summarize_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"


@pytest.mark.parametrize(
    ("config_name", "sample_name"),
    [
        ("bbb_martins.yaml", "bbb_martins_sample.csv"),
        ("caco2_wang.yaml", "caco2_wang_sample.csv"),
        ("herg_karim.yaml", "herg_karim_sample.csv"),
    ],
)
def test_sample_csv_validates_against_endpoint_config(config_name: str, sample_name: str) -> None:
    config = load_endpoint_config(CONFIG_DIR / config_name)
    df = pd.read_csv(SAMPLE_DIR / sample_name)

    summary = summarize_dataset(df, config)

    assert summary["endpoint_id"] == config.endpoint_id
    assert summary["row_count"] == len(df)
    assert summary["split_counts"] == {"train": 3, "validation": 2, "test": 2}


def test_missing_required_column_raises_value_error() -> None:
    config = load_endpoint_config(CONFIG_DIR / "bbb_martins.yaml")
    df = pd.read_csv(SAMPLE_DIR / "bbb_martins_sample.csv").drop(columns=["molecule_id"])

    with pytest.raises(ValueError, match="missing required column"):
        summarize_dataset(df, config)


def test_invalid_split_value_raises_value_error() -> None:
    config = load_endpoint_config(CONFIG_DIR / "bbb_martins.yaml")
    df = pd.read_csv(SAMPLE_DIR / "bbb_martins_sample.csv")
    df.loc[0, "split"] = "holdout"

    with pytest.raises(ValueError, match="invalid value"):
        summarize_dataset(df, config)


def test_invalid_binary_target_raises_value_error() -> None:
    config = load_endpoint_config(CONFIG_DIR / "bbb_martins.yaml")
    df = pd.read_csv(SAMPLE_DIR / "bbb_martins_sample.csv")
    df.loc[0, "target"] = 2

    with pytest.raises(ValueError, match="only 0 or 1"):
        summarize_dataset(df, config)


def test_non_numeric_regression_target_raises_value_error() -> None:
    config = load_endpoint_config(CONFIG_DIR / "caco2_wang.yaml")
    df = pd.read_csv(SAMPLE_DIR / "caco2_wang_sample.csv")
    df["target"] = df["target"].astype(object)
    df.loc[0, "target"] = "not_numeric"

    with pytest.raises(ValueError, match="must be numeric"):
        summarize_dataset(df, config)


def test_empty_dataframe_raises_value_error() -> None:
    config = load_endpoint_config(CONFIG_DIR / "bbb_martins.yaml")
    df = pd.DataFrame(columns=["molecule_id", "smiles", "target", "split"])

    with pytest.raises(ValueError, match="must not be empty"):
        summarize_dataset(df, config)
