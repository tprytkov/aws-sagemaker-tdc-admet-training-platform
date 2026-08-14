from pathlib import Path

import pandas as pd
import pytest

from admet_platform.chemprop.config import load_chemprop_config
from admet_platform.chemprop.data import assert_no_development_leakage, load_verified_development_data


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["multitask_admet_regression", "bbb_martins"])
def test_real_development_splits_verify_without_opening_locked_test(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_chemprop_config(ROOT / f"configs/chemprop/{name}.yaml")
    locked_tests = (
        {(config.prepared_root / endpoint / config.split_files["test"]).resolve()
         for endpoint in config.tasks}
        if config.tasks
        else {(config.prepared_root / config.split_files["test"]).resolve()}
    )
    original = Path.read_bytes

    def guarded(path: Path) -> bytes:
        if path.resolve() in locked_tests:
            raise AssertionError("locked test was opened")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    data = load_verified_development_data(config)
    assert len(data.train) > 0 and len(data.validation) > 0
    assert all(not key.endswith("/test") and key != "test" for key in data.actual_hashes)
    assert any(key.endswith("/test") or key == "test" for key in data.expected_hashes)
    if config.tasks:
        assert data.label_counts == {
            "caco2_wang": {"train": 713, "validation": 89},
            "lipophilicity_astrazeneca": {"train": 3390, "validation": 399},
            "solubility_aqsoldb": {"train": 6885, "validation": 1787},
            "ppbr_az": {"train": 1296, "validation": 158},
            "vdss_lombardo": {"train": 877, "validation": 108},
        }
        assert data.train[list(config.tasks)].isna().any().any()


def test_canonical_and_scaffold_leakage_are_blocking() -> None:
    train = pd.DataFrame({"canonical_smiles": ["CCO"], "murcko_scaffold": ["ACYCLIC::CCO"]})
    validation = pd.DataFrame({"canonical_smiles": ["CCO"], "murcko_scaffold": ["ACYCLIC::CCO"]})
    with pytest.raises(ValueError, match="Canonical-SMILES overlap"):
        assert_no_development_leakage(train, validation)
