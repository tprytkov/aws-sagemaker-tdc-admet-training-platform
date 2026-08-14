from pathlib import Path

import pytest

from admet_platform.chemprop.config import load_chemprop_config
from admet_platform.chemprop.preflight import build_real_data_preflight_audit


ROOT = Path(__file__).resolve().parents[1]


def test_real_preflight_is_locked_test_safe_and_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    regression = load_chemprop_config(
        ROOT / "configs/chemprop/multitask_admet_regression.yaml"
    )
    bbb = load_chemprop_config(ROOT / "configs/chemprop/bbb_martins.yaml")
    locked = {
        (regression.prepared_root / endpoint / regression.split_files["test"]).resolve()
        for endpoint in regression.tasks
    }
    locked.add((bbb.prepared_root / bbb.split_files["test"]).resolve())
    original = Path.read_bytes

    def guarded(path: Path) -> bytes:
        if path.resolve() in locked:
            raise AssertionError("locked-test CSV was opened")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    audit, _, _ = build_real_data_preflight_audit(regression, bbb)
    assert audit["locked_test_csv_opened"] is False
    assert audit["endpoint_order"] == list(regression.tasks)
    assert audit["regression"]["sparse_matrix"]["train"]["zero_target_rows"] == 0
    assert audit["bbb"]["calibration_fit_split"] == "validation"
    assert audit["bbb"]["threshold_selection_split"] == "validation"
