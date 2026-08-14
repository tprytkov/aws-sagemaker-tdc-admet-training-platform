import subprocess
import sys
from pathlib import Path

import pytest

from admet_platform.chemprop.config import load_chemprop_config


ROOT = Path(__file__).resolve().parents[1]


def test_multitask_regression_and_bbb_configs_are_valid_and_independent() -> None:
    regression = load_chemprop_config(
        ROOT / "configs/chemprop/multitask_admet_regression.yaml"
    )
    bbb = load_chemprop_config(ROOT / "configs/chemprop/bbb_martins.yaml")
    assert regression.task_type == "regression"
    assert bbb.task_type == "binary_classification"
    assert tuple(regression.tasks) == (
        "caco2_wang", "lipophilicity_astrazeneca", "solubility_aqsoldb", "ppbr_az",
        "vdss_lombardo",
    )
    assert regression.endpoint != bbb.endpoint
    assert regression.raw["chemprop_version"] == bbb.raw["chemprop_version"] == "2.3.1"
    assert bbb.raw["thresholds"]["selected_method"] == "maximum_validation_mcc"
    assert bbb.raw["replacement_policy"] == "retain_existing_predictors_until_locked_test_approval"


def test_missing_config_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("task_type: regression\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required fields"):
        load_chemprop_config(path)


def test_package_import_does_not_import_chemprop_or_load_models() -> None:
    command = (
        f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r}); "
        "import admet_platform.chemprop; "
        "assert 'chemprop' not in sys.modules; assert 'lightning' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", command], check=True)
