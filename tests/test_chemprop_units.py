import numpy as np
import pytest

from admet_platform.chemprop.units import (
    CACO2_LABEL_CONTRACT_VERSION,
    CACO2_MODEL_OUTPUT_UNIT,
    caco2_cm_per_s_to_log10,
    caco2_log10_to_cm_per_s,
)


def test_caco2_label_contract_forward_and_inverse_conversion() -> None:
    stored = np.asarray([-7.7600002, -5.0, -3.51])
    physical = caco2_log10_to_cm_per_s(stored)
    assert physical == pytest.approx([1.737800028e-8, 1.0e-5, 3.090295433e-4])
    assert caco2_cm_per_s_to_log10(physical) == pytest.approx(stored)
    assert CACO2_LABEL_CONTRACT_VERSION == "caco2_wang_log10_cm_per_s_v1"
    assert CACO2_MODEL_OUTPUT_UNIT == "log10(Papp [cm/s])"


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_caco2_physical_conversion_rejects_nonpositive_or_nonfinite(value: float) -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        caco2_cm_per_s_to_log10([value])
