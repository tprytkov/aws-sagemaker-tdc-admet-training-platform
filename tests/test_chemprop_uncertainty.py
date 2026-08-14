import pandas as pd
import pytest

from admet_platform.chemprop.uncertainty import aggregate_endpoint_seed_predictions


def test_endpoint_ensemble_uncertainty_requires_identical_membership() -> None:
    first = pd.DataFrame({
        "molecule_id": ["a", "b"],
        "canonical_smiles": ["CC", "CCC"],
        "target": [1.0, 2.0],
        "prediction": [1.1, 1.8],
    })
    second = first.assign(prediction=[0.9, 2.2])
    result = aggregate_endpoint_seed_predictions({13: first, 37: second})
    assert result["ensemble_mean"].tolist() == pytest.approx([1.0, 2.0])
    assert (result["seed_count"] == 2).all()
    with pytest.raises(ValueError, match="membership or labels differ"):
        aggregate_endpoint_seed_predictions({13: first, 37: second.assign(target=[1.0, 3.0])})
