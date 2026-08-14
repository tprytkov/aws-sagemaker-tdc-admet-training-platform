import pandas as pd

from admet_platform.chemprop.applicability import applicability_domain
from admet_platform.chemprop.baselines import run_classification_baselines, run_regression_baselines


def _frame(targets: list[float]) -> pd.DataFrame:
    smiles = ["CCO", "CCN", "CCC", "CCCl", "c1ccccc1", "c1ccncc1", "C1CCCCC1", "C1CCNCC1"]
    return pd.DataFrame({"molecule_id": range(8), "canonical_smiles": smiles, "target": targets})


def test_inexpensive_baseline_sets() -> None:
    train_reg = _frame([-0.1, -0.5, -1.0, -1.2, -2.0, -2.2, -2.5, -2.7])
    valid_reg = train_reg.iloc[:4].copy()
    regression = run_regression_baselines(train_reg, valid_reg)
    assert set(regression) == {"training_median", "esol", "morgan_ridge", "rdkit_descriptor_ridge"}
    train_cls = _frame([0, 1, 0, 1, 0, 1, 0, 1])
    valid_cls = train_cls.iloc[:6].copy()
    classification = run_classification_baselines(train_cls, valid_cls)
    assert set(classification) == {"majority", "prevalence", "morgan_logistic", "rdkit_descriptor_logistic"}


def test_applicability_reports_similarity_and_scaffold_familiarity() -> None:
    result = applicability_domain(["c1ccccc1", "CCO"], ["c1ccccc1", "C1CCCCC1"])
    assert result.loc[0, "maximum_train_tanimoto"] == 1.0
    assert bool(result.loc[0, "scaffold_in_training"])
    assert set(result["applicability_domain"]) <= {"in_domain", "out_of_domain"}
