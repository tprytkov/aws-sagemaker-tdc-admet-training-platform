import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from admet_platform.config import load_endpoint_config
from admet_platform.data import classification_acquisition
from admet_platform.data.classification_acquisition import (
    acquire_and_audit_binary_tdc_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
CANDIDATE_CONFIGS = {
    "hia_hou.yaml": ("hia_hou", "HIA_Hou"),
    "pgp_broccatelli.yaml": ("pgp_broccatelli", "Pgp_Broccatelli"),
    "cyp1a2_veith.yaml": ("cyp1a2_veith", "CYP1A2_Veith"),
    "cyp2c19_veith.yaml": ("cyp2c19_veith", "CYP2C19_Veith"),
    "cyp2c9_veith.yaml": ("cyp2c9_veith", "CYP2C9_Veith"),
    "cyp2d6_veith.yaml": ("cyp2d6_veith", "CYP2D6_Veith"),
    "cyp3a4_veith.yaml": ("cyp3a4_veith", "CYP3A4_Veith"),
}


def test_candidate_configs_use_exact_adme_binary_dataset_names() -> None:
    for filename, (endpoint_id, tdc_name) in CANDIDATE_CONFIGS.items():
        config = load_endpoint_config(CONFIG_DIR / filename)
        assert config.endpoint_id == endpoint_id
        assert config.tdc_name == tdc_name
        assert config.task_group == "ADME"
        assert config.task_type == "binary_classification"
        assert config.split_strategy == "scaffold"


def test_candidate_names_match_pytdc_adme_registry() -> None:
    pytest.importorskip("tdc")
    from tdc.metadata import adme_dataset_names

    configured_registry_names = {
        load_endpoint_config(CONFIG_DIR / filename).tdc_name.lower()
        for filename in CANDIDATE_CONFIGS
    }
    assert configured_registry_names <= set(adme_dataset_names)


def test_candidate_configs_record_label_direction() -> None:
    hia = load_endpoint_config(CONFIG_DIR / "hia_hou.yaml")
    pgp = load_endpoint_config(CONFIG_DIR / "pgp_broccatelli.yaml")
    assert "label 1 is good absorption" in hia.problem_description
    assert "above 30%" in hia.problem_description
    assert "label 1 is inhibitor" in pgp.problem_description
    assert any("not substrate status" in limitation for limitation in pgp.limitations)

    for filename in CANDIDATE_CONFIGS:
        if filename.startswith("cyp"):
            config = load_endpoint_config(CONFIG_DIR / filename)
            assert "label 1 is inhibitor" in config.problem_description
            assert any("not CYP" in limitation for limitation in config.limitations)


def test_acquisition_audit_retains_rows_and_writes_quarantines(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw = pd.DataFrame(
        {
            "Drug_ID": [f"mol_{index}" for index in range(7)],
            "Drug": ["CCO", "OCC", "CCN", "NCC", "not smiles", "CCC", "CCCC"],
            "Y": [1, 1, 0, 1, 1, None, 2],
        }
    )
    monkeypatch.setattr(classification_acquisition, "load_tdc_data", lambda config: raw)
    monkeypatch.setattr(
        classification_acquisition,
        "_verify_adme_registry_name",
        lambda name: name.lower(),
    )
    monkeypatch.setattr(
        classification_acquisition.importlib.metadata,
        "version",
        lambda distribution: "0.3.9",
    )

    audit = acquire_and_audit_binary_tdc_dataset(
        CONFIG_DIR / "hia_hou.yaml",
        tmp_path,
    )

    normalized_path = tmp_path / "normalized.csv"
    normalized = pd.read_csv(normalized_path)
    written_audit = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert len(normalized) == len(raw)
    assert list(normalized.columns) == ["molecule_id", "smiles", "target"]
    assert "split" not in normalized.columns
    assert audit == written_audit
    assert audit["raw_columns"] == ["Drug_ID", "Drug", "Y"]
    assert audit["raw_row_count"] == 7
    assert audit["normalized_row_count"] == 7
    assert audit["unique_raw_label_values"] == [0.0, 1.0, 2.0, None]
    assert audit["class_counts"] == {"0": 1, "1": 4}
    assert audit["missing_label_count"] == 1
    assert audit["invalid_label_count"] == 1
    assert audit["invalid_smiles_count"] == 1
    assert audit["exact_duplicate_count"] == 1
    assert audit["exact_duplicate_group_count"] == 1
    assert audit["conflicting_label_duplicate_count"] == 1
    assert audit["conflicting_label_duplicate_row_count"] == 2
    assert audit["rows_discarded"] == 0
    assert audit["split_status"] == "unsplit"
    assert audit["normalized_csv_sha256"] == hashlib.sha256(
        normalized_path.read_bytes()
    ).hexdigest()

    invalid = pd.read_csv(tmp_path / "quarantined_invalid_smiles.csv")
    conflicts = pd.read_csv(tmp_path / "quarantined_conflicting_labels.csv")
    duplicates = pd.read_csv(tmp_path / "exact_duplicates.csv")
    assert invalid["smiles"].tolist() == ["not smiles"]
    assert set(conflicts["smiles"]) == {"CCN", "NCC"}
    assert set(duplicates["smiles"]) == {"CCO", "OCC"}
    assert not any(tmp_path.glob("train.csv"))
    assert not any(tmp_path.glob("validation.csv"))
    assert not any(tmp_path.glob("test.csv"))


def test_acquisition_rejects_unverified_pytdc_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        classification_acquisition.importlib.metadata,
        "version",
        lambda distribution: "0.3.8",
    )
    with pytest.raises(RuntimeError, match="Expected PyTDC==0.3.9"):
        acquire_and_audit_binary_tdc_dataset(
            CONFIG_DIR / "hia_hou.yaml",
            tmp_path,
        )


def test_registry_verification_rejects_non_adme_name(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("tdc")
    with pytest.raises(ValueError, match="not registered"):
        classification_acquisition._verify_adme_registry_name("not_a_real_dataset")
