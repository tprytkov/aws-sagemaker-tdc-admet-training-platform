from __future__ import annotations

import hashlib
import json
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, BertConfig, BertModel

import admet_platform.models.chemberta_production_inference as inference
from admet_platform.models.chemberta_production_inference import (
    ChemBERTaProductionInferenceError,
    ChemBERTaProductionPredictor,
    MODEL_ENDPOINT_ORDER,
    PRODUCTION_ENDPOINT_ORDER,
    predict_chemberta_batch,
    verify_bundle_hashes,
)
from admet_platform.models.multitask_chemberta import (
    MultiTaskChemBERTa,
    MultiTaskChemBERTaConfig,
)
from admet_platform.training.multitask_calibration import apply_platt_calibration


CHECKPOINT_SHA256 = "a" * 64


@pytest.fixture()
def frozen_bundle(tmp_path: Path, tiny_model_tokenizer_dir: Path) -> Path:
    bundle = tmp_path / inference.BUNDLE_VERSION
    tokenizer_dir = bundle / "tokenizer"
    tokenizer_dir.mkdir(parents=True)
    source_tokenizer = AutoTokenizer.from_pretrained(
        tiny_model_tokenizer_dir, local_files_only=True
    )
    source_tokenizer.save_pretrained(tokenizer_dir)

    torch.manual_seed(23)
    encoder = BertModel(
        BertConfig(
            vocab_size=len(source_tokenizer),
            hidden_size=12,
            num_hidden_layers=1,
            num_attention_heads=3,
            intermediate_size=16,
            max_position_embeddings=160,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
    )
    model = MultiTaskChemBERTa(
        MultiTaskChemBERTaConfig(
            model_name_or_path="packaged-only",
            tasks=MODEL_ENDPOINT_ORDER,
            pooling="masked_mean",
            dropout=0.15,
            local_files_only=True,
        ),
        encoder=encoder,
    )
    model.save_model(bundle / "model")

    endpoint_items = {}
    calibration_items = {}
    for index, endpoint in enumerate(MODEL_ENDPOINT_ORDER):
        is_hia = endpoint == "hia_hou"
        endpoint_items[endpoint] = {
            "display_name": endpoint,
            "positive_class": f"positive {endpoint}",
            "evidence_status": f"evidence_{index}",
            "calibration": "identity" if is_hia else "platt",
        }
        calibration_items[endpoint] = {
            "endpoint": endpoint,
            "calibration_method": "platt_scaling",
            "coefficient_a": None if is_hia else 1.25 + index / 100,
            "intercept_b": None if is_hia else -0.2 + index / 100,
            "fit_status": "insufficient_class_support" if is_hia else "fitted",
            "source_split": "validation",
            "checkpoint": {"path": "private-source-not-opened", "sha256": CHECKPOINT_SHA256},
        }
    _write_json(
        bundle / "endpoint_metadata.json",
        {
            "schema_version": "1.0.0",
            "endpoint_order": list(MODEL_ENDPOINT_ORDER),
            "endpoints": endpoint_items,
        },
    )
    calibration_path = bundle / "calibration" / "calibration_parameters.json"
    _write_json(
        calibration_path,
        {
            "schema_version": "1.0.0",
            "source_split": "validation",
            "endpoint_order": list(MODEL_ENDPOINT_ORDER),
            "endpoints": calibration_items,
        },
    )
    _write_json(
        bundle / "model_manifest.json",
        {
            "schema_version": "1.0.0",
            "bundle_version": inference.BUNDLE_VERSION,
            "model_family": inference.MODEL_FAMILY,
            "base_model": "packaged-only",
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "endpoint_order": list(MODEL_ENDPOINT_ORDER),
            "pooling": "masked_mean",
            "dropout": 0.15,
            "max_sequence_length": 128,
            "descriptive_threshold": 0.5,
            "threshold_optimized_on_test": False,
            "calibration_source": "validation",
            "calibration_parameters_sha256": _digest(calibration_path),
            "test_data_included": False,
            "training_state_included": False,
        },
    )
    _write_sums(bundle)
    return bundle


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_sums(bundle: Path) -> None:
    lines = [
        f"{_digest(bundle / Path(*relative.split('/')))}  "
        f"exports/{bundle.name}/{relative}"
        for relative in inference._HASHED_FILES
    ]
    (bundle / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_bundle_hash_verification_rejects_tampering(frozen_bundle: Path) -> None:
    hashes = verify_bundle_hashes(frozen_bundle)
    assert tuple(hashes) == inference._HASHED_FILES
    (frozen_bundle / "endpoint_metadata.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(ChemBERTaProductionInferenceError, match="SHA-256 mismatch"):
        verify_bundle_hashes(frozen_bundle)


def test_bbb_is_suppressed_and_nine_endpoint_order_is_stable(frozen_bundle: Path) -> None:
    output = predict_chemberta_batch(
        [{"molecule_id": "ethanol", "source_smiles": "OCC"}],
        bundle_path=frozen_bundle,
        device="cpu",
        batch_size=1,
    )[0]
    assert output["endpoint_order"] == list(PRODUCTION_ENDPOINT_ORDER)
    assert len(output["endpoints"]) == 9
    assert list(output["endpoints"]) == list(PRODUCTION_ENDPOINT_ORDER)
    assert "bbb_martins" not in output["endpoints"]
    assert output["canonical_smiles"] == "CCO"


def test_identity_hia_calibration_and_threshold_exactly_point_five() -> None:
    predictor = object.__new__(ChemBERTaProductionPredictor)
    predictor.calibration = {"endpoints": {}}
    predictor.endpoint_metadata = {
        "endpoints": {
            "hia_hou": {
                "evidence_status": "limited_support",
                "positive_class": "favorable intestinal absorption",
            }
        }
    }
    output = predictor._endpoint_output("hia_hou", np.float64(0.0))
    assert output["raw_probability"] == 0.5
    assert output["calibrated_probability"] == output["raw_probability"]
    assert output["predicted_class"] == 1
    assert output["calibration_method"] == "identity"


def test_frozen_platt_calibration_reuses_existing_implementation() -> None:
    parameters = {
        "fit_status": "fitted",
        "coefficient_a": 1.7,
        "intercept_b": -0.4,
    }
    predictor = object.__new__(ChemBERTaProductionPredictor)
    predictor.calibration = {"endpoints": {"ames": parameters}}
    predictor.endpoint_metadata = {
        "endpoints": {
            "ames": {"evidence_status": "moderate", "positive_class": "mutagenicity"}
        }
    }
    output = predictor._endpoint_output("ames", np.float64(0.25))
    expected = apply_platt_calibration([0.25], parameters)[0]
    assert output["calibrated_probability"] == pytest.approx(expected)
    assert output["calibration_method"] == "platt_scaling"


def test_invalid_smiles_isolated_and_has_no_predictions(frozen_bundle: Path) -> None:
    outputs = predict_chemberta_batch(
        [
            {"molecule_id": "valid-a", "source_smiles": "CCO"},
            {"molecule_id": "invalid", "source_smiles": "C("},
            {"molecule_id": "valid-b", "source_smiles": "C[C@H](O)Cl"},
        ],
        bundle_path=frozen_bundle,
        device="cpu",
        batch_size=2,
    )
    assert [row["molecule_id"] for row in outputs] == ["valid-a", "invalid", "valid-b"]
    assert [row["status"] for row in outputs] == ["success", "failed", "success"]
    assert outputs[1]["error_code"] == "invalid_smiles"
    assert outputs[1]["canonical_smiles"] is None
    assert outputs[1]["endpoints"] == {}


def test_inference_is_deterministic(frozen_bundle: Path) -> None:
    rows = [{"molecule_id": "x", "source_smiles": "c1ccccc1"}]
    first = predict_chemberta_batch(rows, bundle_path=frozen_bundle, device="cpu")
    second = predict_chemberta_batch(rows, bundle_path=frozen_bundle, device="cpu")
    assert first == second


def test_model_state_loading_is_strict(frozen_bundle: Path) -> None:
    state_path = frozen_bundle / "model" / "model_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    state.pop(next(iter(state)))
    torch.save(state, state_path)
    _write_sums(frozen_bundle)
    with pytest.raises(ChemBERTaProductionInferenceError, match="load.*strictly"):
        ChemBERTaProductionPredictor(bundle_path=frozen_bundle, device="cpu")


def test_all_loading_is_offline(frozen_bundle: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "create_connection", reject_network)
    output = predict_chemberta_batch(
        [{"molecule_id": "x", "source_smiles": "CCN"}],
        bundle_path=frozen_bundle,
        device="cpu",
    )
    assert output[0]["status"] == "success"


def test_cli_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "existing.csv"
    output.write_text("existing\n", encoding="utf-8")
    script = Path(__file__).parents[1] / "scripts" / "predict_chemberta_classification.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--bundle-path",
            str(tmp_path / "missing-bundle"),
            "--input-csv",
            str(tmp_path / "missing-input.csv"),
            "--output-csv",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Refusing to overwrite existing output" in result.stderr
    assert output.read_text(encoding="utf-8") == "existing\n"


def test_csv_contract_contains_only_production_endpoint_columns() -> None:
    script_path = Path(__file__).parents[1] / "scripts"
    sys.path.insert(0, str(script_path))
    try:
        import predict_chemberta_classification as cli

        fields = cli._fieldnames()
    finally:
        sys.path.remove(str(script_path))
    assert "endpoint_order_json" in fields
    assert any(field.startswith("hia_hou_") for field in fields)
    assert not any(field.startswith("bbb_martins_") for field in fields)
