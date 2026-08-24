"""Fail-closed batch inference for the frozen MolOptima ChemBERTa bundle."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from rdkit import Chem
from transformers import AutoTokenizer

from admet_platform.models.multitask_chemberta import MultiTaskChemBERTa
from admet_platform.training.multitask_calibration import (
    apply_platt_calibration,
    sigmoid,
)


BUNDLE_SCHEMA_VERSION = "1.0.0"
BUNDLE_VERSION = "moloptima_admet_classifier_v1"
MODEL_FAMILY = "multitask_chemberta_binary_classifier"
MODEL_ENDPOINT_ORDER = (
    "hia_hou",
    "pgp_broccatelli",
    "bbb_martins",
    "cyp1a2_veith",
    "cyp2c19_veith",
    "cyp2c9_veith",
    "cyp2d6_veith",
    "cyp3a4_veith",
    "herg_karim",
    "ames",
)
PRODUCTION_ENDPOINT_ORDER = tuple(
    endpoint for endpoint in MODEL_ENDPOINT_ORDER if endpoint != "bbb_martins"
)
MAX_SEQUENCE_LENGTH = 128
DECISION_THRESHOLD = 0.5

_HASHED_FILES = (
    "calibration/calibration_parameters.json",
    "endpoint_metadata.json",
    "model/encoder_config/config.json",
    "model/model_state.pt",
    "model/multitask_model_config.json",
    "model_manifest.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
)


class ChemBERTaProductionInferenceError(RuntimeError):
    """The frozen bundle or model violated its production inference contract."""


def predict_chemberta_batch(
    inputs: Iterable[Mapping[str, object]],
    *,
    bundle_path: str | Path,
    device: str | torch.device | None = None,
    batch_size: int = 32,
) -> list[dict[str, object]]:
    """Predict an ordered batch using only the packaged frozen classifier bundle."""

    predictor = ChemBERTaProductionPredictor(
        bundle_path=bundle_path,
        device=device,
        batch_size=batch_size,
    )
    return predictor.predict(inputs)


class ChemBERTaProductionPredictor:
    """Loaded, reusable view of one verified frozen classifier bundle."""

    def __init__(
        self,
        *,
        bundle_path: str | Path,
        device: str | torch.device | None = None,
        batch_size: int = 32,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        self.bundle_path = Path(bundle_path).resolve()
        self.batch_size = batch_size
        self.device = _resolve_device(device)

        # Integrity is intentionally checked before any bundle JSON or model file is loaded.
        self.file_hashes = verify_bundle_hashes(self.bundle_path)
        self.manifest = _load_json(self.bundle_path / "model_manifest.json", "model manifest")
        self.endpoint_metadata = _load_json(
            self.bundle_path / "endpoint_metadata.json", "endpoint metadata"
        )
        self.calibration = _load_json(
            self.bundle_path / "calibration" / "calibration_parameters.json",
            "calibration parameters",
        )
        model_config = _load_json(
            self.bundle_path / "model" / "multitask_model_config.json",
            "multi-task model configuration",
        )
        _verify_bundle_contract(
            self.bundle_path,
            self.manifest,
            self.endpoint_metadata,
            self.calibration,
            model_config,
            self.file_hashes,
        )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.bundle_path / "tokenizer", local_files_only=True
            )
            self.model = MultiTaskChemBERTa.load_model(
                self.bundle_path / "model", map_location="cpu"
            )
            self.model.to(self.device)
            self.model.eval()
        except Exception as exc:
            raise ChemBERTaProductionInferenceError(
                "Unable to load the frozen ChemBERTa model strictly from the bundle."
            ) from exc

    def predict(self, inputs: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
        """Return one output per input row while isolating invalid SMILES rows."""

        rows = list(inputs)
        validated = [_validate_input_row(row, index) for index, row in enumerate(rows)]
        valid_positions = [
            index for index, row in enumerate(validated) if row["status"] == "success"
        ]
        endpoint_logits: dict[str, np.ndarray] = {
            endpoint: np.empty(len(valid_positions), dtype=np.float64)
            for endpoint in PRODUCTION_ENDPOINT_ORDER
        }

        try:
            with torch.no_grad():
                for start in range(0, len(valid_positions), self.batch_size):
                    positions = valid_positions[start : start + self.batch_size]
                    smiles = [str(validated[position]["canonical_smiles"]) for position in positions]
                    encoded = self.tokenizer(
                        smiles,
                        max_length=MAX_SEQUENCE_LENGTH,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )
                    input_ids = encoded["input_ids"].to(self.device)
                    attention_mask = encoded["attention_mask"].to(self.device)
                    for endpoint in PRODUCTION_ENDPOINT_ORDER:
                        values = self.model(input_ids, attention_mask, endpoint)
                        values_array = values.detach().to(device="cpu", dtype=torch.float64).numpy()
                        if values_array.shape != (len(positions),):
                            raise ValueError(
                                f"Endpoint {endpoint} returned shape {values_array.shape}, "
                                f"expected {(len(positions),)}."
                            )
                        endpoint_logits[endpoint][start : start + len(positions)] = values_array
        except Exception as exc:
            raise ChemBERTaProductionInferenceError(
                "ChemBERTa batch inference failed; no partial model result is safe to use."
            ) from exc

        if any(not np.isfinite(values).all() for values in endpoint_logits.values()):
            raise ChemBERTaProductionInferenceError(
                "ChemBERTa batch inference produced a non-finite raw logit."
            )

        outputs: list[dict[str, object]] = []
        valid_index = 0
        for row in validated:
            if row["status"] != "success":
                outputs.append(self._failed_output(row))
                continue
            predictions = {
                endpoint: self._endpoint_output(endpoint, endpoint_logits[endpoint][valid_index])
                for endpoint in PRODUCTION_ENDPOINT_ORDER
            }
            outputs.append(self._successful_output(row, predictions))
            valid_index += 1
        return outputs

    def _endpoint_output(self, endpoint: str, raw_logit: np.float64) -> dict[str, object]:
        logits = np.asarray([raw_logit], dtype=np.float64)
        raw_probability = float(sigmoid(logits)[0])
        if endpoint == "hia_hou":
            calibrated_probability = raw_probability
            calibration_method = "identity"
        else:
            parameters = self.calibration["endpoints"][endpoint]
            calibrated_probability = float(apply_platt_calibration(logits, parameters)[0])
            calibration_method = "platt_scaling"
        metadata = self.endpoint_metadata["endpoints"][endpoint]
        return {
            "raw_logit": float(raw_logit),
            "raw_probability": raw_probability,
            "calibrated_probability": calibrated_probability,
            "predicted_class": int(calibrated_probability >= DECISION_THRESHOLD),
            "calibration_method": calibration_method,
            "evidence_status": metadata["evidence_status"],
            "positive_class": metadata["positive_class"],
        }

    def _common_output(self, row: Mapping[str, object]) -> dict[str, object]:
        return {
            "molecule_id": row.get("molecule_id"),
            "source_smiles": row.get("source_smiles"),
            "canonical_smiles": row.get("canonical_smiles"),
            "model_family": self.manifest["model_family"],
            "bundle_version": self.manifest["bundle_version"],
            "checkpoint_sha256": self.manifest["checkpoint_sha256"],
            "endpoint_order": list(PRODUCTION_ENDPOINT_ORDER),
        }

    def _successful_output(
        self,
        row: Mapping[str, object],
        predictions: Mapping[str, Mapping[str, object]],
    ) -> dict[str, object]:
        return {
            **self._common_output(row),
            "status": "success",
            "error_code": None,
            "error_message": None,
            "endpoints": dict(predictions),
        }

    def _failed_output(self, row: Mapping[str, object]) -> dict[str, object]:
        return {
            **self._common_output(row),
            "canonical_smiles": None,
            "status": "failed",
            "error_code": row.get("error_code"),
            "error_message": row.get("error_message"),
            "endpoints": {},
        }


def verify_bundle_hashes(bundle_path: str | Path) -> dict[str, str]:
    """Verify the exact frozen file inventory recorded by ``SHA256SUMS.txt``."""

    bundle = Path(bundle_path).resolve()
    sums_path = bundle / "SHA256SUMS.txt"
    if not bundle.is_dir():
        raise ChemBERTaProductionInferenceError(f"Bundle directory does not exist: {bundle}")
    if not sums_path.is_file():
        raise ChemBERTaProductionInferenceError(f"Bundle checksum file is missing: {sums_path}")

    recorded: dict[str, str] = {}
    try:
        lines = sums_path.read_text(encoding="utf-8-sig").splitlines()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            parts = line.split(None, 1)
            if len(parts) != 2 or len(parts[0]) != 64:
                raise ValueError(f"invalid checksum line {line_number}")
            digest, declared_path = parts[0].lower(), parts[1].strip().lstrip("*")
            if any(character not in "0123456789abcdef" for character in digest):
                raise ValueError(f"invalid SHA-256 on line {line_number}")
            relative = _bundle_relative_checksum_path(bundle.name, declared_path)
            if relative in recorded:
                raise ValueError(f"duplicate checksum entry for {relative}")
            recorded[relative] = digest
    except (OSError, UnicodeError, ValueError) as exc:
        raise ChemBERTaProductionInferenceError(
            f"Invalid bundle checksum file: {sums_path}"
        ) from exc

    if set(recorded) != set(_HASHED_FILES):
        missing = sorted(set(_HASHED_FILES) - set(recorded))
        unexpected = sorted(set(recorded) - set(_HASHED_FILES))
        raise ChemBERTaProductionInferenceError(
            f"Bundle checksum inventory mismatch; missing={missing}, unexpected={unexpected}."
        )
    for relative in _HASHED_FILES:
        path = bundle / Path(*PurePosixPath(relative).parts)
        if not path.is_file():
            raise ChemBERTaProductionInferenceError(f"Hashed bundle file is missing: {relative}")
        actual = _sha256_file(path)
        if actual != recorded[relative]:
            raise ChemBERTaProductionInferenceError(
                f"SHA-256 mismatch for bundle file: {relative}"
            )
    return recorded


def _bundle_relative_checksum_path(bundle_name: str, declared_path: str) -> str:
    path = PurePosixPath(declared_path.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("checksum path must be portable and remain inside the bundle")
    parts = path.parts
    if bundle_name in parts:
        index = len(parts) - 1 - tuple(reversed(parts)).index(bundle_name)
        parts = parts[index + 1 :]
    relative = PurePosixPath(*parts).as_posix()
    if not relative or relative == ".":
        raise ValueError("checksum path does not identify a file")
    return relative


def _verify_bundle_contract(
    bundle: Path,
    manifest: Mapping[str, Any],
    metadata: Mapping[str, Any],
    calibration: Mapping[str, Any],
    model_config: Mapping[str, Any],
    hashes: Mapping[str, str],
) -> None:
    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ChemBERTaProductionInferenceError(message)

    require(manifest.get("schema_version") == BUNDLE_SCHEMA_VERSION, "Unsupported schema version.")
    require(manifest.get("bundle_version") == BUNDLE_VERSION, "Bundle version differs.")
    require(bundle.name == manifest["bundle_version"], "Bundle directory/version identity differs.")
    require(manifest.get("model_family") == MODEL_FAMILY, "Model family differs.")
    require(tuple(manifest.get("endpoint_order", ())) == MODEL_ENDPOINT_ORDER, "Endpoint order differs.")
    require(manifest.get("max_sequence_length") == MAX_SEQUENCE_LENGTH, "Sequence length differs.")
    require(manifest.get("descriptive_threshold") == DECISION_THRESHOLD, "Threshold differs.")
    require(manifest.get("threshold_optimized_on_test") is False, "Threshold contract differs.")
    require(manifest.get("calibration_source") == "validation", "Calibration source differs.")
    require(manifest.get("test_data_included") is False, "Bundle must not contain test data.")
    require(manifest.get("training_state_included") is False, "Bundle contains training state.")
    checkpoint_sha = manifest.get("checkpoint_sha256")
    require(
        isinstance(checkpoint_sha, str)
        and len(checkpoint_sha) == 64
        and all(character in "0123456789abcdef" for character in checkpoint_sha.lower()),
        "Checkpoint SHA-256 identity is invalid.",
    )
    require(
        manifest.get("calibration_parameters_sha256")
        == hashes["calibration/calibration_parameters.json"],
        "Calibration SHA-256 identity differs.",
    )

    require(metadata.get("schema_version") == BUNDLE_SCHEMA_VERSION, "Metadata schema differs.")
    require(tuple(metadata.get("endpoint_order", ())) == MODEL_ENDPOINT_ORDER, "Metadata order differs.")
    endpoint_metadata = metadata.get("endpoints")
    require(
        isinstance(endpoint_metadata, dict) and tuple(endpoint_metadata) == MODEL_ENDPOINT_ORDER,
        "Metadata endpoint mapping differs.",
    )
    for endpoint in MODEL_ENDPOINT_ORDER:
        item = endpoint_metadata[endpoint]
        require(isinstance(item, dict), f"Metadata is invalid for {endpoint}.")
        require(
            isinstance(item.get("positive_class"), str) and bool(item["positive_class"]),
            f"Positive class is missing for {endpoint}.",
        )
        require(
            isinstance(item.get("evidence_status"), str) and bool(item["evidence_status"]),
            f"Evidence status is missing for {endpoint}.",
        )
        expected_method = "identity" if endpoint == "hia_hou" else "platt"
        require(item.get("calibration") == expected_method, f"Calibration metadata differs for {endpoint}.")

    require(calibration.get("schema_version") == BUNDLE_SCHEMA_VERSION, "Calibration schema differs.")
    require(calibration.get("source_split") == "validation", "Calibration split differs.")
    require(
        tuple(calibration.get("endpoint_order", ())) == MODEL_ENDPOINT_ORDER,
        "Calibration endpoint order differs.",
    )
    calibration_endpoints = calibration.get("endpoints")
    require(
        isinstance(calibration_endpoints, dict)
        and tuple(calibration_endpoints) == MODEL_ENDPOINT_ORDER,
        "Calibration endpoint mapping differs.",
    )
    for endpoint in MODEL_ENDPOINT_ORDER:
        item = calibration_endpoints[endpoint]
        require(isinstance(item, dict) and item.get("endpoint") == endpoint, f"Calibration identity differs for {endpoint}.")
        require(item.get("calibration_method") == "platt_scaling", f"Calibration method differs for {endpoint}.")
        if endpoint == "hia_hou":
            require(item.get("fit_status") != "fitted", "HIA must use identity calibration.")
        else:
            require(item.get("fit_status") == "fitted", f"Frozen Platt fit is missing for {endpoint}.")
            for key in ("coefficient_a", "intercept_b"):
                value = item.get(key)
                require(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value)),
                    f"Frozen Platt parameter {key} is invalid for {endpoint}.",
                )
        item_checkpoint = item.get("checkpoint")
        require(
            isinstance(item_checkpoint, dict)
            and item_checkpoint.get("sha256") == checkpoint_sha,
            f"Calibration checkpoint identity differs for {endpoint}.",
        )

    require(tuple(model_config.get("tasks", ())) == MODEL_ENDPOINT_ORDER, "Model task order differs.")
    require(
        isinstance(manifest.get("base_model"), str)
        and model_config.get("model_name_or_path") == manifest.get("base_model"),
        "Base-model identity differs.",
    )
    require(model_config.get("pooling") == manifest.get("pooling") == "masked_mean", "Pooling differs.")
    require(model_config.get("dropout") == manifest.get("dropout") == 0.15, "Dropout differs.")
    require(model_config.get("head_type") == "linear", "Head type differs.")
    require(model_config.get("head_output_size") == 1, "Head output size differs.")
    require(model_config.get("local_files_only") is True, "Model is not offline-only.")


def _resolve_device(requested: str | torch.device | None) -> torch.device:
    if requested is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        resolved = torch.device(requested)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"Invalid torch device: {requested}") from exc
    if resolved.type == "cuda":
        requested_index = resolved.index if resolved.index is not None else 0
        if not torch.cuda.is_available() or requested_index >= torch.cuda.device_count():
            return torch.device("cpu")
    return resolved


def _validate_input_row(row: Mapping[str, object], index: int) -> dict[str, object]:
    if not isinstance(row, Mapping):
        return _invalid_row(None, None, "invalid_row", f"Input row {index} is not a mapping.")
    molecule_id = row.get("molecule_id")
    source_smiles = row.get("source_smiles")
    if not isinstance(molecule_id, str) or not molecule_id:
        return _invalid_row(
            molecule_id,
            source_smiles,
            "invalid_molecule_id",
            "molecule_id must be a non-empty string.",
        )
    if not isinstance(source_smiles, str) or not source_smiles.strip():
        return _invalid_row(
            molecule_id,
            source_smiles,
            "invalid_smiles",
            "source_smiles must be a non-empty string.",
        )
    molecule = Chem.MolFromSmiles(source_smiles)
    if molecule is None:
        return _invalid_row(
            molecule_id,
            source_smiles,
            "invalid_smiles",
            "RDKit could not parse source_smiles.",
        )
    return {
        "molecule_id": molecule_id,
        "source_smiles": source_smiles,
        "canonical_smiles": Chem.MolToSmiles(
            molecule, canonical=True, isomericSmiles=True
        ),
        "status": "success",
    }


def _invalid_row(
    molecule_id: object,
    source_smiles: object,
    code: str,
    message: str,
) -> dict[str, object]:
    return {
        "molecule_id": molecule_id,
        "source_smiles": source_smiles,
        "canonical_smiles": None,
        "status": "failed",
        "error_code": code,
        "error_message": message,
    }


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ChemBERTaProductionInferenceError(f"Unable to read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ChemBERTaProductionInferenceError(f"{label.capitalize()} must contain a JSON object.")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "ChemBERTaProductionInferenceError",
    "ChemBERTaProductionPredictor",
    "DECISION_THRESHOLD",
    "MODEL_ENDPOINT_ORDER",
    "PRODUCTION_ENDPOINT_ORDER",
    "predict_chemberta_batch",
    "verify_bundle_hashes",
]
