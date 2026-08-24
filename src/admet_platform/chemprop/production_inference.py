"""Fail-closed batch inference for the frozen five-endpoint Chemprop release."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from rdkit import Chem

from admet_platform.chemprop.production_manifest import (
    CHECKPOINT_SHA256,
    ENDPOINT_ORDER,
    PRODUCTION_SEEDS,
    RELEASE_STATUS,
    SCHEMA_VERSION,
    UNCERTAINTY_INTERPRETATION,
)
from admet_platform.chemprop.units import caco2_log10_to_cm_per_s


class ProductionInferenceContractError(RuntimeError):
    """A release-wide invariant failed, so no predictions may be returned."""


class SeedPredictionBackend(Protocol):
    """Backend contract; returned values have already passed the checkpoint unscaler once."""

    def prepare(self, canonical_smiles: Sequence[str], *, num_workers: int) -> list[int]: ...

    def predict_checkpoint(
        self,
        checkpoint_path: Path,
        *,
        scaler_mean: np.ndarray,
        scaler_scale: np.ndarray,
    ) -> np.ndarray: ...

    def close(self) -> None: ...


class ChempropRegressionPredictor:
    """Validated predictor for one immutable production manifest and its five checkpoints."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        artifact_root: str | Path | None = None,
        verify_runtime: bool = True,
        num_workers: int = 0,
        _backend_factory: Callable[[], SeedPredictionBackend] | None = None,
    ) -> None:
        if num_workers < 0:
            raise ValueError("num_workers must be non-negative.")
        self.manifest_path = Path(manifest_path).resolve()
        self.artifact_root = (
            Path(artifact_root).resolve()
            if artifact_root is not None
            else _infer_artifact_root(self.manifest_path)
        )
        self.num_workers = num_workers
        self._backend_factory = _backend_factory or _ChempropBackend
        self.manifest, self.manifest_sha256 = _load_and_validate_manifest(
            self.manifest_path, self.artifact_root, verify_runtime=verify_runtime
        )
        self._checkpoints = _verify_checkpoints(self.manifest, self.artifact_root)
        scaler = self.manifest["target_scaler"]["per_endpoint"]
        self._scaler_mean = np.asarray(
            [float(scaler[name]["mean"]) for name in ENDPOINT_ORDER], dtype=float
        )
        self._scaler_scale = np.asarray(
            [float(scaler[name]["scale"]) for name in ENDPOINT_ORDER], dtype=float
        )

    def predict(self, inputs: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
        """Predict rows in input order; invalid molecules fail independently."""

        rows = [_validate_input_row(row, index) for index, row in enumerate(inputs)]
        valid_indices = [index for index, row in enumerate(rows) if row["status"] == "valid"]
        if not valid_indices:
            return [_failed_output(row, self.manifest, self.manifest_sha256) for row in rows]

        backend = self._backend_factory()
        seed_predictions: dict[int, np.ndarray] = {}
        try:
            preprocessing_failures = backend.prepare(
                [str(rows[index]["canonical_smiles"]) for index in valid_indices],
                num_workers=self.num_workers,
            )
            if len(set(preprocessing_failures)) != len(preprocessing_failures) or any(
                index < 0 or index >= len(valid_indices) for index in preprocessing_failures
            ):
                raise ProductionInferenceContractError(
                    "The prediction backend returned invalid preprocessing failures."
                )
            failed_global_indices = {valid_indices[index] for index in preprocessing_failures}
            for index in failed_global_indices:
                rows[index].update(
                    {
                        "status": "invalid",
                        "canonical_smiles": None,
                        "error_code": "molecule_preprocessing_failed",
                        "error_message": "Chemprop could not featurize this molecule.",
                    }
                )
            valid_indices = [index for index in valid_indices if index not in failed_global_indices]
            if not valid_indices:
                return [_failed_output(row, self.manifest, self.manifest_sha256) for row in rows]
            for seed, checkpoint_path in self._checkpoints:
                predicted = np.asarray(
                    backend.predict_checkpoint(
                        checkpoint_path,
                        scaler_mean=self._scaler_mean.copy(),
                        scaler_scale=self._scaler_scale.copy(),
                    ),
                    dtype=float,
                )
                expected_shape = (len(valid_indices), len(ENDPOINT_ORDER))
                if predicted.shape != expected_shape or not np.isfinite(predicted).all():
                    raise ProductionInferenceContractError(
                        f"Seed {seed} returned invalid prediction shape or non-finite values."
                    )
                seed_predictions[seed] = predicted
        except ProductionInferenceContractError:
            raise
        except Exception as exc:
            raise ProductionInferenceContractError(
                "A checkpoint or Chemprop inference failure invalidated the whole ensemble."
            ) from exc
        finally:
            backend.close()

        if tuple(seed_predictions) != PRODUCTION_SEEDS:
            raise ProductionInferenceContractError("The exact five-seed ensemble was not produced.")
        outputs: list[dict[str, object]] = []
        valid_position = 0
        for row in rows:
            if row["status"] != "valid":
                outputs.append(_failed_output(row, self.manifest, self.manifest_sha256))
                continue
            per_seed = {
                seed: _scientific_inverse(seed_predictions[seed][valid_position])
                for seed in PRODUCTION_SEEDS
            }
            outputs.append(_successful_output(row, per_seed, self.manifest, self.manifest_sha256))
            valid_position += 1
        return outputs


def predict_regression_batch(
    inputs: Sequence[Mapping[str, object]],
    *,
    manifest_path: str | Path,
    artifact_root: str | Path | None = None,
    verify_runtime: bool = True,
    num_workers: int = 0,
) -> list[dict[str, object]]:
    """Convenience API for a single batch against the frozen production release."""

    return ChempropRegressionPredictor(
        manifest_path,
        artifact_root=artifact_root,
        verify_runtime=verify_runtime,
        num_workers=num_workers,
    ).predict(inputs)


class _ChempropBackend:
    """Lazy Chemprop adapter that caches molecular graphs and loads one checkpoint at a time."""

    def __init__(self) -> None:
        self._loader: Any = None
        self._trainer: Any = None

    def prepare(self, canonical_smiles: Sequence[str], *, num_workers: int) -> list[int]:
        import lightning.pytorch as pl
        import torch
        from chemprop import data, featurizers

        def make_dataset(smiles_values: Sequence[str]) -> Any:
            datapoints = [data.MoleculeDatapoint.from_smi(smiles) for smiles in smiles_values]
            result = data.MoleculeDataset(
                datapoints,
                featurizer=featurizers.SimpleMoleculeMolGraphFeaturizer(),
                n_workers=num_workers,
            )
            result.cache = True
            return result

        failed: list[int] = []
        try:
            dataset = make_dataset(canonical_smiles)
        except Exception:
            retained: list[str] = []
            for index, smiles in enumerate(canonical_smiles):
                try:
                    make_dataset([smiles])
                except Exception:
                    failed.append(index)
                else:
                    retained.append(smiles)
            if not retained:
                return failed
            dataset = make_dataset(retained)
        self._loader = data.build_dataloader(
            dataset,
            batch_size=64,
            num_workers=num_workers,
            shuffle=False,
        )
        self._trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        return failed

    def predict_checkpoint(
        self,
        checkpoint_path: Path,
        *,
        scaler_mean: np.ndarray,
        scaler_scale: np.ndarray,
    ) -> np.ndarray:
        import torch
        from chemprop import models, nn

        if self._loader is None or self._trainer is None:
            raise RuntimeError("Backend was not prepared.")
        # This trusted, hash-verified checkpoint contains Chemprop metric objects as well as tensors.
        # Chemprop 2.3.1 injects a metric-only task-weight key while reconstructing custom
        # training metrics. Permit that loader quirk, then require the checkpoint's actual
        # serialized state to match the reconstructed inference model exactly.
        model = models.MPNN.load_from_checkpoint(checkpoint_path, map_location="cpu", strict=False)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, dict):
            raise ProductionInferenceContractError("Checkpoint state dictionary is missing.")
        model.load_state_dict(state_dict, strict=True)
        del checkpoint, state_dict
        transform = getattr(model.predictor, "output_transform", None)
        if not isinstance(transform, nn.UnscaleTransform):
            raise ProductionInferenceContractError(
                "Checkpoint does not contain the required target UnscaleTransform."
            )
        recorded_mean = transform.mean.detach().cpu().numpy().reshape(-1)
        recorded_scale = transform.scale.detach().cpu().numpy().reshape(-1)
        if (
            recorded_mean.shape != scaler_mean.shape
            or recorded_scale.shape != scaler_scale.shape
            or not np.allclose(recorded_mean, scaler_mean, rtol=1e-6, atol=1e-7)
            or not np.allclose(recorded_scale, scaler_scale, rtol=1e-6, atol=1e-7)
        ):
            raise ProductionInferenceContractError(
                "Checkpoint target unscaler differs from the production manifest."
            )
        model.eval()  # UnscaleTransform is active only in evaluation mode.
        batches = self._trainer.predict(model, self._loader)
        values = np.concatenate([np.asarray(batch) for batch in batches], axis=0)
        del model, batches
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return values

    def close(self) -> None:
        self._loader = None
        self._trainer = None
        gc.collect()


def _load_and_validate_manifest(
    path: Path, artifact_root: Path, *, verify_runtime: bool
) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    _require_within_root(artifact_root, path, "manifest")
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file():
        raise ProductionInferenceContractError("Production manifest SHA-256 sidecar is missing.")
    manifest_bytes = path.read_bytes()
    match = re.fullmatch(r"([0-9a-f]{64})  (.+)\n?", sidecar.read_text(encoding="utf-8"))
    if match is None or match.group(2) != path.name:
        raise ProductionInferenceContractError("Production manifest sidecar is malformed.")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha256 != match.group(1):
        raise ProductionInferenceContractError("Production manifest SHA-256 mismatch.")
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProductionInferenceContractError("Production manifest is not valid JSON.") from exc
    if not isinstance(manifest, dict):
        raise ProductionInferenceContractError("Production manifest must be a JSON object.")
    _validate_release_contract(manifest)
    if verify_runtime:
        _validate_runtime(manifest)
    return manifest, manifest_sha256


def _validate_release_contract(manifest: Mapping[str, Any]) -> None:
    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ProductionInferenceContractError(message)

    require(manifest.get("schema_version") == SCHEMA_VERSION, "Unsupported manifest schema.")
    require(manifest.get("release_status") == RELEASE_STATUS, "Manifest is not inference-ready.")
    require(tuple(manifest.get("endpoint_order", ())) == ENDPOINT_ORDER, "Endpoint order differs.")
    model = manifest.get("model", {})
    require(isinstance(model, dict), "Model contract is missing.")
    require(model.get("provider") == "moloptima_internal_chemprop", "Model provider differs.")
    require(model.get("family") == "chemprop_dmpnn", "Model family differs.")
    require(model.get("chemprop_version") == "2.3.1", "Chemprop model version differs.")
    require(model.get("task_type") == "regression", "Model is not regression.")
    require(
        model.get("checkpoint_contents") == "five_regression_outputs_no_classification_outputs",
        "Checkpoint output contract differs.",
    )
    ensemble = manifest.get("production_ensemble", {})
    require(tuple(ensemble.get("seeds", ())) == PRODUCTION_SEEDS, "Seed order differs.")
    require(ensemble.get("checkpoint_selection") == "best.ckpt_per_seed", "Wrong checkpoints.")
    require(
        ensemble.get("rule") == "unweighted_arithmetic_mean_of_exactly_all_five_seed_predictions",
        "Ensemble rule differs.",
    )
    require(ensemble.get("missing_seed_policy") == "fail_closed", "Seed policy differs.")
    require(ensemble.get("seed_weighting") == "none", "Seed weights are not allowed.")
    require(ensemble.get("calibration") == "none", "Calibration is not allowed.")
    disagreement = ensemble.get("disagreement", {})
    require(
        disagreement
        == {
            "statistic": "sample_standard_deviation_across_seed_predictions",
            "ddof": 1,
            "seed_count": 5,
            "interpretation": UNCERTAINTY_INTERPRETATION,
        },
        "Disagreement contract differs.",
    )
    locked = manifest.get("data", {}).get("locked_test", {})
    require(
        locked == {"status": "not_accessed", "used_for_manifest": False},
        "Locked-test isolation is not authoritative.",
    )
    preprocessing = manifest.get("data", {}).get("preprocessing", {})
    require(
        preprocessing.get("canonical_isomeric_smiles") is True
        and preprocessing.get("preserve_disconnected_fragments") is True
        and preprocessing.get("preserve_charges") is True
        and preprocessing.get("preserve_stereochemistry") is True
        and preprocessing.get("largest_fragment_selection") is False
        and preprocessing.get("neutralization") is False,
        "Preprocessing contract differs.",
    )
    scaler = manifest.get("target_scaler", {})
    require(scaler.get("fit_split") == "train", "Target scaler was not train-only.")
    require(scaler.get("validation_statistics_used") is False, "Validation entered scaler.")
    require(scaler.get("test_statistics_used") is False, "Test entered scaler.")
    per_endpoint = scaler.get("per_endpoint")
    require(isinstance(per_endpoint, dict), "Target scaler endpoints are missing.")
    require(set(per_endpoint) == set(ENDPOINT_ORDER), "Target scaler endpoint set differs.")
    for endpoint in ENDPOINT_ORDER:
        item = per_endpoint[endpoint]
        require(isinstance(item, dict), f"Scaler contract is invalid for {endpoint}.")
        expected_transform = "log10" if endpoint == "vdss_lombardo" else "identity"
        require(item.get("scientific_transform") == expected_transform, "Scaler transform differs.")
        try:
            mean, scale = float(item["mean"]), float(item["scale"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProductionInferenceContractError("Scaler values are invalid.") from exc
        require(math.isfinite(mean) and math.isfinite(scale) and scale > 0, "Invalid scaler.")
    endpoints = manifest.get("endpoints")
    require(
        isinstance(endpoints, dict) and set(endpoints) == set(ENDPOINT_ORDER),
        "Endpoint set differs.",
    )
    expected = {
        "caco2_wang": (
            "identity",
            "identity_in_stored_label_space; physical_conversion=10**y",
            "log10(Papp [cm/s])",
            "cm/s",
            "stored_log10_papp_space_not_physical_cm_per_s",
        ),
        "lipophilicity_astrazeneca": (
            "identity",
            "identity",
            "log-ratio",
            "log-ratio",
            "user_facing_output_space",
        ),
        "solubility_aqsoldb": (
            "identity",
            "identity",
            "log mol/L",
            "log mol/L",
            "user_facing_output_space",
        ),
        "ppbr_az": (
            "identity",
            "identity",
            "percent bound",
            "percent bound",
            "user_facing_output_space",
        ),
        "vdss_lombardo": ("log10", "10**y", "log10(L/kg)", "L/kg", "user_facing_output_space"),
    }
    for index, endpoint in enumerate(ENDPOINT_ORDER):
        item = endpoints[endpoint]
        transform, inverse, internal_unit, output_unit, validation_space = expected[endpoint]
        require(item.get("order_index") == index, f"Order index differs for {endpoint}.")
        require(item.get("scientific_transform") == transform, f"Transform differs for {endpoint}.")
        require(
            item.get("inverse_transform_for_user_output") == inverse,
            f"Inverse differs for {endpoint}.",
        )
        require(
            item.get("internal_model_output_unit") == internal_unit,
            f"Internal unit differs for {endpoint}.",
        )
        require(item.get("user_facing_unit") == output_unit, f"Output unit differs for {endpoint}.")
        require(item.get("clipping") == "none", f"Clipping differs for {endpoint}.")
        require(
            item.get("training_only_scaler") == per_endpoint[endpoint], "Scaler identity differs."
        )
        require(
            item.get("validation_metrics", {}).get("space") == validation_space,
            "Validation space differs.",
        )


def _verify_checkpoints(manifest: Mapping[str, Any], artifact_root: Path) -> list[tuple[int, Path]]:
    items = manifest.get("checkpoints")
    if not isinstance(items, list) or len(items) != len(PRODUCTION_SEEDS):
        raise ProductionInferenceContractError("Manifest must contain exactly five checkpoints.")
    scaler_sha = manifest["target_scaler"].get("sha256")
    result: list[tuple[int, Path]] = []
    for item, seed in zip(items, PRODUCTION_SEEDS, strict=True):
        if not isinstance(item, dict) or item.get("seed") != seed:
            raise ProductionInferenceContractError("Checkpoint seed order differs.")
        if item.get("sha256") != CHECKPOINT_SHA256[seed]:
            raise ProductionInferenceContractError(f"Seed {seed} approved hash differs.")
        if item.get("target_scaler_sha256") != scaler_sha:
            raise ProductionInferenceContractError(f"Seed {seed} target scaler differs.")
        path = _resolve_portable_path(artifact_root, item.get("path"), f"seed {seed} checkpoint")
        if path.name != "best.ckpt" or not path.is_file():
            raise ProductionInferenceContractError(f"Seed {seed} best checkpoint is missing.")
        if _sha256_file(path) != item["sha256"]:
            raise ProductionInferenceContractError(f"Seed {seed} checkpoint SHA-256 mismatch.")
        result.append((seed, path))
    return result


def _validate_runtime(manifest: Mapping[str, Any]) -> None:
    expected = (
        manifest.get("environment", {}).get("expected_pinned", {}).get("package_versions", {})
    )
    if not isinstance(expected, dict):
        raise ProductionInferenceContractError("Pinned runtime contract is missing.")
    try:
        actual = {
            "chemprop": importlib.metadata.version("chemprop"),
            "torch": importlib.metadata.version("torch"),
            "rdkit": importlib.metadata.version("rdkit"),
        }
    except importlib.metadata.PackageNotFoundError as exc:
        raise ProductionInferenceContractError(
            "Chemprop inference dependencies are missing."
        ) from exc
    if not platform.python_version().startswith("3.11"):
        raise ProductionInferenceContractError("Production inference requires Python 3.11.")
    for package, version in actual.items():
        if version != expected.get(package):
            raise ProductionInferenceContractError(
                f"{package} version differs from the production manifest."
            )


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
            molecule_id, source_smiles, "invalid_smiles", "RDKit could not parse source_smiles."
        )
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    return {
        "status": "valid",
        "molecule_id": molecule_id,
        "source_smiles": source_smiles,
        "canonical_smiles": canonical,
    }


def _invalid_row(
    molecule_id: object, source_smiles: object, code: str, message: str
) -> dict[str, object]:
    return {
        "status": "invalid",
        "molecule_id": molecule_id,
        "source_smiles": source_smiles,
        "canonical_smiles": None,
        "error_code": code,
        "error_message": message,
    }


def _scientific_inverse(unscaled_model_values: np.ndarray) -> dict[str, float]:
    values = {
        endpoint: float(unscaled_model_values[index])
        for index, endpoint in enumerate(ENDPOINT_ORDER)
    }
    values["vdss_lombardo"] = float(10.0 ** values["vdss_lombardo"])
    if not all(math.isfinite(value) for value in values.values()):
        raise ProductionInferenceContractError(
            "Scientific inverse transform produced non-finite output."
        )
    return values


def _successful_output(
    row: Mapping[str, object],
    per_seed: Mapping[int, Mapping[str, float]],
    manifest: Mapping[str, Any],
    manifest_sha256: str,
) -> dict[str, object]:
    endpoints: dict[str, object] = {}
    for endpoint in ENDPOINT_ORDER:
        values = np.asarray([per_seed[seed][endpoint] for seed in PRODUCTION_SEEDS], dtype=float)
        endpoint_result: dict[str, object] = {
            "per_seed": {
                str(seed): float(value)
                for seed, value in zip(PRODUCTION_SEEDS, values, strict=True)
            },
            "ensemble_mean": float(np.mean(values)),
            "seed_standard_deviation": float(np.std(values, ddof=1)),
            "seed_standard_deviation_ddof": 1,
            "uncertainty_interpretation": UNCERTAINTY_INTERPRETATION,
            "unit": manifest["endpoints"][endpoint]["user_facing_unit"],
        }
        if endpoint == "caco2_wang":
            endpoint_result["unit"] = manifest["endpoints"][endpoint]["internal_model_output_unit"]
            log_mean = float(endpoint_result.pop("ensemble_mean"))
            log_std = float(endpoint_result.pop("seed_standard_deviation"))
            endpoint_result.update(
                {
                    "representation": "stored_log10_papp_space",
                    "ensemble_mean_log10_papp_cm_per_s": log_mean,
                    "seed_standard_deviation_log10_papp_cm_per_s": log_std,
                    "physical_papp_cm_per_s_from_ensemble_log10": float(
                        caco2_log10_to_cm_per_s([log_mean])[0]
                    ),
                }
            )
        elif endpoint == "vdss_lombardo":
            endpoint_result["representation"] = "physical_volume_of_distribution"
        else:
            endpoint_result["representation"] = manifest["endpoints"][endpoint][
                "user_facing_output_representation"
            ]
        endpoints[endpoint] = endpoint_result
    return {
        "status": "success",
        "molecule_id": row["molecule_id"],
        "source_smiles": row["source_smiles"],
        "canonical_smiles": row["canonical_smiles"],
        "manifest_schema_version": manifest["schema_version"],
        "manifest_sha256": manifest_sha256,
        "release_status": manifest["release_status"],
        "model_family": manifest["model"]["family"],
        "endpoint_order": list(ENDPOINT_ORDER),
        "applicability_domain": {
            "status": "unavailable_frozen_training_reference_not_packaged",
            "maximum_similarity": None,
            "scaffold_familiarity": None,
            "in_domain": None,
        },
        "endpoints": endpoints,
        "error_code": None,
        "error_message": None,
    }


def _failed_output(
    row: Mapping[str, object], manifest: Mapping[str, Any], manifest_sha256: str
) -> dict[str, object]:
    return {
        "status": "failed",
        "molecule_id": row.get("molecule_id"),
        "source_smiles": row.get("source_smiles"),
        "canonical_smiles": None,
        "manifest_schema_version": manifest["schema_version"],
        "manifest_sha256": manifest_sha256,
        "release_status": manifest["release_status"],
        "model_family": manifest["model"]["family"],
        "endpoint_order": list(ENDPOINT_ORDER),
        "applicability_domain": {
            "status": "not_evaluated_invalid_input",
            "maximum_similarity": None,
            "scaffold_familiarity": None,
            "in_domain": None,
        },
        "endpoints": {},
        "error_code": row.get("error_code"),
        "error_message": row.get("error_message"),
    }


def _infer_artifact_root(manifest_path: Path) -> Path:
    for candidate in (manifest_path.parent, *manifest_path.parents):
        if (candidate / ".git").exists():
            return candidate.resolve()
    raise ValueError("artifact_root is required when the manifest is outside a Git worktree.")


def _resolve_portable_path(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ProductionInferenceContractError(f"{label} path must be repository-root-relative.")
    path = (root / Path(*value.split("/"))).resolve()
    _require_within_root(root, path, label)
    return path


def _require_within_root(root: Path, path: Path, label: str) -> None:
    resolved_root = root.resolve()
    resolved = path.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ProductionInferenceContractError(f"{label} escapes the artifact root.")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "ChempropRegressionPredictor",
    "ProductionInferenceContractError",
    "SeedPredictionBackend",
    "predict_regression_batch",
]
