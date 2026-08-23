"""Batch-first production inference for the frozen GMC-MPNN BBB ensemble."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Final, Mapping, Protocol, Sequence

import numpy as np
from rdkit import Chem

from admet_platform.gmc_mpnn.geometry import (
    GeometryError,
    GeometryResult,
    generate_deterministic_geometry,
)
from admet_platform.gmc_mpnn.ggl import (
    GGL_FEATURE_NAMES,
    GGLPreprocessingError,
    compute_ggl_features,
)
from admet_platform.gmc_mpnn.model import (
    BOND_FEATURE_DIM,
    GGL_ATOM_FEATURE_DIM,
    MODEL_INTERFACE_VERSION,
    TOTAL_ATOM_FEATURE_DIM,
    GMCMPNNArchitecture,
    build_gmc_mpnn_model,
)
from admet_platform.gmc_mpnn.production_manifest import (
    MANIFEST_VERSION,
    PRODUCTION_SEEDS,
    PRODUCTION_THRESHOLD,
    load_inference_manifest,
)
from admet_platform.gmc_mpnn.scaling import (
    FrozenGGLScaler,
    load_frozen_ggl_scaler,
    transform_frozen_ggl,
)
from admet_platform.gmc_mpnn.standardization import (
    EXCLUDED_BY_POLICY,
    GMCStandardizationError,
    standardize_for_gmc_geometry,
)


INFERENCE_ADAPTER_VERSION: Final = "gmc-mpnn-bbb-production-batch-inference-v1"
MODEL_FAMILY: Final = "GMC-MPNN"
SUCCESS: Final = "success"
FAILED: Final = "failed"
NOT_STARTED: Final = "not_started"
OUTPUT_FIELDS: Final = (
    "molecule_id",
    "source_smiles",
    "canonical_smiles",
    "preprocessing_status",
    "parent_standardization_status",
    "seed13_probability",
    "seed37_probability",
    "seed73_probability",
    "seed101_probability",
    "seed137_probability",
    "ensemble_probability",
    "ensemble_standard_deviation",
    "threshold",
    "prediction",
    "model_family",
    "manifest_version",
    "model_interface_version",
    "status",
    "error_code",
    "error_message",
)


class GMCProductionInferenceError(RuntimeError):
    """A batch-level production inference contract failure."""


class MoleculePreprocessingError(ValueError):
    """A per-molecule preprocessing failure with a stable machine-readable code."""

    def __init__(self, code: str, message: str, *, parent_status: str = NOT_STARTED):
        super().__init__(message)
        self.code = code
        self.parent_status = parent_status


@dataclass(frozen=True)
class GMCInferenceInput:
    """One production input row."""

    molecule_id: str
    source_smiles: str


@dataclass(frozen=True)
class PreparedProductionMolecule:
    """One preprocessed molecule reused unchanged by all five checkpoints."""

    input_index: int
    molecule_id: str
    source_smiles: str
    canonical_smiles: str
    parent_standardization_status: str
    geometry_smiles: str
    molecule: Chem.Mol
    geometry: GeometryResult
    scaled_ggl_features: np.ndarray


@dataclass(frozen=True)
class ModelExecutionResult:
    """Per-input model probabilities and isolated errors."""

    probabilities: Mapping[int, Mapping[int, float]]
    errors: Mapping[int, tuple[str, str]]


@dataclass(frozen=True)
class InferenceRuntime:
    """Lazily imported ECHO model runtime."""

    chemprop: Any
    lightning: Any
    torch: Any


class ModelExecutor(Protocol):
    def predict(self, prepared: Sequence[PreparedProductionMolecule]) -> ModelExecutionResult:
        """Return probabilities keyed by input index and seed."""


PreprocessFunction = Callable[[GMCInferenceInput, int, FrozenGGLScaler], PreparedProductionMolecule]


class GMCProductionPredictor:
    """Manifest-bound, batch-first GMC-MPNN BBB production predictor."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        artifact_root: str | Path | None = None,
        verify_runtime: bool = True,
        num_workers: int = 0,
        manifest_loader: Callable[..., dict[str, Any]] = load_inference_manifest,
        scaler_loader: Callable[[str | Path], FrozenGGLScaler] = load_frozen_ggl_scaler,
        preprocess_function: PreprocessFunction | None = None,
        model_executor: ModelExecutor | None = None,
        runtime: InferenceRuntime | None = None,
    ) -> None:
        if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
            raise ValueError("num_workers must be a nonnegative integer.")
        self.manifest_path = Path(manifest_path)
        self.artifact_root = (
            Path(artifact_root).resolve()
            if artifact_root is not None
            else self.manifest_path.parent.resolve()
        )
        # This must remain the first artifact-dependent operation.
        self.manifest = manifest_loader(
            self.manifest_path,
            artifact_root=self.artifact_root,
            verify_runtime=verify_runtime,
        )
        self._validate_loaded_manifest_identity()
        scaler_directory = self._scaler_directory()
        self.scaler = scaler_loader(scaler_directory)
        expected_scaler_hash = self.manifest["scaler"]["portable_sha256"]
        if self.scaler.portable_scaler_sha256 != expected_scaler_hash:
            raise GMCProductionInferenceError(
                "Loaded scaler identity does not match the production manifest."
            )
        self._preprocess = preprocess_function or preprocess_production_molecule
        self._executor = model_executor or SequentialChempropExecutor(
            manifest=self.manifest,
            artifact_root=self.artifact_root,
            runtime=runtime,
            num_workers=num_workers,
        )

    def predict_batch(
        self, inputs: Sequence[GMCInferenceInput | Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        """Predict an ordered batch with per-molecule failure isolation."""

        if isinstance(inputs, (str, bytes)) or not isinstance(inputs, Sequence):
            raise TypeError("Production inference inputs must be a sequence of molecule records.")
        outputs: list[dict[str, Any]] = []
        prepared: list[PreparedProductionMolecule] = []
        for index, raw_input in enumerate(inputs):
            base, normalized = _base_output(raw_input, self.manifest)
            outputs.append(base)
            if normalized is None:
                continue
            try:
                molecule = self._preprocess(normalized, index, self.scaler)
            except MoleculePreprocessingError as exc:
                _set_failure(
                    outputs[index],
                    code=exc.code,
                    message=str(exc),
                    preprocessing_status=FAILED,
                    parent_status=exc.parent_status,
                )
                continue
            except Exception as exc:  # pragma: no cover - defensive plugin boundary
                _set_failure(
                    outputs[index],
                    code="preprocessing_failed",
                    message=str(exc),
                    preprocessing_status=FAILED,
                )
                continue
            prepared.append(molecule)
            outputs[index]["canonical_smiles"] = molecule.canonical_smiles
            outputs[index]["preprocessing_status"] = SUCCESS
            outputs[index]["parent_standardization_status"] = molecule.parent_standardization_status

        if not prepared:
            return outputs
        try:
            execution = self._executor.predict(prepared)
        except Exception as exc:  # pragma: no cover - defensive runtime boundary
            execution = ModelExecutionResult(
                probabilities={},
                errors={
                    item.input_index: ("model_inference_failed", str(exc)) for item in prepared
                },
            )
        prepared_indices = {item.input_index for item in prepared}
        if set(execution.probabilities).difference(prepared_indices) or set(
            execution.errors
        ).difference(prepared_indices):
            raise GMCProductionInferenceError("Model executor returned an unknown input index.")
        for item in prepared:
            index = item.input_index
            error = execution.errors.get(index)
            if error is not None:
                _set_failure(
                    outputs[index],
                    code=error[0],
                    message=error[1],
                    preprocessing_status=outputs[index]["preprocessing_status"],
                    parent_status=outputs[index]["parent_standardization_status"],
                )
                continue
            seed_values = execution.probabilities.get(index)
            try:
                probabilities = _validate_seed_probabilities(seed_values)
            except GMCProductionInferenceError as exc:
                _set_failure(
                    outputs[index],
                    code="model_inference_failed",
                    message=str(exc),
                    preprocessing_status=outputs[index]["preprocessing_status"],
                    parent_status=outputs[index]["parent_standardization_status"],
                )
                continue
            values = np.asarray(
                [probabilities[seed] for seed in PRODUCTION_SEEDS], dtype=np.float64
            )
            ensemble = float(values.mean(dtype=np.float64))
            disagreement = float(values.std(ddof=0, dtype=np.float64))
            for seed in PRODUCTION_SEEDS:
                outputs[index][f"seed{seed}_probability"] = probabilities[seed]
            outputs[index]["ensemble_probability"] = ensemble
            outputs[index]["ensemble_standard_deviation"] = disagreement
            outputs[index]["prediction"] = "BBB+" if ensemble >= PRODUCTION_THRESHOLD else "BBB-"
            outputs[index]["status"] = SUCCESS
        return outputs

    def predict_one(self, molecule: GMCInferenceInput | Mapping[str, Any]) -> dict[str, Any]:
        """Convenience wrapper around the batch-first interface."""

        return self.predict_batch([molecule])[0]

    def _validate_loaded_manifest_identity(self) -> None:
        if self.manifest.get("manifest_version") != MANIFEST_VERSION:
            raise GMCProductionInferenceError("Unsupported production manifest version.")
        if self.manifest.get("model_interface_version") != MODEL_INTERFACE_VERSION:
            raise GMCProductionInferenceError("Manifest model interface is incompatible.")
        if self.manifest.get("model_family") != MODEL_FAMILY:
            raise GMCProductionInferenceError("Manifest model family is incompatible.")
        if self.manifest.get("ensemble", {}).get("seeds") != list(PRODUCTION_SEEDS):
            raise GMCProductionInferenceError("Manifest production seeds are incompatible.")
        if self.manifest.get("production_decision", {}).get("threshold") != (PRODUCTION_THRESHOLD):
            raise GMCProductionInferenceError("Manifest production threshold is incompatible.")

    def _scaler_directory(self) -> Path:
        artifacts = self.manifest["scaler"]["artifacts"]
        paths = {
            key: _resolve_artifact_path(record["path"], self.artifact_root)
            for key, record in artifacts.items()
        }
        if {path.parent for path in paths.values()} != {paths["json"].parent}:
            raise GMCProductionInferenceError("Scaler artifacts must share one directory.")
        expected_names = {
            "json": "scaler.json",
            "npz": "scaler.npz",
            "fit_summary": "fit_summary.json",
        }
        if any(paths[key].name != name for key, name in expected_names.items()):
            raise GMCProductionInferenceError("Scaler artifact filenames are incompatible.")
        return paths["json"].parent


class SequentialChempropExecutor:
    """Load one checkpoint at a time while reusing one prepared graph loader."""

    def __init__(
        self,
        *,
        manifest: Mapping[str, Any],
        artifact_root: Path,
        runtime: InferenceRuntime | None = None,
        num_workers: int = 0,
    ) -> None:
        self.manifest = manifest
        self.artifact_root = artifact_root
        self.runtime = runtime
        self.num_workers = num_workers

    def predict(self, prepared: Sequence[PreparedProductionMolecule]) -> ModelExecutionResult:
        if not prepared:
            return ModelExecutionResult(probabilities={}, errors={})
        runtime = self.runtime or _load_runtime()
        bundle = build_gmc_mpnn_model(
            chemprop_module=runtime.chemprop,
            architecture=GMCMPNNArchitecture(),
        )
        datapoints: list[tuple[PreparedProductionMolecule, Any]] = []
        errors: dict[int, tuple[str, str]] = {}
        for item in prepared:
            try:
                datapoint = _chemprop_datapoint(item, bundle, runtime.chemprop)
            except Exception as exc:
                errors[item.input_index] = ("feature_construction_failed", str(exc))
                continue
            datapoints.append((item, datapoint))
        if not datapoints:
            return ModelExecutionResult(probabilities={}, errors=errors)
        eligible = [item for item, _ in datapoints]
        try:
            dataset = runtime.chemprop.data.MoleculeDataset(
                [datapoint for _, datapoint in datapoints], featurizer=bundle.featurizer
            )
            loader = _build_prediction_loader(
                dataset, runtime.chemprop, self.num_workers, GMCMPNNArchitecture().batch_size
            )
        except Exception as exc:
            for item in eligible:
                errors[item.input_index] = ("feature_construction_failed", str(exc))
            return ModelExecutionResult(probabilities={}, errors=errors)

        probabilities: dict[int, dict[int, float]] = {item.input_index: {} for item in eligible}
        individual_loaders: dict[int, Any] = {}
        checkpoints = {int(record["seed"]): record for record in self.manifest["checkpoints"]}
        for seed in PRODUCTION_SEEDS:
            checkpoint_path = _resolve_artifact_path(checkpoints[seed]["path"], self.artifact_root)
            _seed_deterministically(runtime, seed)
            trainer = _prediction_trainer(runtime)
            try:
                batch_values = _run_prediction(
                    trainer,
                    bundle.model,
                    loader,
                    checkpoint_path,
                    expected_count=len(eligible),
                )
            except Exception:
                batch_values = None
            if batch_values is not None:
                for item, probability in zip(eligible, batch_values, strict=True):
                    probabilities[item.input_index][seed] = float(probability)
                continue
            for item, datapoint in datapoints:
                if item.input_index in errors:
                    continue
                try:
                    if item.input_index not in individual_loaders:
                        individual_dataset = runtime.chemprop.data.MoleculeDataset(
                            [datapoint], featurizer=bundle.featurizer
                        )
                        individual_loaders[item.input_index] = _build_prediction_loader(
                            individual_dataset, runtime.chemprop, self.num_workers, 1
                        )
                    values = _run_prediction(
                        _prediction_trainer(runtime),
                        bundle.model,
                        individual_loaders[item.input_index],
                        checkpoint_path,
                        expected_count=1,
                    )
                    probabilities[item.input_index][seed] = float(values[0])
                except Exception as exc:
                    errors[item.input_index] = (
                        "model_inference_failed",
                        f"Seed {seed} inference failed: {exc}",
                    )
                    probabilities.pop(item.input_index, None)
        return ModelExecutionResult(probabilities=probabilities, errors=errors)


def preprocess_production_molecule(
    inference_input: GMCInferenceInput,
    input_index: int,
    scaler: FrozenGGLScaler,
) -> PreparedProductionMolecule:
    """Preprocess one new molecule once using only frozen production transforms."""

    if inference_input.molecule_id is None:
        raise MoleculePreprocessingError("invalid_molecule_id", "molecule_id must be nonempty.")
    molecule_id = str(inference_input.molecule_id).strip()
    if not molecule_id:
        raise MoleculePreprocessingError("invalid_molecule_id", "molecule_id must be nonempty.")
    source_smiles = inference_input.source_smiles
    if not isinstance(source_smiles, str) or not source_smiles.strip():
        raise MoleculePreprocessingError("invalid_smiles", "source_smiles must be nonempty.")
    parsed = Chem.MolFromSmiles(source_smiles)
    if parsed is None:
        raise MoleculePreprocessingError("invalid_smiles", "RDKit could not parse source_smiles.")
    canonical_smiles = Chem.MolToSmiles(parsed, canonical=True, isomericSmiles=True)
    try:
        standardized = standardize_for_gmc_geometry(molecule_id, canonical_smiles)
    except GMCStandardizationError as exc:
        raise MoleculePreprocessingError(
            f"parent_standardization_{exc.status}", str(exc), parent_status=FAILED
        ) from exc
    if standardized.action == EXCLUDED_BY_POLICY or not standardized.geometry_canonical_smiles:
        raise MoleculePreprocessingError(
            "parent_standardization_excluded",
            standardized.exclusion_reason or "Molecule was excluded by the frozen parent policy.",
            parent_status=standardized.action,
        )
    try:
        geometry = generate_deterministic_geometry(standardized.geometry_canonical_smiles)
    except GeometryError as exc:
        raise MoleculePreprocessingError(
            f"geometry_{exc.status}", str(exc), parent_status=standardized.action
        ) from exc
    try:
        ggl = compute_ggl_features(
            geometry.coordinates,
            np.asarray(geometry.heavy_atom_atomic_numbers, dtype=np.int64),
            geometry_fingerprint=geometry.geometry_fingerprint,
        )
    except GGLPreprocessingError as exc:
        raise MoleculePreprocessingError(
            f"ggl_{exc.status}", str(exc), parent_status=standardized.action
        ) from exc
    if tuple(ggl.feature_names) != tuple(GGL_FEATURE_NAMES):
        raise MoleculePreprocessingError(
            "ggl_feature_order_mismatch",
            "GGL feature order differs from the frozen scaler contract.",
            parent_status=standardized.action,
        )
    try:
        scaled = transform_frozen_ggl(ggl.features, scaler)
    except Exception as exc:
        raise MoleculePreprocessingError(
            "scaler_transform_failed", str(exc), parent_status=standardized.action
        ) from exc
    model_molecule = Chem.MolFromSmiles(geometry.canonical_isomeric_smiles)
    if model_molecule is None:
        raise MoleculePreprocessingError(
            "feature_construction_failed",
            "Standardized geometry SMILES could not be parsed for Chemprop.",
            parent_status=standardized.action,
        )
    atomic_numbers = np.asarray(
        [atom.GetAtomicNum() for atom in model_molecule.GetAtoms()], dtype=np.int64
    )
    if not np.array_equal(
        atomic_numbers, np.asarray(geometry.heavy_atom_atomic_numbers, dtype=np.int64)
    ):
        raise MoleculePreprocessingError(
            "feature_atom_alignment_failed",
            "Chemprop molecule atom order differs from geometry/GGL atom order.",
            parent_status=standardized.action,
        )
    if scaled.shape != (model_molecule.GetNumAtoms(), GGL_ATOM_FEATURE_DIM):
        raise MoleculePreprocessingError(
            "feature_shape_mismatch",
            "Scaled GGL features are not aligned [n_atoms,6].",
            parent_status=standardized.action,
        )
    return PreparedProductionMolecule(
        input_index=input_index,
        molecule_id=molecule_id,
        source_smiles=source_smiles,
        canonical_smiles=canonical_smiles,
        parent_standardization_status=standardized.action,
        geometry_smiles=geometry.canonical_isomeric_smiles,
        molecule=model_molecule,
        geometry=geometry,
        scaled_ggl_features=np.array(scaled, dtype=np.float64, copy=True),
    )


def predict_bbb_batch(
    inputs: Sequence[GMCInferenceInput | Mapping[str, Any]],
    *,
    manifest_path: str | Path,
    artifact_root: str | Path | None = None,
    verify_runtime: bool = True,
    num_workers: int = 0,
) -> list[dict[str, Any]]:
    """Public functional batch API."""

    predictor = GMCProductionPredictor(
        manifest_path,
        artifact_root=artifact_root,
        verify_runtime=verify_runtime,
        num_workers=num_workers,
    )
    return predictor.predict_batch(inputs)


def _base_output(
    value: GMCInferenceInput | Mapping[str, Any], manifest: Mapping[str, Any]
) -> tuple[dict[str, Any], GMCInferenceInput | None]:
    if isinstance(value, GMCInferenceInput):
        normalized = value
    elif isinstance(value, Mapping):
        normalized = GMCInferenceInput(
            molecule_id=value.get("molecule_id"),  # type: ignore[arg-type]
            source_smiles=value.get("source_smiles"),  # type: ignore[arg-type]
        )
    else:
        normalized = None
    molecule_id = "" if normalized is None else str(normalized.molecule_id)
    source_smiles = None if normalized is None else normalized.source_smiles
    output = {
        "molecule_id": molecule_id,
        "source_smiles": source_smiles,
        "canonical_smiles": None,
        "preprocessing_status": NOT_STARTED,
        "parent_standardization_status": NOT_STARTED,
        **{f"seed{seed}_probability": None for seed in PRODUCTION_SEEDS},
        "ensemble_probability": None,
        "ensemble_standard_deviation": None,
        "threshold": PRODUCTION_THRESHOLD,
        "prediction": None,
        "model_family": MODEL_FAMILY,
        "manifest_version": manifest["manifest_version"],
        "model_interface_version": manifest["model_interface_version"],
        "status": FAILED,
        "error_code": None,
        "error_message": None,
    }
    if normalized is None:
        _set_failure(
            output,
            code="invalid_input_record",
            message="Input must be GMCInferenceInput or a mapping.",
            preprocessing_status=FAILED,
        )
    return output, normalized


def _set_failure(
    output: dict[str, Any],
    *,
    code: str,
    message: str,
    preprocessing_status: str,
    parent_status: str | None = None,
) -> None:
    output["status"] = FAILED
    output["preprocessing_status"] = preprocessing_status
    if parent_status is not None:
        output["parent_standardization_status"] = parent_status
    output["prediction"] = None
    output["ensemble_probability"] = None
    output["ensemble_standard_deviation"] = None
    output["error_code"] = code
    output["error_message"] = message or code


def _validate_seed_probabilities(value: Mapping[int, float] | None) -> dict[int, float]:
    if not isinstance(value, Mapping) or set(value) != set(PRODUCTION_SEEDS):
        raise GMCProductionInferenceError("All five production seed probabilities are required.")
    result: dict[int, float] = {}
    for seed in PRODUCTION_SEEDS:
        probability = value[seed]
        if isinstance(probability, bool) or not isinstance(probability, (int, float, np.number)):
            raise GMCProductionInferenceError(f"Seed {seed} probability is not numeric.")
        parsed = float(probability)
        if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
            raise GMCProductionInferenceError(f"Seed {seed} probability is outside [0, 1].")
        result[seed] = parsed
    return result


def _chemprop_datapoint(item: PreparedProductionMolecule, bundle: Any, chemprop: Any) -> Any:
    if bundle.featurizer.atom_fdim != TOTAL_ATOM_FEATURE_DIM:
        raise GMCProductionInferenceError("Chemprop atom feature dimension is not 78.")
    if bundle.featurizer.bond_fdim != BOND_FEATURE_DIM:
        raise GMCProductionInferenceError("Chemprop bond feature dimension is not 14.")
    features = np.asarray(item.scaled_ggl_features, dtype=np.float32)
    if features.shape != (item.molecule.GetNumAtoms(), GGL_ATOM_FEATURE_DIM):
        raise GMCProductionInferenceError("Model-boundary V_f must be float32 [n_atoms,6].")
    return chemprop.data.MoleculeDatapoint(
        mol=item.molecule,
        y=None,
        V_f=features,
        name=item.molecule_id,
    )


def _build_prediction_loader(dataset: Any, chemprop: Any, num_workers: int, batch_size: int) -> Any:
    return chemprop.data.build_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )


def _prediction_trainer(runtime: InferenceRuntime) -> Any:
    accelerator = "gpu" if runtime.torch.cuda.is_available() else "cpu"
    return runtime.lightning.Trainer(
        accelerator=accelerator,
        devices=1,
        deterministic=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )


def _run_prediction(
    trainer: Any,
    model: Any,
    loader: Any,
    checkpoint_path: Path,
    *,
    expected_count: int,
) -> np.ndarray:
    batches = trainer.predict(model, dataloaders=loader, ckpt_path=str(checkpoint_path))
    if not isinstance(batches, Sequence) or isinstance(batches, (str, bytes)) or not batches:
        raise GMCProductionInferenceError("Checkpoint returned no prediction batches.")
    arrays = [_to_numpy(batch) for batch in batches]
    try:
        probabilities = np.concatenate(arrays, axis=0).reshape(-1).astype(np.float64, copy=False)
    except ValueError as exc:
        raise GMCProductionInferenceError("Prediction batches have incompatible shapes.") from exc
    if probabilities.shape != (expected_count,):
        raise GMCProductionInferenceError("Checkpoint returned an incompatible prediction count.")
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise GMCProductionInferenceError("Checkpoint probabilities must be finite within [0, 1].")
    return probabilities


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _load_runtime() -> InferenceRuntime:
    try:
        import chemprop
        import lightning.pytorch as lightning
        import torch
    except ImportError as exc:  # pragma: no cover - ECHO runtime boundary
        raise GMCProductionInferenceError(
            "The frozen Chemprop/Lightning/Torch production environment is required."
        ) from exc
    return InferenceRuntime(chemprop=chemprop, lightning=lightning, torch=torch)


def _seed_deterministically(runtime: InferenceRuntime, seed: int) -> None:
    runtime.lightning.seed_everything(seed, workers=True)
    runtime.torch.use_deterministic_algorithms(True)
    if hasattr(runtime.torch.backends, "cudnn"):
        runtime.torch.backends.cudnn.deterministic = True
        runtime.torch.backends.cudnn.benchmark = False


def _resolve_artifact_path(path_value: str, root: Path) -> Path:
    portable = PurePosixPath(path_value)
    if (
        not path_value
        or "\\" in path_value
        or portable.is_absolute()
        or PureWindowsPath(path_value).is_absolute()
        or ".." in portable.parts
    ):
        raise GMCProductionInferenceError("Manifest artifact path is unsafe.")
    resolved_root = root.resolve()
    resolved = resolved_root.joinpath(*portable.parts).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise GMCProductionInferenceError("Manifest artifact path escapes artifact_root.") from exc
    if not resolved.is_file():
        raise GMCProductionInferenceError(f"Required production artifact is missing: {path_value}")
    return resolved


__all__ = [
    "FAILED",
    "INFERENCE_ADAPTER_VERSION",
    "GMCInferenceInput",
    "GMCProductionInferenceError",
    "GMCProductionPredictor",
    "InferenceRuntime",
    "ModelExecutionResult",
    "MoleculePreprocessingError",
    "OUTPUT_FIELDS",
    "PreparedProductionMolecule",
    "SequentialChempropExecutor",
    "predict_bbb_batch",
    "preprocess_production_molecule",
]
