"""Validation-only five-seed inference for frozen GMC-MPNN BBB checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from admet_platform.gmc_mpnn.model import (  # noqa: E402
    EXPECTED_CHEMPROP_VERSION,
    MODEL_INTERFACE_VERSION,
    GMCMPNNArchitecture,
    build_gmc_mpnn_model,
)
from admet_platform.gmc_mpnn.model_data import (  # noqa: E402
    MODEL_DATA_CONTRACT_VERSION,
    VALIDATION_SPLIT_CONTRACT,
    FrozenValidationFeatures,
    build_chemprop_dataset,
    build_chemprop_validation_dataloader,
    load_frozen_validation_features,
)


EVALUATION_RUNNER_VERSION: Final = "gmc-mpnn-bbb-validation-five-seed-v1"
EXPECTED_SEEDS: Final = (13, 37, 73, 101, 137)
DEFAULT_VALIDATION_DIR: Final = (
    ROOT / "outputs" / "gpu" / "pilot" / "gmc_mpnn_validation_preprocessing_v1"
)
DEFAULT_SCALER_DIR: Final = ROOT / "outputs" / "gpu" / "pilot" / "gmc_mpnn_ggl_scaler_v1"
DEFAULT_CHECKPOINTS: Final = tuple(
    (
        seed,
        ROOT / "outputs" / "gpu" / "pilot" / f"gmc_mpnn_bbb_training_seed{seed}" / "best.ckpt",
    )
    for seed in EXPECTED_SEEDS
)
PREDICTIONS_FILENAME: Final = "validation_predictions.csv"
METRICS_FILENAME: Final = "validation_metrics.json"
ENSEMBLE_FILENAME: Final = "ensemble_summary.json"
TRAINING_SUMMARY_FILENAME: Final = "run_summary.json"
THRESHOLD: Final = 0.5


class GMCValidationEvaluationError(RuntimeError):
    """A validation-only evaluation contract violation."""


@dataclass(frozen=True)
class EvaluationConfig:
    validation_preprocessing_dir: Path
    scaler_dir: Path
    checkpoint_paths: tuple[tuple[int, Path], ...]
    output_dir: Path
    num_workers: int = 0

    def __post_init__(self) -> None:
        seeds = tuple(seed for seed, _ in self.checkpoint_paths)
        if seeds != EXPECTED_SEEDS:
            raise ValueError(f"Checkpoint seeds must be exactly {EXPECTED_SEEDS} in that order.")
        if isinstance(self.num_workers, bool) or self.num_workers < 0:
            raise ValueError("num_workers must be a nonnegative integer.")


@dataclass(frozen=True)
class EvaluationDependencies:
    chemprop: Any
    lightning: Any
    torch: Any


def evaluate_validation(
    config: EvaluationConfig,
    *,
    dependencies: EvaluationDependencies | None = None,
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Evaluate five fixed checkpoints and atomically publish validation-only outputs."""

    output_dir = config.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Evaluation output directory already exists: {output_dir}")
    checkpoint_metadata = _preflight_checkpoints(config.checkpoint_paths)

    runtime = dependencies or _load_evaluation_dependencies()
    _require_runtime_contract(runtime)
    architecture = GMCMPNNArchitecture()
    frozen = load_frozen_validation_features(
        config.validation_preprocessing_dir,
        config.scaler_dir,
    )
    validation = frozen.validation
    if len(validation.features) != VALIDATION_SPLIT_CONTRACT.successful_molecules:
        raise GMCValidationEvaluationError(
            "VALIDATION must contain exactly 196 frozen successful rows."
        )
    _validate_checkpoint_summaries(checkpoint_metadata, frozen, architecture)

    model_bundle = build_gmc_mpnn_model(
        chemprop_module=runtime.chemprop,
        architecture=architecture,
    )
    validation_dataset = build_chemprop_dataset(
        validation,
        model_bundle,
        chemprop_module=runtime.chemprop,
    )
    validation_loader = build_chemprop_validation_dataloader(
        validation_dataset,
        chemprop_module=runtime.chemprop,
        architecture=architecture,
        num_workers=config.num_workers,
    )

    accelerator = "gpu" if runtime.torch.cuda.is_available() else "cpu"
    seed_probabilities: dict[int, np.ndarray] = {}
    for seed, checkpoint_path in config.checkpoint_paths:
        _seed_deterministically(runtime, seed)
        trainer = runtime.lightning.Trainer(
            accelerator=accelerator,
            devices=1,
            deterministic=True,
            logger=False,
            enable_checkpointing=False,
        )
        try:
            batches = trainer.predict(
                model_bundle.model,
                dataloaders=validation_loader,
                ckpt_path=str(checkpoint_path),
            )
        except Exception as exc:
            raise GMCValidationEvaluationError(
                f"Checkpoint/model contract failed for seed {seed}."
            ) from exc
        seed_probabilities[seed] = _prediction_vector(
            batches,
            expected_count=VALIDATION_SPLIT_CONTRACT.successful_molecules,
            seed=seed,
        )

    labels = np.asarray(validation.labels, dtype=np.float64).reshape(-1)
    _validate_labels(labels)
    probability_matrix = np.column_stack([seed_probabilities[seed] for seed in EXPECTED_SEEDS])
    ensemble_probability = probability_matrix.mean(axis=1, dtype=np.float64)
    probability_std = probability_matrix.std(axis=1, ddof=0, dtype=np.float64)
    _validate_probabilities(ensemble_probability, "ensemble")
    if not np.isfinite(probability_std).all():
        raise GMCValidationEvaluationError("Ensemble disagreement contains NaN or infinity.")

    predictions = _prediction_frame(
        frozen,
        labels,
        seed_probabilities,
        ensemble_probability,
        probability_std,
    )
    metric_results = {
        f"seed{seed}": _classification_metrics(labels, seed_probabilities[seed])
        for seed in EXPECTED_SEEDS
    }
    metric_results["ensemble"] = _classification_metrics(labels, ensemble_probability)
    provenance = _provenance(
        config=config,
        frozen=frozen,
        checkpoint_metadata=checkpoint_metadata,
        runtime=runtime,
        accelerator=accelerator,
        git_commit=git_commit or _git_commit(),
    )
    metrics_payload = {
        "evaluation_runner_version": EVALUATION_RUNNER_VERSION,
        "validation_count": len(validation.features),
        "seeds": list(EXPECTED_SEEDS),
        "threshold": THRESHOLD,
        "metrics": metric_results,
        "provenance": provenance,
        "test_artifact_accessed": False,
    }
    ensemble_payload = {
        "evaluation_runner_version": EVALUATION_RUNNER_VERSION,
        "validation_count": len(validation.features),
        "seeds": list(EXPECTED_SEEDS),
        "ensemble_method": "unweighted_arithmetic_mean",
        "probability_standard_deviation_ddof": 0,
        "mean_probability_standard_deviation": float(probability_std.mean()),
        "maximum_probability_standard_deviation": float(probability_std.max()),
        "ensemble_metrics": metric_results["ensemble"],
        "provenance": provenance,
        "test_artifact_accessed": False,
    }
    _publish_outputs(output_dir, predictions, metrics_payload, ensemble_payload)
    return {
        "validation_predictions": predictions,
        "validation_metrics": metrics_payload,
        "ensemble_summary": ensemble_payload,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the fixed five-seed GMC-MPNN ensemble on frozen validation only."
    )
    parser.add_argument("--validation-preprocessing-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    parser.add_argument("--scaler-dir", type=Path, default=DEFAULT_SCALER_DIR)
    for seed, path in DEFAULT_CHECKPOINTS:
        parser.add_argument(f"--checkpoint-seed{seed}", type=Path, default=path)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint_paths = tuple(
        (seed, getattr(args, f"checkpoint_seed{seed}")) for seed in EXPECTED_SEEDS
    )
    result = evaluate_validation(
        EvaluationConfig(
            validation_preprocessing_dir=args.validation_preprocessing_dir,
            scaler_dir=args.scaler_dir,
            checkpoint_paths=checkpoint_paths,
            output_dir=args.output_dir,
            num_workers=args.num_workers,
        )
    )
    print(
        json.dumps(
            result["ensemble_summary"],
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


def _preflight_checkpoints(
    checkpoint_paths: tuple[tuple[int, Path], ...],
) -> dict[int, dict[str, Any]]:
    metadata_by_seed: dict[int, dict[str, Any]] = {}
    for seed, checkpoint_path in checkpoint_paths:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing seed {seed} checkpoint: {checkpoint_path}")
        summary_path = checkpoint_path.parent / TRAINING_SUMMARY_FILENAME
        summary = _read_json(summary_path, f"seed {seed} training summary")
        metadata_by_seed[seed] = {
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "training_summary_path": summary_path,
            "training_summary_sha256": _sha256_file(summary_path),
            "training_summary": summary,
        }
    return metadata_by_seed


def _validate_checkpoint_summaries(
    checkpoint_metadata: Mapping[int, Mapping[str, Any]],
    frozen: FrozenValidationFeatures,
    architecture: GMCMPNNArchitecture,
) -> None:
    expected_architecture = asdict(architecture)
    for seed in EXPECTED_SEEDS:
        summary = checkpoint_metadata[seed]["training_summary"]
        expected = {
            "seed": seed,
            "architecture": expected_architecture,
            "train_successful_rows": 1558,
            "validation_successful_rows": 196,
            "best_checkpoint": "best.ckpt",
            "test_artifact_accessed": False,
        }
        for key, value in expected.items():
            if summary.get(key) != value:
                raise GMCValidationEvaluationError(
                    f"Seed {seed} training summary has incompatible {key}."
                )
        hyperparameters = summary.get("hyperparameters")
        expected_hyperparameters = {
            "batch_size": architecture.batch_size,
            "max_epochs": architecture.maximum_epochs,
            "checkpoint_monitor": architecture.checkpoint_monitor,
            "checkpoint_mode": architecture.checkpoint_mode,
            "checkpoint_save_top_k": architecture.checkpoint_save_top_k,
            "early_stopping_patience": architecture.early_stopping_patience,
            "optimizer": architecture.optimizer,
            "scheduler": architecture.scheduler,
        }
        if not isinstance(hyperparameters, dict) or any(
            hyperparameters.get(key) != value for key, value in expected_hyperparameters.items()
        ):
            raise GMCValidationEvaluationError(
                f"Seed {seed} training summary has incompatible hyperparameters."
            )
        provenance = summary.get("frozen_provenance")
        if not isinstance(provenance, dict):
            raise GMCValidationEvaluationError(
                f"Seed {seed} training summary lacks frozen provenance."
            )
        expected_provenance = {
            "model_interface_version": MODEL_INTERFACE_VERSION,
            "model_data_contract_version": MODEL_DATA_CONTRACT_VERSION,
            "validation_feature_manifest_sha256": frozen.validation.feature_manifest_sha256,
            "validation_molecule_status_sha256": frozen.validation.molecule_status_sha256,
            "scaler_version": frozen.scaler.scaler_version,
            "portable_scaler_sha256": frozen.scaler.portable_scaler_sha256,
            "feature_order": list(frozen.scaler.feature_order),
        }
        for key, value in expected_provenance.items():
            if provenance.get(key) != value:
                raise GMCValidationEvaluationError(
                    f"Seed {seed} checkpoint provenance has incompatible {key}."
                )


def _prediction_vector(batches: Any, *, expected_count: int, seed: int) -> np.ndarray:
    if not isinstance(batches, Sequence) or isinstance(batches, (str, bytes)) or not batches:
        raise GMCValidationEvaluationError(f"Seed {seed} returned no prediction batches.")
    arrays = [_to_numpy(batch) for batch in batches]
    try:
        probabilities = np.concatenate(arrays, axis=0).reshape(-1).astype(np.float64, copy=False)
    except ValueError as exc:
        raise GMCValidationEvaluationError(
            f"Seed {seed} prediction batches have incompatible shapes."
        ) from exc
    if probabilities.shape != (expected_count,):
        raise GMCValidationEvaluationError(
            f"Seed {seed} returned {len(probabilities)} predictions; expected {expected_count}."
        )
    _validate_probabilities(probabilities, f"seed {seed}")
    return probabilities


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _validate_probabilities(probabilities: np.ndarray, label: str) -> None:
    if not np.isfinite(probabilities).all():
        raise GMCValidationEvaluationError(f"{label} probabilities contain NaN or infinity.")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise GMCValidationEvaluationError(f"{label} probabilities fall outside [0, 1].")


def _validate_labels(labels: np.ndarray) -> None:
    if labels.shape != (VALIDATION_SPLIT_CONTRACT.successful_molecules,):
        raise GMCValidationEvaluationError("Validation labels are not aligned to 196 molecules.")
    if not np.isfinite(labels).all() or not np.isin(labels, (0.0, 1.0)).all():
        raise GMCValidationEvaluationError("Validation labels must be finite binary values.")
    if len(np.unique(labels)) != 2:
        raise GMCValidationEvaluationError("Validation metrics require both binary classes.")


def _prediction_frame(
    frozen: FrozenValidationFeatures,
    labels: np.ndarray,
    seed_probabilities: Mapping[int, np.ndarray],
    ensemble_probability: np.ndarray,
    probability_std: np.ndarray,
) -> pd.DataFrame:
    features = frozen.validation.features
    frame = pd.DataFrame(
        {
            "record_key": [feature.record_key for feature in features],
            "molecule_id": [feature.molecule_id for feature in features],
            "canonical_smiles": [feature.canonical_smiles for feature in features],
            "true_label": labels.astype(np.int64),
            **{f"probability_seed{seed}": seed_probabilities[seed] for seed in EXPECTED_SEEDS},
            "ensemble_probability": ensemble_probability,
            "probability_standard_deviation": probability_std,
        }
    )
    if len(frame) != VALIDATION_SPLIT_CONTRACT.successful_molecules:
        raise GMCValidationEvaluationError("Prediction table does not contain exactly 196 rows.")
    if frame["record_key"].duplicated().any():
        raise GMCValidationEvaluationError("Prediction table contains duplicate record keys.")
    return frame


def _classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = (probabilities >= THRESHOLD).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "binary_cross_entropy": float(log_loss(labels, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "threshold": THRESHOLD,
    }


def _provenance(
    *,
    config: EvaluationConfig,
    frozen: FrozenValidationFeatures,
    checkpoint_metadata: Mapping[int, Mapping[str, Any]],
    runtime: EvaluationDependencies,
    accelerator: str,
    git_commit: str,
) -> dict[str, Any]:
    gpu_available = bool(runtime.torch.cuda.is_available())
    return {
        "validation_count": len(frozen.validation.features),
        "seeds": list(EXPECTED_SEEDS),
        "checkpoints": {
            str(seed): {
                "path": _portable_path(config.checkpoint_paths[index][1]),
                "sha256": checkpoint_metadata[seed]["checkpoint_sha256"],
                "training_summary_sha256": checkpoint_metadata[seed]["training_summary_sha256"],
            }
            for index, seed in enumerate(EXPECTED_SEEDS)
        },
        "frozen_validation": {
            "model_interface_version": MODEL_INTERFACE_VERSION,
            "model_data_contract_version": MODEL_DATA_CONTRACT_VERSION,
            "feature_manifest_sha256": frozen.validation.feature_manifest_sha256,
            "molecule_status_sha256": frozen.validation.molecule_status_sha256,
            "scaler_version": frozen.scaler.scaler_version,
            "portable_scaler_sha256": frozen.scaler.portable_scaler_sha256,
            "feature_order": list(frozen.scaler.feature_order),
        },
        "git_commit": git_commit,
        "package_versions": _package_versions(runtime),
        "gpu_cuda": {
            "accelerator": accelerator,
            "gpu_available": gpu_available,
            "gpu_name": runtime.torch.cuda.get_device_name(0) if gpu_available else None,
            "cuda_runtime": runtime.torch.version.cuda,
        },
        "test_artifact_accessed": False,
    }


def _publish_outputs(
    output_dir: Path,
    predictions: pd.DataFrame,
    metrics: dict[str, Any],
    ensemble: dict[str, Any],
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        predictions.to_csv(
            temporary_dir / PREDICTIONS_FILENAME,
            index=False,
            lineterminator="\n",
            float_format="%.17g",
        )
        _write_json(temporary_dir / METRICS_FILENAME, metrics)
        _write_json(temporary_dir / ENSEMBLE_FILENAME, ensemble)
        if output_dir.exists():
            raise FileExistsError(f"Evaluation output directory appeared during run: {output_dir}")
        os.replace(temporary_dir, output_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def _load_evaluation_dependencies() -> EvaluationDependencies:
    try:
        import chemprop
        import lightning.pytorch as lightning
        import torch
    except ImportError as exc:  # pragma: no cover - ECHO runtime boundary
        raise GMCValidationEvaluationError(
            "The validated Chemprop 2.1.0/Lightning 2.1.4/Torch 2.1.2 environment is required."
        ) from exc
    return EvaluationDependencies(chemprop=chemprop, lightning=lightning, torch=torch)


def _require_runtime_contract(runtime: EvaluationDependencies) -> None:
    if getattr(runtime.chemprop, "__version__", None) != EXPECTED_CHEMPROP_VERSION:
        raise GMCValidationEvaluationError(
            f"Evaluation requires Chemprop {EXPECTED_CHEMPROP_VERSION}."
        )


def _seed_deterministically(runtime: EvaluationDependencies, seed: int) -> None:
    runtime.lightning.seed_everything(seed, workers=True)
    runtime.torch.use_deterministic_algorithms(True)
    if hasattr(runtime.torch.backends, "cudnn"):
        runtime.torch.backends.cudnn.deterministic = True
        runtime.torch.backends.cudnn.benchmark = False


def _package_versions(runtime: EvaluationDependencies) -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "chemprop": str(runtime.chemprop.__version__),
        "lightning": _distribution_version("lightning", runtime.lightning),
        "torch": str(runtime.torch.__version__),
        "numpy": _distribution_version("numpy"),
        "scikit_learn": _distribution_version("scikit-learn"),
        "rdkit": _distribution_version("rdkit"),
    }


def _distribution_version(name: str, module: Any | None = None) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        observed = getattr(module, "__version__", None)
        return str(observed) if observed is not None else "unavailable"


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GMCValidationEvaluationError(f"Cannot read {label}: {path.name}") from exc
    if not isinstance(value, dict):
        raise GMCValidationEvaluationError(f"{label} must be a JSON object.")
    return value


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise GMCValidationEvaluationError(f"Cannot hash required file: {path.name}") from exc


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise GMCValidationEvaluationError("Unable to record the Git commit.") from exc


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
