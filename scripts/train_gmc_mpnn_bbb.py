"""Train the frozen-data Chemprop 2.1.0 GMC-MPNN BBB model."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Final, Sequence


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
    TRAIN_SPLIT_CONTRACT,
    VALIDATION_SPLIT_CONTRACT,
    FrozenDevelopmentFeatures,
    build_chemprop_dataloaders,
    build_chemprop_dataset,
    load_frozen_development_features,
)


TRAINING_RUNNER_VERSION: Final = "gmc-mpnn-bbb-training-runner-v1"
DEFAULT_TRAIN_DIR: Final = ROOT / "outputs" / "gpu" / "pilot" / "gmc_mpnn_training_preprocessing_v2"
DEFAULT_VALIDATION_DIR: Final = (
    ROOT / "outputs" / "gpu" / "pilot" / "gmc_mpnn_validation_preprocessing_v1"
)
DEFAULT_SCALER_DIR: Final = ROOT / "outputs" / "gpu" / "pilot" / "gmc_mpnn_ggl_scaler_v1"
RUN_SUMMARY_FILENAME: Final = "run_summary.json"
BEST_CHECKPOINT_FILENAME: Final = "best.ckpt"
LAST_CHECKPOINT_FILENAME: Final = "last.ckpt"


class GMCTrainingRunnerError(RuntimeError):
    """Training-runner configuration or publication failure."""


@dataclass(frozen=True)
class TrainingConfig:
    """One deterministic TRAIN/validation-only GMC-MPNN run."""

    train_preprocessing_dir: Path
    validation_preprocessing_dir: Path
    scaler_dir: Path
    output_dir: Path
    seed: int
    max_epochs: int = 100
    num_workers: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        if isinstance(self.max_epochs, bool) or self.max_epochs <= 0:
            raise ValueError("max_epochs must be a positive integer.")
        if isinstance(self.num_workers, bool) or self.num_workers < 0:
            raise ValueError("num_workers must be a nonnegative integer.")


@dataclass(frozen=True)
class TrainingDependencies:
    """Lazily imported runtime modules, injectable for no-training unit tests."""

    chemprop: Any
    lightning: Any
    torch: Any


def run_training(
    config: TrainingConfig,
    *,
    dependencies: TrainingDependencies | None = None,
    clock: Callable[[], float] = time.perf_counter,
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Run one seed and atomically publish checkpoints and provenance."""

    output_dir = config.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Training output directory already exists: {output_dir}")

    started = clock()
    runtime = dependencies or _load_training_dependencies()
    _require_runtime_versions(runtime)
    architecture = GMCMPNNArchitecture(maximum_epochs=config.max_epochs)
    _seed_deterministically(runtime, config.seed)

    frozen = load_frozen_development_features(
        config.train_preprocessing_dir,
        config.validation_preprocessing_dir,
        config.scaler_dir,
    )
    _validate_frozen_counts(frozen)

    model_bundle = build_gmc_mpnn_model(
        chemprop_module=runtime.chemprop,
        architecture=architecture,
    )
    train_dataset = build_chemprop_dataset(
        frozen.train,
        model_bundle,
        chemprop_module=runtime.chemprop,
    )
    validation_dataset = build_chemprop_dataset(
        frozen.validation,
        model_bundle,
        chemprop_module=runtime.chemprop,
    )
    loaders = build_chemprop_dataloaders(
        train_dataset,
        validation_dataset,
        chemprop_module=runtime.chemprop,
        architecture=architecture,
        seed=config.seed,
        num_workers=config.num_workers,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        checkpoint = runtime.lightning.callbacks.ModelCheckpoint(
            dirpath=temporary_dir,
            filename="best",
            monitor=architecture.checkpoint_monitor,
            mode=architecture.checkpoint_mode,
            save_top_k=architecture.checkpoint_save_top_k,
            save_last=True,
            auto_insert_metric_name=False,
        )
        early_stopping = runtime.lightning.callbacks.EarlyStopping(
            monitor=architecture.checkpoint_monitor,
            mode=architecture.checkpoint_mode,
            patience=architecture.early_stopping_patience,
        )
        accelerator = "gpu" if runtime.torch.cuda.is_available() else "cpu"
        trainer = runtime.lightning.Trainer(
            accelerator=accelerator,
            devices=1,
            max_epochs=config.max_epochs,
            deterministic=True,
            callbacks=[checkpoint, early_stopping],
            default_root_dir=temporary_dir,
            logger=False,
            enable_checkpointing=True,
        )
        trainer.fit(
            model_bundle.model,
            train_dataloaders=loaders.train_loader,
            val_dataloaders=loaders.validation_loader,
        )
        elapsed_seconds = clock() - started
        _validate_checkpoints(temporary_dir, checkpoint)
        summary = _build_run_summary(
            config=config,
            architecture=architecture,
            frozen=frozen,
            runtime=runtime,
            trainer=trainer,
            checkpoint=checkpoint,
            early_stopping=early_stopping,
            accelerator=accelerator,
            elapsed_seconds=elapsed_seconds,
            git_commit=git_commit or _git_commit(),
        )
        _write_json(temporary_dir / RUN_SUMMARY_FILENAME, summary)
        if output_dir.exists():
            raise FileExistsError(f"Training output directory appeared during run: {output_dir}")
        os.replace(temporary_dir, output_dir)
        return summary
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one deterministic GMC-MPNN BBB seed from frozen TRAIN/validation data."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-preprocessing-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--validation-preprocessing-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    parser.add_argument("--scaler-dir", type=Path, default=DEFAULT_SCALER_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = TrainingConfig(
        train_preprocessing_dir=args.train_preprocessing_dir,
        validation_preprocessing_dir=args.validation_preprocessing_dir,
        scaler_dir=args.scaler_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        max_epochs=args.max_epochs,
        num_workers=args.num_workers,
    )
    summary = run_training(config)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


def _load_training_dependencies() -> TrainingDependencies:
    try:
        import chemprop
        import lightning.pytorch as lightning
        import torch
    except ImportError as exc:  # pragma: no cover - ECHO runtime boundary
        raise GMCTrainingRunnerError(
            "The validated Chemprop 2.1.0/Lightning 2.1.4/Torch 2.1.2 environment is required."
        ) from exc
    return TrainingDependencies(chemprop=chemprop, lightning=lightning, torch=torch)


def _require_runtime_versions(runtime: TrainingDependencies) -> None:
    observed = getattr(runtime.chemprop, "__version__", None)
    if observed != EXPECTED_CHEMPROP_VERSION:
        raise GMCTrainingRunnerError(
            f"Training requires Chemprop {EXPECTED_CHEMPROP_VERSION}; observed {observed!r}."
        )


def _seed_deterministically(runtime: TrainingDependencies, seed: int) -> None:
    runtime.lightning.seed_everything(seed, workers=True)
    runtime.torch.use_deterministic_algorithms(True)
    if hasattr(runtime.torch.backends, "cudnn"):
        runtime.torch.backends.cudnn.deterministic = True
        runtime.torch.backends.cudnn.benchmark = False


def _validate_frozen_counts(frozen: FrozenDevelopmentFeatures) -> None:
    if len(frozen.train.features) != TRAIN_SPLIT_CONTRACT.successful_molecules:
        raise GMCTrainingRunnerError("TRAIN must contain exactly 1,558 frozen successful rows.")
    if len(frozen.validation.features) != VALIDATION_SPLIT_CONTRACT.successful_molecules:
        raise GMCTrainingRunnerError("VALIDATION must contain exactly 196 frozen successful rows.")


def _validate_checkpoints(output_dir: Path, checkpoint: Any) -> None:
    best = output_dir / BEST_CHECKPOINT_FILENAME
    last = output_dir / LAST_CHECKPOINT_FILENAME
    if not best.is_file() or not last.is_file():
        raise GMCTrainingRunnerError("Training did not produce both best.ckpt and last.ckpt.")
    if Path(str(checkpoint.best_model_path)).name != BEST_CHECKPOINT_FILENAME:
        raise GMCTrainingRunnerError("Lightning did not identify best.ckpt as the best checkpoint.")


def _build_run_summary(
    *,
    config: TrainingConfig,
    architecture: GMCMPNNArchitecture,
    frozen: FrozenDevelopmentFeatures,
    runtime: TrainingDependencies,
    trainer: Any,
    checkpoint: Any,
    early_stopping: Any,
    accelerator: str,
    elapsed_seconds: float,
    git_commit: str,
) -> dict[str, Any]:
    best_score = _optional_float(checkpoint.best_model_score)
    if best_score is None:
        raise GMCTrainingRunnerError("Lightning did not report a finite best validation loss.")
    gpu_available = bool(runtime.torch.cuda.is_available())
    return {
        "training_runner_version": TRAINING_RUNNER_VERSION,
        "seed": config.seed,
        "architecture": asdict(architecture),
        "hyperparameters": {
            "batch_size": architecture.batch_size,
            "max_epochs": config.max_epochs,
            "num_workers": config.num_workers,
            "deterministic": True,
            "checkpoint_monitor": architecture.checkpoint_monitor,
            "checkpoint_mode": architecture.checkpoint_mode,
            "checkpoint_save_top_k": architecture.checkpoint_save_top_k,
            "early_stopping_patience": architecture.early_stopping_patience,
            "optimizer": architecture.optimizer,
            "scheduler": architecture.scheduler,
        },
        "train_successful_rows": len(frozen.train.features),
        "validation_successful_rows": len(frozen.validation.features),
        "data_counts": {
            "train_source_rows": TRAIN_SPLIT_CONTRACT.source_rows,
            "train_successful_rows": TRAIN_SPLIT_CONTRACT.successful_molecules,
            "train_heavy_atoms": TRAIN_SPLIT_CONTRACT.heavy_atoms,
            "train_excluded_rows": len(TRAIN_SPLIT_CONTRACT.exclusions),
            "train_failed_rows": len(TRAIN_SPLIT_CONTRACT.failures),
            "validation_source_rows": VALIDATION_SPLIT_CONTRACT.source_rows,
            "validation_successful_rows": VALIDATION_SPLIT_CONTRACT.successful_molecules,
            "validation_heavy_atoms": VALIDATION_SPLIT_CONTRACT.heavy_atoms,
            "validation_excluded_rows": len(VALIDATION_SPLIT_CONTRACT.exclusions),
            "validation_failed_rows": len(VALIDATION_SPLIT_CONTRACT.failures),
        },
        "frozen_provenance": {
            "model_interface_version": MODEL_INTERFACE_VERSION,
            "model_data_contract_version": MODEL_DATA_CONTRACT_VERSION,
            "train_feature_manifest_sha256": frozen.train.feature_manifest_sha256,
            "train_molecule_status_sha256": frozen.train.molecule_status_sha256,
            "validation_feature_manifest_sha256": frozen.validation.feature_manifest_sha256,
            "validation_molecule_status_sha256": frozen.validation.molecule_status_sha256,
            "scaler_version": frozen.scaler.scaler_version,
            "portable_scaler_sha256": frozen.scaler.portable_scaler_sha256,
            "feature_order": list(frozen.scaler.feature_order),
        },
        "package_versions": _package_versions(runtime),
        "git_commit": git_commit,
        "gpu_cuda": {
            "accelerator": accelerator,
            "gpu_available": gpu_available,
            "gpu_name": runtime.torch.cuda.get_device_name(0) if gpu_available else None,
            "cuda_runtime": runtime.torch.version.cuda,
        },
        "best_val_loss": best_score,
        "best_checkpoint": BEST_CHECKPOINT_FILENAME,
        "last_checkpoint": LAST_CHECKPOINT_FILENAME,
        "final_epoch": int(trainer.current_epoch),
        "elapsed_seconds": float(elapsed_seconds),
        "early_stopping": {
            "stopped_early": bool(trainer.should_stop),
            "stopped_epoch": int(early_stopping.stopped_epoch),
            "wait_count": int(early_stopping.wait_count),
            "best_score": _optional_float(early_stopping.best_score),
        },
        "test_artifact_accessed": False,
    }


def _package_versions(runtime: TrainingDependencies) -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "chemprop": str(runtime.chemprop.__version__),
        "lightning": _distribution_version("lightning", runtime.lightning),
        "torch": str(runtime.torch.__version__),
        "numpy": _distribution_version("numpy"),
        "scikit_learn": _distribution_version("scikit-learn"),
        "rdkit": _distribution_version("rdkit"),
        "setuptools": _distribution_version("setuptools"),
        "pytest": _distribution_version("pytest"),
    }


def _distribution_version(name: str, module: Any | None = None) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        observed = getattr(module, "__version__", None)
        return str(observed) if observed is not None else "unavailable"


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        value = value.item()
    result = float(value)
    if not (-float("inf") < result < float("inf")):
        raise GMCTrainingRunnerError("Training metric is NaN or infinite.")
    return result


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
        raise GMCTrainingRunnerError("Unable to record the Git commit for this run.") from exc


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
