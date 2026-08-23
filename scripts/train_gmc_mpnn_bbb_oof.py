"""Train the fixed five-seed GMC-MPNN ensemble in each frozen TRAIN OOF fold."""

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
import time
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Final, Mapping, Sequence

import numpy as np
import pandas as pd


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
    FrozenSupervisedSplit,
    build_chemprop_dataloaders,
    build_chemprop_dataset,
    build_chemprop_validation_dataloader,
    load_frozen_feature_split,
)
from admet_platform.gmc_mpnn.scaling import (  # noqa: E402
    FIT_SUMMARY_FILENAME,
    FrozenGGLScaler,
    load_frozen_ggl_scaler,
)


OOF_TRAINING_RUNNER_VERSION: Final = "gmc-mpnn-bbb-nested-oof-training-v1"
OOF_MANIFEST_VERSION: Final = "gmc-mpnn-bbb-train-nested-oof-v1"
EXPECTED_TRAIN_COUNT: Final = 1558
OUTER_FOLDS: Final = (0, 1, 2, 3, 4)
PRODUCTION_SEEDS: Final = (13, 37, 73, 101, 137)
EXPECTED_SPLIT_SEED: Final = 1729
OUTER_MANIFEST_FILENAME: Final = "outer_fold_manifest.csv"
INNER_MANIFEST_FILENAME: Final = "inner_split_manifest.csv"
SPLIT_SUMMARY_FILENAME: Final = "split_summary.json"
OOF_PREDICTIONS_FILENAME: Final = "oof_predictions.csv"
OOF_SUMMARY_FILENAME: Final = "oof_summary.json"
BEST_CHECKPOINT_FILENAME: Final = "best.ckpt"
LAST_CHECKPOINT_FILENAME: Final = "last.ckpt"
SEED_PREDICTIONS_FILENAME: Final = "holdout_predictions.csv"
SEED_SUMMARY_FILENAME: Final = "run_summary.json"

DEFAULT_TRAIN_DIR: Final = ROOT / "outputs" / "gpu" / "pilot" / "gmc_mpnn_training_preprocessing_v2"
DEFAULT_SCALER_DIR: Final = ROOT / "outputs" / "gpu" / "pilot" / "gmc_mpnn_ggl_scaler_v1"
DEFAULT_OOF_MANIFEST_DIR: Final = (
    ROOT / "outputs" / "gpu" / "pilot" / "gmc_mpnn_bbb_oof_manifest_v1"
)


class GMCOOFTrainingError(RuntimeError):
    """A nested OOF training, prediction, or provenance contract violation."""


@dataclass(frozen=True)
class OOFTrainingConfig:
    train_preprocessing_dir: Path
    scaler_dir: Path
    oof_manifest_dir: Path
    output_dir: Path
    max_epochs: int = 100
    num_workers: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.max_epochs, bool) or self.max_epochs <= 0:
            raise ValueError("max_epochs must be a positive integer.")
        if isinstance(self.num_workers, bool) or self.num_workers < 0:
            raise ValueError("num_workers must be a nonnegative integer.")


@dataclass(frozen=True)
class OOFTrainingDependencies:
    chemprop: Any
    lightning: Any
    torch: Any


@dataclass(frozen=True)
class FrozenOOFMembership:
    outer: pd.DataFrame
    inner: pd.DataFrame
    summary: Mapping[str, Any]
    hashes: Mapping[str, str]


@dataclass(frozen=True)
class SeedRunResult:
    probabilities: np.ndarray
    best_val_loss: float
    final_epoch: int
    elapsed_seconds: float
    early_stopping: Mapping[str, Any]


def run_oof_training(
    config: OOFTrainingConfig,
    *,
    dependencies: OOFTrainingDependencies | None = None,
    clock: Callable[[], float] = time.perf_counter,
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Train 25 models and atomically publish one complete TRAIN OOF prediction table."""

    output_dir = config.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"OOF training output directory already exists: {output_dir}")

    membership = _load_oof_membership(config.oof_manifest_dir)
    scaler, scaler_summary, train = _load_frozen_train(config)
    _validate_membership(membership, train, scaler, scaler_summary)
    runtime = dependencies or _load_dependencies()
    _require_runtime(runtime)
    architecture = GMCMPNNArchitecture(maximum_epochs=config.max_epochs)
    accelerator = "gpu" if runtime.torch.cuda.is_available() else "cpu"
    resolved_git_commit = git_commit or _git_commit()
    started = clock()

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        prediction_rows: list[pd.DataFrame] = []
        model_runs: list[dict[str, Any]] = []
        for outer_fold in OUTER_FOLDS:
            inner_train, inner_early_stop, holdout, holdout_manifest = _select_fold_data(
                train, membership, outer_fold
            )
            for seed in PRODUCTION_SEEDS:
                seed_dir = temporary_dir / f"fold{outer_fold}" / f"seed{seed}"
                seed_dir.mkdir(parents=True)
                result = _train_predict_one(
                    runtime=runtime,
                    architecture=architecture,
                    inner_train=inner_train,
                    inner_early_stop=inner_early_stop,
                    holdout=holdout,
                    output_dir=seed_dir,
                    seed=seed,
                    num_workers=config.num_workers,
                    accelerator=accelerator,
                    clock=clock,
                )
                probabilities = _validate_prediction_vector(
                    result.probabilities,
                    expected_count=len(holdout.features),
                    label=f"outer fold {outer_fold} seed {seed}",
                )
                seed_frame = _seed_prediction_frame(
                    holdout_manifest,
                    holdout,
                    seed,
                    probabilities,
                )
                seed_predictions_path = seed_dir / SEED_PREDICTIONS_FILENAME
                seed_frame.to_csv(
                    seed_predictions_path,
                    index=False,
                    lineterminator="\n",
                    float_format="%.17g",
                )
                checkpoint_metadata = _checkpoint_metadata(seed_dir)
                checkpoint_metadata.update(
                    {
                        "best_checkpoint_path": (
                            f"fold{outer_fold}/seed{seed}/{BEST_CHECKPOINT_FILENAME}"
                        ),
                        "last_checkpoint_path": (
                            f"fold{outer_fold}/seed{seed}/{LAST_CHECKPOINT_FILENAME}"
                        ),
                    }
                )
                seed_summary = {
                    "oof_training_runner_version": OOF_TRAINING_RUNNER_VERSION,
                    "outer_fold": outer_fold,
                    "seed": seed,
                    "architecture": asdict(architecture),
                    "inner_train_count": len(inner_train.features),
                    "inner_early_stop_validation_count": len(inner_early_stop.features),
                    "outer_holdout_count": len(holdout.features),
                    "best_val_loss": result.best_val_loss,
                    "final_epoch": result.final_epoch,
                    "elapsed_seconds": result.elapsed_seconds,
                    "early_stopping": dict(result.early_stopping),
                    "checkpoints": checkpoint_metadata,
                    "holdout_predictions_path": (
                        f"fold{outer_fold}/seed{seed}/{SEED_PREDICTIONS_FILENAME}"
                    ),
                    "holdout_predictions_sha256": _sha256_file(seed_predictions_path),
                    "outer_holdout_usage": "prediction_only_after_checkpoint_selection",
                    "validation_artifact_accessed": False,
                    "test_artifact_accessed": False,
                }
                _write_json(seed_dir / SEED_SUMMARY_FILENAME, seed_summary)
                model_runs.append(seed_summary)
                prediction_rows.append(seed_frame)

        predictions = _aggregate_oof_predictions(prediction_rows, membership.outer)
        predictions_path = temporary_dir / OOF_PREDICTIONS_FILENAME
        predictions.to_csv(
            predictions_path,
            index=False,
            lineterminator="\n",
            float_format="%.17g",
        )
        elapsed_seconds = clock() - started
        summary = _build_oof_summary(
            config=config,
            membership=membership,
            train=train,
            scaler=scaler,
            scaler_summary=scaler_summary,
            architecture=architecture,
            runtime=runtime,
            accelerator=accelerator,
            model_runs=model_runs,
            predictions=predictions,
            predictions_sha256=_sha256_file(predictions_path),
            elapsed_seconds=elapsed_seconds,
            git_commit=resolved_git_commit,
        )
        _write_json(temporary_dir / OOF_SUMMARY_FILENAME, summary)
        if output_dir.exists():
            raise FileExistsError(
                f"OOF training output directory appeared during run: {output_dir}"
            )
        os.replace(temporary_dir, output_dir)
        return summary
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the fixed 5-fold x 5-seed TRAIN-only GMC-MPNN OOF ensemble."
    )
    parser.add_argument("--train-preprocessing-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--scaler-dir", type=Path, default=DEFAULT_SCALER_DIR)
    parser.add_argument("--oof-manifest-dir", type=Path, default=DEFAULT_OOF_MANIFEST_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_oof_training(
        OOFTrainingConfig(
            train_preprocessing_dir=args.train_preprocessing_dir,
            scaler_dir=args.scaler_dir,
            oof_manifest_dir=args.oof_manifest_dir,
            output_dir=args.output_dir,
            max_epochs=args.max_epochs,
            num_workers=args.num_workers,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


def _load_frozen_train(
    config: OOFTrainingConfig,
) -> tuple[FrozenGGLScaler, dict[str, Any], FrozenSupervisedSplit]:
    scaler = load_frozen_ggl_scaler(config.scaler_dir)
    scaler_summary = _read_json(config.scaler_dir / FIT_SUMMARY_FILENAME, "scaler fit summary")
    _require_false(scaler_summary, "validation_artifact_accessed", "scaler fit summary")
    _require_false(scaler_summary, "test_artifact_accessed", "scaler fit summary")
    train = load_frozen_feature_split(
        config.train_preprocessing_dir,
        split="train",
        scaler=scaler,
        scaler_summary=scaler_summary,
    )
    if len(train.features) != EXPECTED_TRAIN_COUNT:
        raise GMCOOFTrainingError("Frozen TRAIN must contain exactly 1,558 successful molecules.")
    return scaler, scaler_summary, train


def _load_oof_membership(directory: Path) -> FrozenOOFMembership:
    source = directory.resolve()
    outer_path = source / OUTER_MANIFEST_FILENAME
    inner_path = source / INNER_MANIFEST_FILENAME
    summary_path = source / SPLIT_SUMMARY_FILENAME
    summary = _read_json(summary_path, "OOF split summary")
    _require_false(summary, "validation_artifact_accessed", "OOF split summary")
    _require_false(summary, "test_artifact_accessed", "OOF split summary")
    outer = _read_csv(outer_path, "outer OOF manifest")
    inner = _read_csv(inner_path, "inner OOF manifest")
    hashes = {
        "outer_fold_manifest_sha256": _sha256_file(outer_path),
        "inner_split_manifest_sha256": _sha256_file(inner_path),
        "split_summary_sha256": _sha256_file(summary_path),
    }
    expected_hashes = summary.get("assignment_hashes")
    if not isinstance(expected_hashes, Mapping):
        raise GMCOOFTrainingError("OOF split summary lacks assignment hashes.")
    for field in ("outer_fold_manifest_sha256", "inner_split_manifest_sha256"):
        if expected_hashes.get(field) != hashes[field]:
            raise GMCOOFTrainingError(f"Frozen OOF {field} does not match split_summary.json.")
    return FrozenOOFMembership(outer=outer, inner=inner, summary=summary, hashes=hashes)


def _validate_membership(
    membership: FrozenOOFMembership,
    train: FrozenSupervisedSplit,
    scaler: FrozenGGLScaler,
    scaler_summary: Mapping[str, Any],
) -> None:
    summary = membership.summary
    expected_summary = {
        "oof_manifest_version": OOF_MANIFEST_VERSION,
        "train_count": EXPECTED_TRAIN_COUNT,
        "outer_fold_count": len(OUTER_FOLDS),
        "inner_fold_count": 8,
        "split_seed": EXPECTED_SPLIT_SEED,
        "validation_artifact_accessed": False,
        "test_artifact_accessed": False,
    }
    for field, expected in expected_summary.items():
        if summary.get(field) != expected:
            raise GMCOOFTrainingError(f"OOF split summary has incompatible {field}.")
    summary_checks = summary.get("checks")
    if not isinstance(summary_checks, Mapping) or not summary_checks:
        raise GMCOOFTrainingError("OOF split summary lacks deterministic safety checks.")
    if any(value is not True for value in summary_checks.values()):
        raise GMCOOFTrainingError("OOF split summary contains an unproven safety check.")
    outer = membership.outer
    inner = membership.inner
    if list(outer.columns) != [
        "record_key",
        "molecule_id",
        "canonical_smiles",
        "label",
        "scaffold_key",
        "outer_fold",
    ]:
        raise GMCOOFTrainingError("Outer OOF manifest schema is incompatible.")
    if list(inner.columns) != ["outer_fold", "record_key", "inner_role"]:
        raise GMCOOFTrainingError("Inner OOF manifest schema is incompatible.")
    if len(outer) != EXPECTED_TRAIN_COUNT or outer["record_key"].duplicated().any():
        raise GMCOOFTrainingError("Outer OOF assignments must contain 1,558 unique records.")
    if set(outer["outer_fold"]) != set(OUTER_FOLDS):
        raise GMCOOFTrainingError("Outer OOF assignments must contain exactly folds 0 through 4.")

    features_by_key = {feature.record_key: feature for feature in train.features}
    labels_by_key = {
        feature.record_key: int(label[0])
        for feature, label in zip(train.features, train.labels, strict=True)
    }
    if set(outer["record_key"]) != set(features_by_key):
        raise GMCOOFTrainingError("OOF record identities do not match frozen TRAIN.")
    observed_labels = dict(zip(outer["record_key"], outer["label"], strict=True))
    if observed_labels != labels_by_key:
        raise GMCOOFTrainingError("OOF labels do not match frozen TRAIN labels.")
    for row in outer.itertuples(index=False):
        feature = features_by_key[str(row.record_key)]
        if str(row.molecule_id) != feature.molecule_id:
            raise GMCOOFTrainingError("OOF molecule identity does not match frozen TRAIN.")
        if str(row.canonical_smiles) != feature.canonical_smiles:
            raise GMCOOFTrainingError("OOF canonical SMILES does not match frozen TRAIN.")

    frozen_provenance = summary.get("frozen_train_provenance")
    if not isinstance(frozen_provenance, Mapping):
        raise GMCOOFTrainingError("OOF split summary lacks frozen TRAIN provenance.")
    expected_provenance = {
        "feature_manifest_sha256": train.feature_manifest_sha256,
        "molecule_status_sha256": train.molecule_status_sha256,
        "ordered_input_artifact_sha256": scaler_summary.get("ordered_input_artifact_sha256"),
    }
    for field, expected in expected_provenance.items():
        if frozen_provenance.get(field) != expected:
            raise GMCOOFTrainingError(f"OOF frozen TRAIN provenance has incompatible {field}.")
    if train.portable_scaler_sha256 != scaler.portable_scaler_sha256:
        raise GMCOOFTrainingError("Frozen TRAIN and scaler hashes disagree.")
    if scaler_summary.get("training_molecule_count") != EXPECTED_TRAIN_COUNT:
        raise GMCOOFTrainingError("Frozen global scaler was not fitted on 1,558 TRAIN molecules.")
    if scaler_summary.get("training_atom_count") != 38245:
        raise GMCOOFTrainingError("Frozen global scaler was not fitted on 38,245 TRAIN atoms.")

    for fold in OUTER_FOLDS:
        holdout = outer.loc[outer["outer_fold"] == fold]
        development = outer.loc[outer["outer_fold"] != fold]
        if set(int(value) for value in holdout["label"].unique()) != {0, 1}:
            raise GMCOOFTrainingError(f"Outer fold {fold} holdout lacks a binary class.")
        if set(holdout["scaffold_key"]) & set(development["scaffold_key"]):
            raise GMCOOFTrainingError(f"Outer fold {fold} has scaffold leakage.")
        fold_inner = inner.loc[inner["outer_fold"] == fold]
        if fold_inner["record_key"].duplicated().any():
            raise GMCOOFTrainingError(f"Outer fold {fold} has duplicate inner assignments.")
        if set(fold_inner["record_key"]) != set(development["record_key"]):
            raise GMCOOFTrainingError(f"Outer fold {fold} inner membership is incomplete.")
        roles = set(fold_inner["inner_role"])
        if roles != {"inner_train", "inner_early_stop_validation"}:
            raise GMCOOFTrainingError(f"Outer fold {fold} has incompatible inner roles.")
        train_keys = fold_inner.loc[fold_inner["inner_role"] == "inner_train", "record_key"]
        early_keys = fold_inner.loc[
            fold_inner["inner_role"] == "inner_early_stop_validation", "record_key"
        ]
        labels_by_key = outer.set_index("record_key")["label"]
        if set(int(value) for value in labels_by_key.loc[train_keys].unique()) != {0, 1}:
            raise GMCOOFTrainingError(f"Outer fold {fold} inner train lacks a binary class.")
        if set(int(value) for value in labels_by_key.loc[early_keys].unique()) != {0, 1}:
            raise GMCOOFTrainingError(
                f"Outer fold {fold} inner early-stop validation lacks a binary class."
            )
        scaffold_by_key = outer.set_index("record_key")["scaffold_key"]
        if set(scaffold_by_key.loc[train_keys]) & set(scaffold_by_key.loc[early_keys]):
            raise GMCOOFTrainingError(f"Outer fold {fold} inner split has scaffold leakage.")
        if set(holdout["record_key"]) & set(fold_inner["record_key"]):
            raise GMCOOFTrainingError(f"Outer fold {fold} holdout appears in inner membership.")


def _select_fold_data(
    train: FrozenSupervisedSplit,
    membership: FrozenOOFMembership,
    outer_fold: int,
) -> tuple[FrozenSupervisedSplit, FrozenSupervisedSplit, FrozenSupervisedSplit, pd.DataFrame]:
    fold_inner = membership.inner.loc[membership.inner["outer_fold"] == outer_fold]
    inner_train_keys = set(fold_inner.loc[fold_inner["inner_role"] == "inner_train", "record_key"])
    early_stop_keys = set(
        fold_inner.loc[fold_inner["inner_role"] == "inner_early_stop_validation", "record_key"]
    )
    holdout_manifest = membership.outer.loc[membership.outer["outer_fold"] == outer_fold].copy()
    holdout_keys = set(holdout_manifest["record_key"])
    if inner_train_keys & early_stop_keys or (inner_train_keys | early_stop_keys) & holdout_keys:
        raise GMCOOFTrainingError(f"Outer fold {outer_fold} roles overlap.")
    return (
        _subset_train(train, inner_train_keys),
        _subset_train(train, early_stop_keys),
        _subset_train(train, holdout_keys),
        holdout_manifest,
    )


def _subset_train(split: FrozenSupervisedSplit, keys: set[str]) -> FrozenSupervisedSplit:
    selected = [
        (feature, label)
        for feature, label in zip(split.features, split.labels, strict=True)
        if feature.record_key in keys
    ]
    if len(selected) != len(keys):
        raise GMCOOFTrainingError("Frozen TRAIN subset selection lost a record.")
    return FrozenSupervisedSplit(
        split="train",
        features=tuple(feature for feature, _ in selected),
        labels=np.asarray([label for _, label in selected], dtype=np.float64),
        source_row_count=split.source_row_count,
        feature_manifest_sha256=split.feature_manifest_sha256,
        molecule_status_sha256=split.molecule_status_sha256,
        portable_scaler_sha256=split.portable_scaler_sha256,
    )


def _train_predict_one(
    *,
    runtime: OOFTrainingDependencies,
    architecture: GMCMPNNArchitecture,
    inner_train: FrozenSupervisedSplit,
    inner_early_stop: FrozenSupervisedSplit,
    holdout: FrozenSupervisedSplit,
    output_dir: Path,
    seed: int,
    num_workers: int,
    accelerator: str,
    clock: Callable[[], float],
) -> SeedRunResult:
    started = clock()
    _seed_deterministically(runtime, seed)
    bundle = build_gmc_mpnn_model(chemprop_module=runtime.chemprop, architecture=architecture)
    train_dataset = build_chemprop_dataset(inner_train, bundle, chemprop_module=runtime.chemprop)
    early_dataset = build_chemprop_dataset(
        inner_early_stop, bundle, chemprop_module=runtime.chemprop
    )
    holdout_dataset = build_chemprop_dataset(holdout, bundle, chemprop_module=runtime.chemprop)
    loaders = build_chemprop_dataloaders(
        train_dataset,
        early_dataset,
        chemprop_module=runtime.chemprop,
        architecture=architecture,
        seed=seed,
        num_workers=num_workers,
    )
    holdout_loader = build_chemprop_validation_dataloader(
        holdout_dataset,
        chemprop_module=runtime.chemprop,
        architecture=architecture,
        num_workers=num_workers,
    )
    checkpoint = runtime.lightning.callbacks.ModelCheckpoint(
        dirpath=output_dir,
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
    trainer = runtime.lightning.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=architecture.maximum_epochs,
        deterministic=True,
        callbacks=[checkpoint, early_stopping],
        default_root_dir=output_dir,
        logger=False,
        enable_checkpointing=True,
    )
    trainer.fit(
        bundle.model,
        train_dataloaders=loaders.train_loader,
        val_dataloaders=loaders.validation_loader,
    )
    _validate_checkpoints(output_dir, checkpoint)
    try:
        batches = trainer.predict(
            bundle.model,
            dataloaders=holdout_loader,
            ckpt_path=str(checkpoint.best_model_path),
        )
    except Exception as exc:
        raise GMCOOFTrainingError(f"Best-checkpoint prediction failed for seed {seed}.") from exc
    probabilities = _prediction_batches(batches, len(holdout.features), seed)
    best_val_loss = _finite_float(checkpoint.best_model_score, "best validation loss")
    return SeedRunResult(
        probabilities=probabilities,
        best_val_loss=best_val_loss,
        final_epoch=int(trainer.current_epoch),
        elapsed_seconds=float(clock() - started),
        early_stopping={
            "stopped_early": bool(trainer.should_stop),
            "stopped_epoch": int(early_stopping.stopped_epoch),
            "wait_count": int(early_stopping.wait_count),
            "best_score": _optional_finite_float(early_stopping.best_score),
        },
    )


def _seed_prediction_frame(
    holdout_manifest: pd.DataFrame,
    holdout: FrozenSupervisedSplit,
    seed: int,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    source_rows = {feature.record_key: feature.source_row_index for feature in holdout.features}
    labels = {
        feature.record_key: int(label[0])
        for feature, label in zip(holdout.features, holdout.labels, strict=True)
    }
    frame = holdout_manifest[["record_key", "outer_fold"]].copy()
    frame["source_row_index"] = frame["record_key"].map(source_rows)
    frame["label"] = frame["record_key"].map(labels)
    probabilities_by_key = dict(
        zip((feature.record_key for feature in holdout.features), probabilities, strict=True)
    )
    frame["probability"] = frame["record_key"].map(probabilities_by_key)
    frame["seed"] = seed
    if frame[["source_row_index", "label", "probability"]].isna().any().any():
        raise GMCOOFTrainingError("Per-seed holdout prediction identity join is incomplete.")
    return frame[["record_key", "source_row_index", "label", "outer_fold", "seed", "probability"]]


def _aggregate_oof_predictions(
    seed_frames: Sequence[pd.DataFrame],
    outer_manifest: pd.DataFrame,
) -> pd.DataFrame:
    combined = pd.concat(seed_frames, ignore_index=True)
    if combined.duplicated(["record_key", "seed"]).any():
        raise GMCOOFTrainingError("Duplicate record/seed OOF predictions were produced.")
    expected_rows = EXPECTED_TRAIN_COUNT * len(PRODUCTION_SEEDS)
    if len(combined) != expected_rows:
        raise GMCOOFTrainingError("OOF predictions do not contain exactly five seeds per record.")
    if set(combined["record_key"]) != set(outer_manifest["record_key"]):
        raise GMCOOFTrainingError("OOF prediction identities are incomplete.")
    if set(combined["seed"]) != set(PRODUCTION_SEEDS):
        raise GMCOOFTrainingError("OOF prediction seeds differ from the production ensemble.")
    counts = combined.groupby("record_key")["seed"].nunique()
    if not (counts == len(PRODUCTION_SEEDS)).all():
        raise GMCOOFTrainingError("Every OOF record must have five unique seed predictions.")

    identity = combined.groupby("record_key", sort=False)[
        ["source_row_index", "label", "outer_fold"]
    ].nunique()
    if not (identity == 1).all().all():
        raise GMCOOFTrainingError("Per-seed OOF identity metadata is inconsistent.")
    wide = combined.pivot(index="record_key", columns="seed", values="probability")
    wide = wide.reindex(columns=PRODUCTION_SEEDS)
    if wide.isna().any().any():
        raise GMCOOFTrainingError("At least one OOF seed probability is missing.")
    base = outer_manifest[["record_key", "outer_fold"]].copy()
    metadata = combined.drop_duplicates("record_key").set_index("record_key")
    base["source_row_index"] = base["record_key"].map(metadata["source_row_index"])
    base["label"] = base["record_key"].map(metadata["label"])
    for seed in PRODUCTION_SEEDS:
        base[f"probability_seed{seed}"] = base["record_key"].map(wide[seed])
    probability_columns = [f"probability_seed{seed}" for seed in PRODUCTION_SEEDS]
    matrix = base[probability_columns].to_numpy(dtype=np.float64)
    _validate_prediction_vector(matrix.reshape(-1), matrix.size, "all OOF seeds")
    base["ensemble_probability"] = matrix.mean(axis=1, dtype=np.float64)
    base["seed_probability_std"] = matrix.std(axis=1, ddof=0, dtype=np.float64)
    base = base[
        [
            "record_key",
            "source_row_index",
            "label",
            "outer_fold",
            *probability_columns,
            "ensemble_probability",
            "seed_probability_std",
        ]
    ]
    if len(base) != EXPECTED_TRAIN_COUNT or base["record_key"].duplicated().any():
        raise GMCOOFTrainingError("Final OOF table must contain 1,558 unique records.")
    return base


def _build_oof_summary(
    *,
    config: OOFTrainingConfig,
    membership: FrozenOOFMembership,
    train: FrozenSupervisedSplit,
    scaler: FrozenGGLScaler,
    scaler_summary: Mapping[str, Any],
    architecture: GMCMPNNArchitecture,
    runtime: OOFTrainingDependencies,
    accelerator: str,
    model_runs: Sequence[Mapping[str, Any]],
    predictions: pd.DataFrame,
    predictions_sha256: str,
    elapsed_seconds: float,
    git_commit: str,
) -> dict[str, Any]:
    if len(model_runs) != len(OUTER_FOLDS) * len(PRODUCTION_SEEDS):
        raise GMCOOFTrainingError("Exactly 25 completed fold/seed model runs are required.")
    gpu_available = bool(runtime.torch.cuda.is_available())
    fold_counts = {
        str(fold): {
            "inner_train": int(model_runs[fold * len(PRODUCTION_SEEDS)]["inner_train_count"]),
            "inner_early_stop_validation": int(
                model_runs[fold * len(PRODUCTION_SEEDS)]["inner_early_stop_validation_count"]
            ),
            "outer_holdout": int(model_runs[fold * len(PRODUCTION_SEEDS)]["outer_holdout_count"]),
        }
        for fold in OUTER_FOLDS
    }
    return {
        "oof_training_runner_version": OOF_TRAINING_RUNNER_VERSION,
        "train_count": EXPECTED_TRAIN_COUNT,
        "outer_fold_count": len(OUTER_FOLDS),
        "model_count": len(model_runs),
        "seeds": list(PRODUCTION_SEEDS),
        "architecture": asdict(architecture),
        "hyperparameters": {
            "batch_size": architecture.batch_size,
            "max_epochs": config.max_epochs,
            "num_workers": config.num_workers,
            "checkpoint_monitor": architecture.checkpoint_monitor,
            "checkpoint_mode": architecture.checkpoint_mode,
            "checkpoint_save_top_k": architecture.checkpoint_save_top_k,
            "early_stopping_patience": architecture.early_stopping_patience,
            "optimizer": architecture.optimizer,
            "scheduler": architecture.scheduler,
            "deterministic": True,
        },
        "fold_membership_counts": fold_counts,
        "model_runs": list(model_runs),
        "manifest_hashes": dict(membership.hashes),
        "frozen_train": {
            "feature_manifest_sha256": train.feature_manifest_sha256,
            "molecule_status_sha256": train.molecule_status_sha256,
            "ordered_input_artifact_sha256": scaler_summary.get("ordered_input_artifact_sha256"),
            "training_preprocessing_version": scaler_summary.get("training_preprocessing_version"),
            "standardization_version": scaler_summary.get("standardization_version"),
            "geometry_preprocessing_version": scaler_summary.get("geometry_preprocessing_version"),
            "ggl_preprocessing_version": scaler_summary.get("ggl_preprocessing_version"),
            "rdkit_version": scaler_summary.get("rdkit_version"),
            "model_interface_version": MODEL_INTERFACE_VERSION,
            "model_data_contract_version": MODEL_DATA_CONTRACT_VERSION,
        },
        "frozen_global_scaler": {
            "scaler_version": scaler.scaler_version,
            "portable_scaler_sha256": scaler.portable_scaler_sha256,
            "feature_order": list(scaler.feature_order),
            "fit_population": "all successful atoms from the complete frozen TRAIN set",
            "training_molecule_count": scaler_summary.get("training_molecule_count"),
            "training_atom_count": scaler_summary.get("training_atom_count"),
            "unsupervised_global_preprocessing_component": True,
            "outer_holdout_included_in_preexisting_scaler_fit": True,
            "refit_during_oof_training": False,
        },
        "oof_predictions_sha256": predictions_sha256,
        "probability_aggregation": "unweighted arithmetic mean across five production seeds",
        "probability_standard_deviation_ddof": 0,
        "elapsed_seconds": float(elapsed_seconds),
        "git_commit": git_commit,
        "package_versions": _package_versions(runtime),
        "gpu_cuda": {
            "accelerator": accelerator,
            "gpu_available": gpu_available,
            "gpu_name": runtime.torch.cuda.get_device_name(0) if gpu_available else None,
            "cuda_runtime": runtime.torch.version.cuda,
        },
        "checks": {
            "exactly_1558_unique_oof_records": len(predictions) == EXPECTED_TRAIN_COUNT
            and predictions["record_key"].is_unique,
            "five_seed_predictions_per_record": True,
            "outer_holdout_excluded_from_model_fit_early_stopping_checkpoint_selection": True,
            "outer_scaffold_overlap_absent": True,
            "inner_scaffold_overlap_absent": True,
            "scaler_refit_absent": True,
        },
        "validation_artifact_accessed": False,
        "test_artifact_accessed": False,
    }


def _checkpoint_metadata(output_dir: Path) -> dict[str, Any]:
    best = output_dir / BEST_CHECKPOINT_FILENAME
    last = output_dir / LAST_CHECKPOINT_FILENAME
    if not best.is_file() or not last.is_file():
        raise GMCOOFTrainingError("Training did not produce best.ckpt and last.ckpt.")
    return {
        "best_checkpoint": BEST_CHECKPOINT_FILENAME,
        "best_checkpoint_sha256": _sha256_file(best),
        "last_checkpoint": LAST_CHECKPOINT_FILENAME,
        "last_checkpoint_sha256": _sha256_file(last),
    }


def _validate_checkpoints(output_dir: Path, checkpoint: Any) -> None:
    _checkpoint_metadata(output_dir)
    if Path(str(checkpoint.best_model_path)).name != BEST_CHECKPOINT_FILENAME:
        raise GMCOOFTrainingError("Lightning did not select best.ckpt as the best checkpoint.")


def _prediction_batches(batches: Any, expected_count: int, seed: int) -> np.ndarray:
    if not isinstance(batches, Sequence) or isinstance(batches, (str, bytes)) or not batches:
        raise GMCOOFTrainingError(f"Seed {seed} returned no prediction batches.")
    arrays = [_to_numpy(batch) for batch in batches]
    try:
        probabilities = np.concatenate(arrays, axis=0).reshape(-1).astype(np.float64, copy=False)
    except ValueError as exc:
        raise GMCOOFTrainingError(
            f"Seed {seed} prediction batches have incompatible shapes."
        ) from exc
    return _validate_prediction_vector(probabilities, expected_count, f"seed {seed}")


def _validate_prediction_vector(
    values: Any,
    expected_count: int,
    label: str,
) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float64).reshape(-1)
    if probabilities.shape != (expected_count,):
        raise GMCOOFTrainingError(f"{label} returned an incompatible prediction count.")
    if not np.isfinite(probabilities).all():
        raise GMCOOFTrainingError(f"{label} probabilities contain NaN or infinity.")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise GMCOOFTrainingError(f"{label} probabilities fall outside [0, 1].")
    return probabilities


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _load_dependencies() -> OOFTrainingDependencies:
    try:
        import chemprop
        import lightning.pytorch as lightning
        import torch
    except ImportError as exc:  # pragma: no cover - ECHO runtime boundary
        raise GMCOOFTrainingError(
            "The validated Chemprop 2.1.0/Lightning 2.1.4/Torch 2.1.2 environment is required."
        ) from exc
    return OOFTrainingDependencies(chemprop=chemprop, lightning=lightning, torch=torch)


def _require_runtime(runtime: OOFTrainingDependencies) -> None:
    if getattr(runtime.chemprop, "__version__", None) != EXPECTED_CHEMPROP_VERSION:
        raise GMCOOFTrainingError(f"OOF training requires Chemprop {EXPECTED_CHEMPROP_VERSION}.")


def _seed_deterministically(runtime: OOFTrainingDependencies, seed: int) -> None:
    runtime.lightning.seed_everything(seed, workers=True)
    runtime.torch.use_deterministic_algorithms(True)
    if hasattr(runtime.torch.backends, "cudnn"):
        runtime.torch.backends.cudnn.deterministic = True
        runtime.torch.backends.cudnn.benchmark = False


def _package_versions(runtime: OOFTrainingDependencies) -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "chemprop": str(runtime.chemprop.__version__),
        "lightning": _distribution_version("lightning", runtime.lightning),
        "torch": str(runtime.torch.__version__),
        "numpy": _distribution_version("numpy"),
        "scikit_learn": _distribution_version("scikit-learn"),
        "rdkit": _distribution_version("rdkit"),
        "setuptools": _distribution_version("setuptools"),
    }


def _distribution_version(name: str, module: Any | None = None) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        observed = getattr(module, "__version__", None)
        return str(observed) if observed is not None else "unavailable"


def _finite_float(value: Any, label: str) -> float:
    result = _optional_finite_float(value)
    if result is None:
        raise GMCOOFTrainingError(f"Training did not report a finite {label}.")
    return result


def _optional_finite_float(value: Any) -> float | None:
    if value is None:
        return None
    for method in ("detach", "cpu", "item"):
        if hasattr(value, method):
            value = getattr(value, method)()
    result = float(value)
    if not np.isfinite(result):
        raise GMCOOFTrainingError("Training metric is NaN or infinite.")
    return result


def _require_false(payload: Mapping[str, Any], field: str, label: str) -> None:
    if field not in payload or payload[field] is not False:
        raise GMCOOFTrainingError(f"{label} {field!r} must exist and be exactly false.")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GMCOOFTrainingError(f"Cannot read {label}: {path.name}") from exc
    if not isinstance(value, dict):
        raise GMCOOFTrainingError(f"{label} must be a JSON object.")
    return value


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, keep_default_na=False)
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        raise GMCOOFTrainingError(f"Cannot read {label}: {path.name}") from exc


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise GMCOOFTrainingError(f"Cannot hash required file: {path.name}") from exc


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
        raise GMCOOFTrainingError("Unable to record the Git commit.") from exc


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
