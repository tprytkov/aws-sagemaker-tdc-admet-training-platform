from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from admet_platform.gmc_mpnn import geometry, ggl, scaling
from admet_platform.gmc_mpnn.model import GMCMPNNArchitecture
from admet_platform.gmc_mpnn.model_data import FrozenSupervisedSplit
from scripts import train_gmc_mpnn_bbb_oof as runner


class _Cuda:
    def is_available(self) -> bool:
        return True

    def get_device_name(self, index: int) -> str:
        assert index == 0
        return "Synthetic GPU"


class _Torch:
    __version__ = "2.1.2+cu121"

    def __init__(self) -> None:
        self.cuda = _Cuda()
        self.version = SimpleNamespace(cuda="12.1")
        self.backends = SimpleNamespace(cudnn=SimpleNamespace(deterministic=False, benchmark=True))

    def use_deterministic_algorithms(self, enabled: bool) -> None:
        assert enabled is True


class _Lightning:
    __version__ = "2.1.4"

    def seed_everything(self, seed: int, *, workers: bool) -> None:
        assert seed in runner.PRODUCTION_SEEDS
        assert workers is True


def test_complete_five_fold_five_seed_oof_orchestration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    membership, train, scaler, scaler_summary = _synthetic_contract()
    dependencies = runner.OOFTrainingDependencies(
        chemprop=SimpleNamespace(__version__="2.1.0"),
        lightning=_Lightning(),
        torch=_Torch(),
    )
    calls: list[tuple[int, int, int, int]] = []

    monkeypatch.setattr(runner, "_load_oof_membership", lambda _: membership)
    monkeypatch.setattr(
        runner,
        "_load_frozen_train",
        lambda _: (scaler, scaler_summary, train),
    )

    def fake_train_predict(**kwargs: Any) -> runner.SeedRunResult:
        output_dir = kwargs["output_dir"]
        seed = kwargs["seed"]
        fold = int(output_dir.parent.name.removeprefix("fold"))
        inner_train = kwargs["inner_train"]
        early = kwargs["inner_early_stop"]
        holdout = kwargs["holdout"]
        calls.append((fold, seed, len(inner_train.features), len(holdout.features)))
        assert not set(feature.record_key for feature in holdout.features) & set(
            feature.record_key for feature in inner_train.features
        )
        assert not set(feature.record_key for feature in holdout.features) & set(
            feature.record_key for feature in early.features
        )
        (output_dir / runner.BEST_CHECKPOINT_FILENAME).write_bytes(f"best-{fold}-{seed}".encode())
        (output_dir / runner.LAST_CHECKPOINT_FILENAME).write_bytes(f"last-{fold}-{seed}".encode())
        probability = 0.1 + fold * 0.05 + runner.PRODUCTION_SEEDS.index(seed) * 0.01
        return runner.SeedRunResult(
            probabilities=np.full(len(holdout.features), probability, dtype=np.float64),
            best_val_loss=0.2 + fold * 0.01,
            final_epoch=12,
            elapsed_seconds=1.5,
            early_stopping={
                "stopped_early": True,
                "stopped_epoch": 12,
                "wait_count": 10,
                "best_score": 0.2,
            },
        )

    monkeypatch.setattr(runner, "_train_predict_one", fake_train_predict)
    output = tmp_path / "oof-training"
    ticks = iter((10.0, 20.0))
    summary = runner.run_oof_training(
        _config(output),
        dependencies=dependencies,
        clock=lambda: next(ticks),
        git_commit="a" * 40,
    )

    assert len(calls) == 25
    assert {(fold, seed) for fold, seed, _, _ in calls} == {
        (fold, seed) for fold in runner.OUTER_FOLDS for seed in runner.PRODUCTION_SEEDS
    }
    assert summary["model_count"] == 25
    assert summary["train_count"] == 1558
    assert summary["seeds"] == [13, 37, 73, 101, 137]
    assert summary["elapsed_seconds"] == pytest.approx(10.0)
    assert summary["validation_artifact_accessed"] is False
    assert summary["test_artifact_accessed"] is False
    assert all(summary["checks"].values())
    assert summary["frozen_global_scaler"] == {
        "scaler_version": "gmc-mpnn-ggl-standard-scaler-v1",
        "portable_scaler_sha256": "e" * 64,
        "feature_order": list(scaling.GGL_FEATURE_NAMES),
        "fit_population": "all successful atoms from the complete frozen TRAIN set",
        "training_molecule_count": 1558,
        "training_atom_count": 38245,
        "unsupervised_global_preprocessing_component": True,
        "outer_holdout_included_in_preexisting_scaler_fit": True,
        "refit_during_oof_training": False,
    }
    architecture = summary["architecture"]
    assert architecture == asdict_architecture(GMCMPNNArchitecture())
    assert architecture["atom_input_dim"] == 78
    assert architecture["bond_input_dim"] == 14
    assert architecture["hidden_dim"] == 300
    assert architecture["message_passing_depth"] == 5
    assert architecture["aggregation_norm"] == 57.0
    assert architecture["ffn_layers"] == 2
    assert architecture["ffn_hidden_dim"] == 900
    assert architecture["ffn_activation"] == "LEAKYRELU"
    assert summary["hyperparameters"]["batch_size"] == 32
    assert summary["hyperparameters"]["checkpoint_monitor"] == "val_loss"
    assert summary["hyperparameters"]["checkpoint_mode"] == "min"
    assert summary["hyperparameters"]["checkpoint_save_top_k"] == 1
    assert summary["hyperparameters"]["early_stopping_patience"] == 10

    predictions = pd.read_csv(output / runner.OOF_PREDICTIONS_FILENAME)
    assert len(predictions) == 1558
    assert predictions["record_key"].is_unique
    assert list(predictions.columns) == [
        "record_key",
        "source_row_index",
        "label",
        "outer_fold",
        "probability_seed13",
        "probability_seed37",
        "probability_seed73",
        "probability_seed101",
        "probability_seed137",
        "ensemble_probability",
        "seed_probability_std",
    ]
    probability_columns = [f"probability_seed{seed}" for seed in runner.PRODUCTION_SEEDS]
    np.testing.assert_allclose(
        predictions["ensemble_probability"], predictions[probability_columns].mean(axis=1)
    )
    np.testing.assert_allclose(
        predictions["seed_probability_std"],
        predictions[probability_columns].std(axis=1, ddof=0),
    )
    for fold in runner.OUTER_FOLDS:
        for seed in runner.PRODUCTION_SEEDS:
            seed_dir = output / f"fold{fold}" / f"seed{seed}"
            assert (seed_dir / "best.ckpt").is_file()
            assert (seed_dir / "last.ckpt").is_file()
            assert (seed_dir / "holdout_predictions.csv").is_file()
            seed_summary = json.loads((seed_dir / "run_summary.json").read_text())
            assert len(seed_summary["checkpoints"]["best_checkpoint_sha256"]) == 64
            assert seed_summary["outer_holdout_usage"] == (
                "prediction_only_after_checkpoint_selection"
            )


def test_fold_selection_excludes_holdout_and_preserves_inner_roles() -> None:
    membership, train, _, _ = _synthetic_contract()
    inner_train, early, holdout, _ = runner._select_fold_data(train, membership, 2)
    train_keys = {feature.record_key for feature in inner_train.features}
    early_keys = {feature.record_key for feature in early.features}
    holdout_keys = {feature.record_key for feature in holdout.features}

    assert not train_keys & early_keys
    assert not train_keys & holdout_keys
    assert not early_keys & holdout_keys
    assert train_keys | early_keys | holdout_keys == {
        feature.record_key for feature in train.features
    }


def test_one_seed_uses_canonical_training_and_checkpoint_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    membership, train, _, _ = _synthetic_contract()
    inner_train, early, holdout, _ = runner._select_fold_data(train, membership, 0)
    state: dict[str, Any] = {}

    class Checkpoint:
        def __init__(self, **kwargs: Any):
            self.kwargs = kwargs
            self.best_model_path = ""
            self.best_model_score = None
            state["checkpoint"] = self

    class EarlyStopping:
        def __init__(self, **kwargs: Any):
            self.kwargs = kwargs
            self.stopped_epoch = 9
            self.wait_count = 10
            self.best_score = 0.25
            state["early_stopping"] = self

    class Trainer:
        def __init__(self, **kwargs: Any):
            self.kwargs = kwargs
            self.current_epoch = 19
            self.should_stop = True
            state["trainer"] = self

        def fit(self, model: Any, **kwargs: Any) -> None:
            state["fit"] = (model, kwargs)
            checkpoint = state["checkpoint"]
            directory = Path(checkpoint.kwargs["dirpath"])
            (directory / "best.ckpt").write_bytes(b"best")
            (directory / "last.ckpt").write_bytes(b"last")
            checkpoint.best_model_path = str(directory / "best.ckpt")
            checkpoint.best_model_score = 0.2

        def predict(self, model: Any, **kwargs: Any) -> list[np.ndarray]:
            state["predict"] = (model, kwargs)
            return [np.full((len(holdout.features), 1), 0.4)]

    lightning = _Lightning()
    lightning.callbacks = SimpleNamespace(
        ModelCheckpoint=Checkpoint,
        EarlyStopping=EarlyStopping,
    )
    lightning.Trainer = Trainer
    runtime = runner.OOFTrainingDependencies(
        chemprop=SimpleNamespace(__version__="2.1.0"),
        lightning=lightning,
        torch=_Torch(),
    )
    bundle = SimpleNamespace(model="model", featurizer="featurizer")
    monkeypatch.setattr(runner, "build_gmc_mpnn_model", lambda **kwargs: bundle)
    monkeypatch.setattr(
        runner,
        "build_chemprop_dataset",
        lambda split, model_bundle, **kwargs: split,
    )
    monkeypatch.setattr(
        runner,
        "build_chemprop_dataloaders",
        lambda *args, **kwargs: SimpleNamespace(
            train_loader="train-loader", validation_loader="early-loader"
        ),
    )
    monkeypatch.setattr(
        runner,
        "build_chemprop_validation_dataloader",
        lambda *args, **kwargs: "holdout-loader",
    )
    ticks = iter((2.0, 5.0))
    result = runner._train_predict_one(
        runtime=runtime,
        architecture=GMCMPNNArchitecture(),
        inner_train=inner_train,
        inner_early_stop=early,
        holdout=holdout,
        output_dir=tmp_path,
        seed=13,
        num_workers=0,
        accelerator="gpu",
        clock=lambda: next(ticks),
    )

    checkpoint = state["checkpoint"]
    assert checkpoint.kwargs["monitor"] == "val_loss"
    assert checkpoint.kwargs["mode"] == "min"
    assert checkpoint.kwargs["save_top_k"] == 1
    assert checkpoint.kwargs["save_last"] is True
    assert state["early_stopping"].kwargs == {
        "monitor": "val_loss",
        "mode": "min",
        "patience": 10,
    }
    assert state["trainer"].kwargs["max_epochs"] == 100
    assert state["trainer"].kwargs["deterministic"] is True
    assert state["fit"] == (
        "model",
        {"train_dataloaders": "train-loader", "val_dataloaders": "early-loader"},
    )
    assert state["predict"] == (
        "model",
        {"dataloaders": "holdout-loader", "ckpt_path": str(tmp_path / "best.ckpt")},
    )
    assert result.best_val_loss == pytest.approx(0.2)
    assert result.final_epoch == 19
    assert result.elapsed_seconds == pytest.approx(3.0)
    np.testing.assert_allclose(result.probabilities, 0.4)


def test_scaffold_leakage_fails_before_training() -> None:
    membership, train, scaler, scaler_summary = _synthetic_contract()
    leaked_outer = membership.outer.copy()
    holdout_index = leaked_outer.index[leaked_outer["outer_fold"] == 0][0]
    development_scaffold = leaked_outer.loc[leaked_outer["outer_fold"] != 0, "scaffold_key"].iloc[0]
    leaked_outer.loc[holdout_index, "scaffold_key"] = development_scaffold
    leaked = runner.FrozenOOFMembership(
        outer=leaked_outer,
        inner=membership.inner,
        summary=membership.summary,
        hashes=membership.hashes,
    )

    with pytest.raises(runner.GMCOOFTrainingError, match="scaffold leakage"):
        runner._validate_membership(leaked, train, scaler, scaler_summary)


def test_incomplete_seed_predictions_fail_closed() -> None:
    membership, train, _, _ = _synthetic_contract()
    _, _, holdout, holdout_manifest = runner._select_fold_data(train, membership, 0)
    frames = [
        runner._seed_prediction_frame(
            holdout_manifest,
            holdout,
            seed,
            np.full(len(holdout.features), 0.5),
        )
        for seed in runner.PRODUCTION_SEEDS[:-1]
    ]

    with pytest.raises(runner.GMCOOFTrainingError, match="exactly five seeds"):
        runner._aggregate_oof_predictions(frames, membership.outer)


def test_output_collision_fails_before_manifest_or_data_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "existing"
    output.mkdir()

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("No frozen artifact may load after output collision.")

    monkeypatch.setattr(runner, "_load_oof_membership", forbidden)
    monkeypatch.setattr(runner, "_load_frozen_train", forbidden)
    with pytest.raises(FileExistsError, match="already exists"):
        runner.run_oof_training(_config(output))


def test_frozen_loader_requests_train_and_global_scaler_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, train, scaler, scaler_summary = _synthetic_contract()
    config = runner.OOFTrainingConfig(
        train_preprocessing_dir=tmp_path / "train",
        scaler_dir=tmp_path / "scaler",
        oof_manifest_dir=tmp_path / "oof-manifest",
        output_dir=tmp_path / "output",
    )
    calls: dict[str, Any] = {}

    def fake_load_scaler(path: Path) -> Any:
        calls["scaler_path"] = path
        return scaler

    def fake_read_json(path: Path, label: str) -> dict[str, Any]:
        calls["summary"] = (path, label)
        return scaler_summary

    def fake_load_split(path: Path, **kwargs: Any) -> FrozenSupervisedSplit:
        calls["split"] = (path, kwargs)
        return train

    monkeypatch.setattr(runner, "load_frozen_ggl_scaler", fake_load_scaler)
    monkeypatch.setattr(runner, "_read_json", fake_read_json)
    monkeypatch.setattr(runner, "load_frozen_feature_split", fake_load_split)

    loaded_scaler, loaded_summary, loaded_train = runner._load_frozen_train(config)

    assert loaded_scaler is scaler
    assert loaded_summary is scaler_summary
    assert loaded_train is train
    assert calls["scaler_path"] == config.scaler_dir
    assert calls["summary"] == (
        config.scaler_dir / scaling.FIT_SUMMARY_FILENAME,
        "scaler fit summary",
    )
    assert calls["split"] == (
        config.train_preprocessing_dir,
        {
            "split": "train",
            "scaler": scaler,
            "scaler_summary": scaler_summary,
        },
    )


def test_manifest_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    outer = pd.DataFrame(
        columns=[
            "record_key",
            "molecule_id",
            "canonical_smiles",
            "label",
            "scaffold_key",
            "outer_fold",
        ]
    )
    inner = pd.DataFrame(columns=["outer_fold", "record_key", "inner_role"])
    outer.to_csv(tmp_path / runner.OUTER_MANIFEST_FILENAME, index=False, lineterminator="\n")
    inner.to_csv(tmp_path / runner.INNER_MANIFEST_FILENAME, index=False, lineterminator="\n")
    (tmp_path / runner.SPLIT_SUMMARY_FILENAME).write_text(
        json.dumps(
            {
                "assignment_hashes": {
                    "outer_fold_manifest_sha256": "0" * 64,
                    "inner_split_manifest_sha256": "0" * 64,
                },
                "validation_artifact_accessed": False,
                "test_artifact_accessed": False,
            }
        )
    )

    with pytest.raises(runner.GMCOOFTrainingError, match="does not match"):
        runner._load_oof_membership(tmp_path)


def test_runner_never_recomputes_or_fits_preprocessing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    membership, train, scaler, scaler_summary = _synthetic_contract()
    dependencies = runner.OOFTrainingDependencies(
        chemprop=SimpleNamespace(__version__="2.1.0"),
        lightning=_Lightning(),
        torch=_Torch(),
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("OOF runner must not recompute or fit preprocessing.")

    monkeypatch.setattr(geometry, "generate_deterministic_geometry", forbidden)
    monkeypatch.setattr(ggl, "compute_ggl_features", forbidden)
    monkeypatch.setattr(scaling.StandardScaler, "fit", forbidden)
    monkeypatch.setattr(scaling.StandardScaler, "fit_transform", forbidden)
    monkeypatch.setattr(scaling.StandardScaler, "partial_fit", forbidden)
    monkeypatch.setattr(runner, "_load_oof_membership", lambda _: membership)
    monkeypatch.setattr(runner, "_load_frozen_train", lambda _: (scaler, scaler_summary, train))

    def fake_train_predict(**kwargs: Any) -> runner.SeedRunResult:
        output_dir = kwargs["output_dir"]
        (output_dir / "best.ckpt").write_bytes(b"best")
        (output_dir / "last.ckpt").write_bytes(b"last")
        return runner.SeedRunResult(
            probabilities=np.full(len(kwargs["holdout"].features), 0.5),
            best_val_loss=0.2,
            final_epoch=1,
            elapsed_seconds=0.1,
            early_stopping={},
        )

    monkeypatch.setattr(runner, "_train_predict_one", fake_train_predict)
    ticks = iter((1.0, 2.0))
    summary = runner.run_oof_training(
        _config(tmp_path / "safe"),
        dependencies=dependencies,
        clock=lambda: next(ticks),
        git_commit="b" * 40,
    )
    assert summary["validation_artifact_accessed"] is False
    assert summary["test_artifact_accessed"] is False


def _config(output: Path) -> runner.OOFTrainingConfig:
    return runner.OOFTrainingConfig(
        train_preprocessing_dir=Path("frozen-train"),
        scaler_dir=Path("frozen-scaler"),
        oof_manifest_dir=Path("frozen-oof-manifest"),
        output_dir=output,
    )


def _synthetic_contract() -> tuple[Any, FrozenSupervisedSplit, Any, dict[str, Any]]:
    group_ids = np.arange(runner.EXPECTED_TRAIN_COUNT) % 200
    outer_folds = group_ids % 5
    labels = group_ids % 2
    records = []
    for index in range(runner.EXPECTED_TRAIN_COUNT):
        records.append(
            SimpleNamespace(
                source_row_index=index,
                record_key=f"record-{index:04d}",
                molecule_id=f"molecule-{index:04d}",
                canonical_smiles="C" if labels[index] == 0 else "CC",
            )
        )
    train = FrozenSupervisedSplit(
        split="train",
        features=tuple(records),
        labels=labels.astype(np.float64).reshape(-1, 1),
        source_row_count=1561,
        feature_manifest_sha256="a" * 64,
        molecule_status_sha256="b" * 64,
        portable_scaler_sha256="e" * 64,
    )
    outer = pd.DataFrame(
        {
            "record_key": [record.record_key for record in records],
            "molecule_id": [record.molecule_id for record in records],
            "canonical_smiles": [record.canonical_smiles for record in records],
            "label": labels,
            "scaffold_key": [f"scaffold-{group:03d}" for group in group_ids],
            "outer_fold": outer_folds,
        }
    )
    inner_rows = []
    for fold in runner.OUTER_FOLDS:
        development_groups = sorted(set(group_ids[outer_folds != fold]))
        early_groups = set(development_groups[:20])
        for index, record in enumerate(records):
            if outer_folds[index] == fold:
                continue
            inner_rows.append(
                {
                    "outer_fold": fold,
                    "record_key": record.record_key,
                    "inner_role": (
                        "inner_early_stop_validation"
                        if group_ids[index] in early_groups
                        else "inner_train"
                    ),
                }
            )
    inner = pd.DataFrame(inner_rows)
    summary = {
        "oof_manifest_version": runner.OOF_MANIFEST_VERSION,
        "train_count": 1558,
        "outer_fold_count": 5,
        "inner_fold_count": 8,
        "split_seed": 1729,
        "frozen_train_provenance": {
            "feature_manifest_sha256": "a" * 64,
            "molecule_status_sha256": "b" * 64,
            "ordered_input_artifact_sha256": "c" * 64,
        },
        "checks": {"synthetic_contract_valid": True},
        "validation_artifact_accessed": False,
        "test_artifact_accessed": False,
    }
    membership = runner.FrozenOOFMembership(
        outer=outer,
        inner=inner,
        summary=summary,
        hashes={
            "outer_fold_manifest_sha256": "1" * 64,
            "inner_split_manifest_sha256": "2" * 64,
            "split_summary_sha256": "3" * 64,
        },
    )
    scaler = SimpleNamespace(
        scaler_version="gmc-mpnn-ggl-standard-scaler-v1",
        portable_scaler_sha256="e" * 64,
        feature_order=scaling.GGL_FEATURE_NAMES,
    )
    scaler_summary = {
        "ordered_input_artifact_sha256": "c" * 64,
        "training_preprocessing_version": "gmc-mpnn-training-raw-ggl-v1",
        "standardization_version": "parent-fragment-v1",
        "geometry_preprocessing_version": "rdkit-etkdgv3-mmff94s-retry2000-v2",
        "ggl_preprocessing_version": "released-pooled-six-feature-v1",
        "rdkit_version": "synthetic",
        "training_molecule_count": 1558,
        "training_atom_count": 38245,
        "validation_artifact_accessed": False,
        "test_artifact_accessed": False,
    }
    return membership, train, scaler, scaler_summary


def asdict_architecture(architecture: GMCMPNNArchitecture) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(architecture)
