from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from admet_platform.gmc_mpnn import geometry, ggl, scaling
from admet_platform.gmc_mpnn.model import GMCMPNNArchitecture
from scripts import train_gmc_mpnn_bbb as runner


class _Checkpoint:
    instances: list[_Checkpoint] = []

    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        self.best_model_path = ""
        self.best_model_score = None
        self.__class__.instances.append(self)


class _EarlyStopping:
    instances: list[_EarlyStopping] = []

    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        self.stopped_epoch = 0
        self.wait_count = 3
        self.best_score = 0.25
        self.__class__.instances.append(self)


class _Trainer:
    instances: list[_Trainer] = []

    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        self.current_epoch = int(kwargs["max_epochs"])
        self.should_stop = False
        self.fit_call: dict[str, Any] | None = None
        self.__class__.instances.append(self)

    def fit(self, model: Any, **kwargs: Any) -> None:
        self.fit_call = {"model": model, **kwargs}
        checkpoint = next(
            callback for callback in self.kwargs["callbacks"] if isinstance(callback, _Checkpoint)
        )
        directory = Path(checkpoint.kwargs["dirpath"])
        (directory / "best.ckpt").write_bytes(b"best")
        (directory / "last.ckpt").write_bytes(b"last")
        checkpoint.best_model_path = str(directory / "best.ckpt")
        checkpoint.best_model_score = 0.2


class _Cuda:
    def is_available(self) -> bool:
        return True

    def get_device_name(self, index: int) -> str:
        assert index == 0
        return "Synthetic GPU"


class _Torch:
    __version__ = "2.1.2+cu121"

    def __init__(self):
        self.cuda = _Cuda()
        self.version = SimpleNamespace(cuda="12.1")
        self.backends = SimpleNamespace(cudnn=SimpleNamespace(deterministic=False, benchmark=True))
        self.deterministic_calls: list[bool] = []

    def use_deterministic_algorithms(self, enabled: bool) -> None:
        self.deterministic_calls.append(enabled)


class _Lightning:
    __version__ = "2.1.4"

    def __init__(self):
        self.callbacks = SimpleNamespace(
            ModelCheckpoint=_Checkpoint,
            EarlyStopping=_EarlyStopping,
        )
        self.Trainer = _Trainer
        self.seed_calls: list[tuple[int, bool]] = []

    def seed_everything(self, seed: int, *, workers: bool) -> None:
        self.seed_calls.append((seed, workers))


def test_training_runner_contract_and_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state, dependencies = _install_frozen_fakes(monkeypatch)
    output = tmp_path / "seed13"
    ticks = iter((10.0, 12.5))
    config = _config(output, seed=13, max_epochs=2)

    summary = runner.run_training(
        config,
        dependencies=dependencies,
        clock=lambda: next(ticks),
        git_commit="a" * 40,
    )

    architecture = summary["architecture"]
    assert summary["seed"] == 13
    assert summary["train_successful_rows"] == 1558
    assert summary["validation_successful_rows"] == 196
    assert architecture == {
        **runner.asdict(GMCMPNNArchitecture(maximum_epochs=2)),
    }
    assert architecture["ordinary_atom_feature_dim"] == 72
    assert architecture["extra_atom_feature_dim"] == 6
    assert architecture["atom_input_dim"] == 78
    assert architecture["bond_input_dim"] == 14
    assert architecture["hidden_dim"] == 300
    assert architecture["message_passing_depth"] == 5
    assert architecture["aggregation_norm"] == 57.0
    assert architecture["loss"] == "binary_cross_entropy_with_logits"
    assert architecture["ffn_layers"] == 2
    assert architecture["ffn_hidden_dim"] == 900
    assert architecture["ffn_activation"] == "LEAKYRELU"
    assert architecture["optimizer"] == "Adam"
    assert architecture["scheduler"] == "Chemprop Noam-like"

    hyperparameters = summary["hyperparameters"]
    assert hyperparameters["batch_size"] == 32
    assert hyperparameters["max_epochs"] == 2
    assert hyperparameters["checkpoint_monitor"] == "val_loss"
    assert hyperparameters["checkpoint_mode"] == "min"
    assert hyperparameters["checkpoint_save_top_k"] == 1
    assert hyperparameters["early_stopping_patience"] == 10

    checkpoint = _Checkpoint.instances[-1]
    assert checkpoint.kwargs["filename"] == "best"
    assert checkpoint.kwargs["monitor"] == "val_loss"
    assert checkpoint.kwargs["mode"] == "min"
    assert checkpoint.kwargs["save_top_k"] == 1
    assert checkpoint.kwargs["save_last"] is True
    early_stopping = _EarlyStopping.instances[-1]
    assert early_stopping.kwargs == {
        "monitor": "val_loss",
        "mode": "min",
        "patience": 10,
    }
    trainer = _Trainer.instances[-1]
    assert trainer.kwargs["max_epochs"] == 2
    assert trainer.kwargs["deterministic"] is True
    assert trainer.kwargs["accelerator"] == "gpu"
    assert trainer.fit_call == {
        "model": "synthetic-model",
        "train_dataloaders": "train-loader",
        "val_dataloaders": "validation-loader",
    }

    lightning = dependencies.lightning
    torch = dependencies.torch
    assert lightning.seed_calls == [(13, True)]
    assert torch.deterministic_calls == [True]
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert state["loader_kwargs"] == {
        "chemprop_module": dependencies.chemprop,
        "architecture": GMCMPNNArchitecture(maximum_epochs=2),
        "seed": 13,
        "num_workers": 0,
    }
    assert state["load_args"] == (
        Path("frozen-train"),
        Path("frozen-validation"),
        Path("frozen-scaler"),
    )
    assert all("test" not in str(path).lower() for path in state["load_args"])

    required_summary_keys = {
        "seed",
        "architecture",
        "hyperparameters",
        "train_successful_rows",
        "validation_successful_rows",
        "data_counts",
        "frozen_provenance",
        "package_versions",
        "git_commit",
        "gpu_cuda",
        "best_val_loss",
        "best_checkpoint",
        "last_checkpoint",
        "final_epoch",
        "elapsed_seconds",
        "early_stopping",
        "test_artifact_accessed",
    }
    assert required_summary_keys.issubset(summary)
    assert summary["best_val_loss"] == pytest.approx(0.2)
    assert summary["best_checkpoint"] == "best.ckpt"
    assert summary["last_checkpoint"] == "last.ckpt"
    assert summary["final_epoch"] == 2
    assert summary["elapsed_seconds"] == pytest.approx(2.5)
    assert summary["test_artifact_accessed"] is False
    assert summary["data_counts"] == {
        "train_source_rows": 1561,
        "train_successful_rows": 1558,
        "train_heavy_atoms": 38245,
        "train_excluded_rows": 2,
        "train_failed_rows": 1,
        "validation_source_rows": 196,
        "validation_successful_rows": 196,
        "validation_heavy_atoms": 3755,
        "validation_excluded_rows": 0,
        "validation_failed_rows": 0,
    }
    assert {
        "python",
        "chemprop",
        "lightning",
        "torch",
        "numpy",
        "scikit_learn",
        "rdkit",
        "setuptools",
        "pytest",
    }.issubset(summary["package_versions"])
    assert summary["frozen_provenance"]["portable_scaler_sha256"] == "e" * 64
    assert summary["gpu_cuda"] == {
        "accelerator": "gpu",
        "gpu_available": True,
        "gpu_name": "Synthetic GPU",
        "cuda_runtime": "12.1",
    }
    assert (output / "best.ckpt").read_bytes() == b"best"
    assert (output / "last.ckpt").read_bytes() == b"last"
    persisted = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    assert persisted == summary


def test_output_collision_fails_before_runtime_or_data_access(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        runner.run_training(_config(output, seed=13))


def test_seed_is_required_and_max_epochs_defaults_to_100() -> None:
    parser = runner.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--output-dir", "out"])

    args = parser.parse_args(["--seed", "37", "--output-dir", "out"])
    assert args.seed == 37
    assert args.max_epochs == 100


def test_invalid_seed_and_epoch_values_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="seed"):
        _config(tmp_path / "negative-seed", seed=-1)
    with pytest.raises(ValueError, match="max_epochs"):
        _config(tmp_path / "zero-epochs", seed=13, max_epochs=0)


def test_wrong_frozen_counts_fail_without_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, dependencies = _install_frozen_fakes(monkeypatch, train_count=1557)

    with pytest.raises(runner.GMCTrainingRunnerError, match="1,558"):
        runner.run_training(
            _config(tmp_path / "wrong-count", seed=13),
            dependencies=dependencies,
        )
    assert not _Trainer.instances


def test_runner_never_calls_geometry_ggl_or_scaler_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, dependencies = _install_frozen_fakes(monkeypatch)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Training runner must not recompute or fit preprocessing.")

    monkeypatch.setattr(geometry, "generate_deterministic_geometry", forbidden)
    monkeypatch.setattr(ggl, "compute_ggl_features", forbidden)
    monkeypatch.setattr(scaling.StandardScaler, "fit", forbidden)
    monkeypatch.setattr(scaling.StandardScaler, "fit_transform", forbidden)
    monkeypatch.setattr(scaling.StandardScaler, "partial_fit", forbidden)
    ticks = iter((1.0, 2.0))

    summary = runner.run_training(
        _config(tmp_path / "safe-run", seed=137, max_epochs=1),
        dependencies=dependencies,
        clock=lambda: next(ticks),
        git_commit="b" * 40,
    )

    assert summary["test_artifact_accessed"] is False


def _config(output: Path, *, seed: int, max_epochs: int = 100) -> runner.TrainingConfig:
    return runner.TrainingConfig(
        train_preprocessing_dir=Path("frozen-train"),
        validation_preprocessing_dir=Path("frozen-validation"),
        scaler_dir=Path("frozen-scaler"),
        output_dir=output,
        seed=seed,
        max_epochs=max_epochs,
    )


def _install_frozen_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    train_count: int = 1558,
    validation_count: int = 196,
) -> tuple[dict[str, Any], runner.TrainingDependencies]:
    _Checkpoint.instances.clear()
    _EarlyStopping.instances.clear()
    _Trainer.instances.clear()
    state: dict[str, Any] = {}
    scaler = SimpleNamespace(
        scaler_version="gmc-mpnn-ggl-standard-scaler-v1",
        portable_scaler_sha256="e" * 64,
        feature_order=(
            "minimum",
            "maximum",
            "sum",
            "mean",
            "median",
            "population_standard_deviation",
        ),
    )
    frozen = SimpleNamespace(
        train=SimpleNamespace(
            features=(None,) * train_count,
            feature_manifest_sha256="a" * 64,
            molecule_status_sha256="b" * 64,
        ),
        validation=SimpleNamespace(
            features=(None,) * validation_count,
            feature_manifest_sha256="c" * 64,
            molecule_status_sha256="d" * 64,
        ),
        scaler=scaler,
    )

    def fake_load(*args: Any, **kwargs: Any) -> Any:
        state["load_args"] = args
        state["load_kwargs"] = kwargs
        return frozen

    def fake_model(*, chemprop_module: Any, architecture: GMCMPNNArchitecture) -> Any:
        state["architecture"] = architecture
        return SimpleNamespace(model="synthetic-model", featurizer="synthetic-featurizer")

    def fake_dataset(split: Any, model_bundle: Any, *, chemprop_module: Any) -> str:
        del model_bundle, chemprop_module
        return "train-dataset" if split is frozen.train else "validation-dataset"

    def fake_loaders(train_dataset: Any, validation_dataset: Any, **kwargs: Any) -> Any:
        assert train_dataset == "train-dataset"
        assert validation_dataset == "validation-dataset"
        state["loader_kwargs"] = kwargs
        return SimpleNamespace(
            train_loader="train-loader",
            validation_loader="validation-loader",
        )

    monkeypatch.setattr(runner, "load_frozen_development_features", fake_load)
    monkeypatch.setattr(runner, "build_gmc_mpnn_model", fake_model)
    monkeypatch.setattr(runner, "build_chemprop_dataset", fake_dataset)
    monkeypatch.setattr(runner, "build_chemprop_dataloaders", fake_loaders)

    dependencies = runner.TrainingDependencies(
        chemprop=SimpleNamespace(__version__="2.1.0"),
        lightning=_Lightning(),
        torch=_Torch(),
    )
    return state, dependencies
