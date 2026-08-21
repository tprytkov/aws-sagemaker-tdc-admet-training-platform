from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from admet_platform.gmc_mpnn import geometry, ggl, scaling
from admet_platform.gmc_mpnn.model import (
    MODEL_INTERFACE_VERSION,
    GMCMPNNArchitecture,
)
from admet_platform.gmc_mpnn.model_data import MODEL_DATA_CONTRACT_VERSION
from scripts import evaluate_gmc_mpnn_bbb_validation as evaluator


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


class _Trainer:
    instances: list[_Trainer] = []
    probabilities: dict[int, np.ndarray] = {}

    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        self.predict_calls: list[dict[str, Any]] = []
        self.__class__.instances.append(self)

    def predict(self, model: Any, **kwargs: Any) -> list[np.ndarray]:
        match = re.search(r"seed(\d+)", str(kwargs["ckpt_path"]))
        assert match is not None
        seed = int(match.group(1))
        self.predict_calls.append({"model": model, **kwargs})
        values = self.probabilities[seed]
        return [values[:100, None], values[100:, None]]


class _Lightning:
    __version__ = "2.1.4"

    def __init__(self):
        self.Trainer = _Trainer
        self.seed_calls: list[tuple[int, bool]] = []

    def seed_everything(self, seed: int, *, workers: bool) -> None:
        self.seed_calls.append((seed, workers))


def test_five_seed_validation_evaluation_outputs_and_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen, config, dependencies, state = _install_fakes(tmp_path, monkeypatch)

    result = evaluator.evaluate_validation(
        config,
        dependencies=dependencies,
        git_commit="f" * 40,
    )

    predictions = result["validation_predictions"]
    assert list(predictions.columns) == [
        "record_key",
        "molecule_id",
        "canonical_smiles",
        "true_label",
        "probability_seed13",
        "probability_seed37",
        "probability_seed73",
        "probability_seed101",
        "probability_seed137",
        "ensemble_probability",
        "probability_standard_deviation",
    ]
    assert len(predictions) == 196
    assert predictions["record_key"].tolist() == [
        feature.record_key for feature in frozen.validation.features
    ]
    np.testing.assert_array_equal(predictions["true_label"], frozen.validation.labels.reshape(-1))
    expected_ensemble = np.where(predictions["true_label"].to_numpy() == 1, 0.8, 0.2)
    np.testing.assert_allclose(predictions["ensemble_probability"], expected_ensemble)
    expected_std = float(np.std([-0.04, -0.02, 0.0, 0.02, 0.04], ddof=0))
    np.testing.assert_allclose(predictions["probability_standard_deviation"], expected_std)

    metrics = result["validation_metrics"]
    assert metrics["validation_count"] == 196
    assert metrics["seeds"] == [13, 37, 73, 101, 137]
    assert metrics["threshold"] == 0.5
    assert metrics["test_artifact_accessed"] is False
    assert set(metrics["metrics"]) == {
        "seed13",
        "seed37",
        "seed73",
        "seed101",
        "seed137",
        "ensemble",
    }
    for values in metrics["metrics"].values():
        assert values["auroc"] == pytest.approx(1.0)
        assert values["auprc"] == pytest.approx(1.0)
        assert values["accuracy"] == pytest.approx(1.0)
        assert values["balanced_accuracy"] == pytest.approx(1.0)
        assert values["sensitivity"] == pytest.approx(1.0)
        assert values["specificity"] == pytest.approx(1.0)
        assert values["mcc"] == pytest.approx(1.0)
        assert values["confusion_matrix"] == [[98, 0], [0, 98]]
        assert values["log_loss"] == values["binary_cross_entropy"]
        assert values["brier_score"] >= 0.0

    ensemble = result["ensemble_summary"]
    assert ensemble["ensemble_method"] == "unweighted_arithmetic_mean"
    assert ensemble["probability_standard_deviation_ddof"] == 0
    assert ensemble["mean_probability_standard_deviation"] == pytest.approx(expected_std)
    assert ensemble["maximum_probability_standard_deviation"] == pytest.approx(expected_std)
    assert ensemble["test_artifact_accessed"] is False

    provenance = metrics["provenance"]
    assert provenance["validation_count"] == 196
    assert provenance["git_commit"] == "f" * 40
    assert provenance["frozen_validation"] == {
        "model_interface_version": MODEL_INTERFACE_VERSION,
        "model_data_contract_version": MODEL_DATA_CONTRACT_VERSION,
        "feature_manifest_sha256": "a" * 64,
        "molecule_status_sha256": "b" * 64,
        "scaler_version": "gmc-mpnn-ggl-standard-scaler-v1",
        "portable_scaler_sha256": "c" * 64,
        "feature_order": [
            "minimum",
            "maximum",
            "sum",
            "mean",
            "median",
            "population_standard_deviation",
        ],
    }
    assert set(provenance["checkpoints"]) == {"13", "37", "73", "101", "137"}
    assert all(len(value["sha256"]) == 64 for value in provenance["checkpoints"].values())
    assert provenance["gpu_cuda"] == {
        "accelerator": "gpu",
        "gpu_available": True,
        "gpu_name": "Synthetic GPU",
        "cuda_runtime": "12.1",
    }

    output = config.output_dir
    assert sorted(path.name for path in output.iterdir()) == [
        "ensemble_summary.json",
        "validation_metrics.json",
        "validation_predictions.csv",
    ]
    persisted_predictions = pd.read_csv(output / "validation_predictions.csv")
    assert len(persisted_predictions) == 196
    assert json.loads((output / "validation_metrics.json").read_text(encoding="utf-8")) == metrics
    assert json.loads((output / "ensemble_summary.json").read_text(encoding="utf-8")) == ensemble

    assert state["validation_load_args"] == (Path("frozen-validation"), Path("frozen-scaler"))
    assert state["validation_loader_kwargs"]["num_workers"] == 0
    assert dependencies.lightning.seed_calls == [
        (13, True),
        (37, True),
        (73, True),
        (101, True),
        (137, True),
    ]
    assert dependencies.torch.deterministic_calls == [True] * 5
    assert len(_Trainer.instances) == 5
    assert all(instance.kwargs["deterministic"] is True for instance in _Trainer.instances)
    assert all(instance.kwargs["enable_checkpointing"] is False for instance in _Trainer.instances)


def test_missing_checkpoint_fails_before_data_or_runtime_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, config, _, state = _install_fakes(tmp_path, monkeypatch)
    config.checkpoint_paths[2][1].unlink()

    with pytest.raises(FileNotFoundError, match="seed 73"):
        evaluator.evaluate_validation(config)
    assert "validation_load_args" not in state


def test_existing_output_directory_is_rejected_before_checkpoint_access(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    config = evaluator.EvaluationConfig(
        validation_preprocessing_dir=Path("validation"),
        scaler_dir=Path("scaler"),
        checkpoint_paths=tuple(
            (seed, tmp_path / f"missing-seed{seed}.ckpt") for seed in evaluator.EXPECTED_SEEDS
        ),
        output_dir=output,
    )

    with pytest.raises(FileExistsError, match="already exists"):
        evaluator.evaluate_validation(config)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -0.01, 1.01])
def test_nonfinite_or_out_of_range_predictions_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_value: float,
) -> None:
    _, config, dependencies, _ = _install_fakes(tmp_path, monkeypatch)
    _Trainer.probabilities[37] = _Trainer.probabilities[37].copy()
    _Trainer.probabilities[37][0] = bad_value

    with pytest.raises(evaluator.GMCValidationEvaluationError, match="seed 37 probabilities"):
        evaluator.evaluate_validation(config, dependencies=dependencies, git_commit="e" * 40)
    assert not config.output_dir.exists()


def test_checkpoint_architecture_or_frozen_provenance_mismatch_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, config, dependencies, _ = _install_fakes(tmp_path, monkeypatch)
    summary_path = config.checkpoint_paths[0][1].parent / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["architecture"]["aggregation_norm"] = 1.0
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(evaluator.GMCValidationEvaluationError, match="incompatible architecture"):
        evaluator.evaluate_validation(config, dependencies=dependencies, git_commit="d" * 40)
    assert not _Trainer.instances


def test_validation_count_mismatch_is_rejected_without_prediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, config, dependencies, _ = _install_fakes(tmp_path, monkeypatch, validation_count=195)

    with pytest.raises(evaluator.GMCValidationEvaluationError, match="exactly 196"):
        evaluator.evaluate_validation(config, dependencies=dependencies, git_commit="d" * 40)
    assert not _Trainer.instances


def test_evaluator_never_loads_development_test_or_recomputes_preprocessing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, config, dependencies, _ = _install_fakes(tmp_path, monkeypatch)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Validation evaluation must not access other data or preprocessing.")

    monkeypatch.setattr(geometry, "generate_deterministic_geometry", forbidden)
    monkeypatch.setattr(ggl, "compute_ggl_features", forbidden)
    monkeypatch.setattr(scaling.StandardScaler, "fit", forbidden)
    monkeypatch.setattr(scaling.StandardScaler, "fit_transform", forbidden)
    monkeypatch.setattr(scaling.StandardScaler, "partial_fit", forbidden)

    result = evaluator.evaluate_validation(
        config,
        dependencies=dependencies,
        git_commit="c" * 40,
    )

    assert result["validation_metrics"]["test_artifact_accessed"] is False


def _install_fakes(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    validation_count: int = 196,
) -> tuple[Any, evaluator.EvaluationConfig, evaluator.EvaluationDependencies, dict[str, Any]]:
    _Trainer.instances.clear()
    state: dict[str, Any] = {}
    labels = np.asarray([index % 2 for index in range(validation_count)], dtype=np.float64).reshape(
        -1, 1
    )
    features = tuple(
        SimpleNamespace(
            record_key=f"record-{index:03d}",
            molecule_id=f"molecule-{index:03d}",
            canonical_smiles="C" if index % 2 == 0 else "CC",
        )
        for index in range(validation_count)
    )
    frozen = SimpleNamespace(
        validation=SimpleNamespace(
            features=features,
            labels=labels,
            feature_manifest_sha256="a" * 64,
            molecule_status_sha256="b" * 64,
        ),
        scaler=SimpleNamespace(
            scaler_version="gmc-mpnn-ggl-standard-scaler-v1",
            portable_scaler_sha256="c" * 64,
            feature_order=(
                "minimum",
                "maximum",
                "sum",
                "mean",
                "median",
                "population_standard_deviation",
            ),
        ),
    )
    offsets = dict(zip(evaluator.EXPECTED_SEEDS, (-0.04, -0.02, 0.0, 0.02, 0.04), strict=True))
    base = np.where(labels.reshape(-1) == 1, 0.8, 0.2)
    _Trainer.probabilities = {seed: base + offset for seed, offset in offsets.items()}

    checkpoint_paths: list[tuple[int, Path]] = []
    for seed in evaluator.EXPECTED_SEEDS:
        run_dir = root / f"gmc_mpnn_bbb_training_seed{seed}"
        run_dir.mkdir()
        checkpoint = run_dir / "best.ckpt"
        checkpoint.write_bytes(f"checkpoint-{seed}".encode())
        summary = {
            "seed": seed,
            "architecture": asdict(GMCMPNNArchitecture()),
            "train_successful_rows": 1558,
            "validation_successful_rows": 196,
            "best_checkpoint": "best.ckpt",
            "test_artifact_accessed": False,
            "hyperparameters": {
                "batch_size": 32,
                "max_epochs": 100,
                "checkpoint_monitor": "val_loss",
                "checkpoint_mode": "min",
                "checkpoint_save_top_k": 1,
                "early_stopping_patience": 10,
                "optimizer": "Adam",
                "scheduler": "Chemprop Noam-like",
            },
            "frozen_provenance": {
                "model_interface_version": MODEL_INTERFACE_VERSION,
                "model_data_contract_version": MODEL_DATA_CONTRACT_VERSION,
                "validation_feature_manifest_sha256": "a" * 64,
                "validation_molecule_status_sha256": "b" * 64,
                "scaler_version": "gmc-mpnn-ggl-standard-scaler-v1",
                "portable_scaler_sha256": "c" * 64,
                "feature_order": list(frozen.scaler.feature_order),
            },
        }
        (run_dir / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        checkpoint_paths.append((seed, checkpoint))

    def fake_validation_load(*args: Any, **kwargs: Any) -> Any:
        state["validation_load_args"] = args
        state["validation_load_kwargs"] = kwargs
        return frozen

    def fake_model(*, chemprop_module: Any, architecture: GMCMPNNArchitecture) -> Any:
        state["model_architecture"] = architecture
        return SimpleNamespace(model="canonical-model", featurizer="canonical-featurizer")

    def fake_dataset(split: Any, model_bundle: Any, *, chemprop_module: Any) -> str:
        assert split is frozen.validation
        del model_bundle, chemprop_module
        return "validation-dataset"

    def fake_validation_loader(dataset: Any, **kwargs: Any) -> str:
        assert dataset == "validation-dataset"
        state["validation_loader_kwargs"] = kwargs
        return "validation-loader"

    monkeypatch.setattr(evaluator, "load_frozen_validation_features", fake_validation_load)
    monkeypatch.setattr(evaluator, "build_gmc_mpnn_model", fake_model)
    monkeypatch.setattr(evaluator, "build_chemprop_dataset", fake_dataset)
    monkeypatch.setattr(
        evaluator,
        "build_chemprop_validation_dataloader",
        fake_validation_loader,
    )
    dependencies = evaluator.EvaluationDependencies(
        chemprop=SimpleNamespace(__version__="2.1.0"),
        lightning=_Lightning(),
        torch=_Torch(),
    )
    config = evaluator.EvaluationConfig(
        validation_preprocessing_dir=Path("frozen-validation"),
        scaler_dir=Path("frozen-scaler"),
        checkpoint_paths=tuple(checkpoint_paths),
        output_dir=root / "evaluation-output",
    )
    return frozen, config, dependencies, state
