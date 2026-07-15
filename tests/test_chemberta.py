import json
import subprocess
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from admet_platform.models import chemberta
from admet_platform.models.chemberta import (
    DEFAULT_CHEMBERTA_MODEL,
    ChemBERTaTrainingConfig,
    _build_torch_dataset,
    _checkpoint_score,
    _load_transformers_components,
    train_chemberta_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def test_classification_configuration_defaults() -> None:
    config = ChemBERTaTrainingConfig()

    assert config.model_name == DEFAULT_CHEMBERTA_MODEL
    assert config.max_sequence_length == 128
    assert config.learning_rate == 2e-5
    assert config.training_epochs == 3
    assert config.random_seed == 42
    assert config.to_dict()["model_name"] == DEFAULT_CHEMBERTA_MODEL


def test_regression_configuration_override() -> None:
    config = ChemBERTaTrainingConfig(model_name="local/model", training_epochs=1, development_row_limit=5)

    assert config.model_name == "local/model"
    assert config.training_epochs == 1
    assert config.development_row_limit == 5


def test_task_head_selection_for_classification_and_regression(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []
    _install_fake_transformers(monkeypatch, calls=calls)

    _load_transformers_components("binary_classification", ChemBERTaTrainingConfig(local_files_only=True))
    _load_transformers_components("regression", ChemBERTaTrainingConfig(local_files_only=True))

    assert calls[0]["num_labels"] == 2
    assert calls[0]["problem_type"] == "single_label_classification"
    assert calls[1]["num_labels"] == 1
    assert calls[1]["problem_type"] == "regression"


def test_tokenization_uses_only_canonical_smiles(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = RecordingTokenizer()
    torch = FakeTorch()
    df = pd.DataFrame(
        {
            "molecule_id": ["mol_001"],
            "canonical_smiles": ["CCO"],
            "target": [1],
            "split": ["train"],
            "metadata": ["not_a_feature"],
        }
    )

    _build_torch_dataset(df, "binary_classification", tokenizer, ChemBERTaTrainingConfig(), torch)

    assert tokenizer.inputs == ["CCO"]


def test_metadata_columns_are_not_tokenized_as_features() -> None:
    tokenizer = RecordingTokenizer()
    torch = FakeTorch()
    df = pd.DataFrame(
        {
            "molecule_id": ["mol_001"],
            "canonical_smiles": ["CCN"],
            "target": [0],
            "endpoint_id": ["bbb_martins"],
            "split": ["train"],
        }
    )

    _build_torch_dataset(df, "binary_classification", tokenizer, ChemBERTaTrainingConfig(), torch)

    assert tokenizer.inputs == ["CCN"]
    assert "bbb_martins" not in tokenizer.inputs
    assert "0" not in tokenizer.inputs


def test_checkpoint_metric_fallback_for_one_class_validation() -> None:
    score, metric_name = _checkpoint_score(
        "binary_classification",
        {"roc_auc": None, "accuracy": 0.75},
    )

    assert score == 0.75
    assert metric_name == "accuracy_fallback"


def test_missing_model_cache_error_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_transformers(monkeypatch, fail_load=True)

    with pytest.raises(RuntimeError, match="local-files-only"):
        _load_transformers_components("binary_classification", ChemBERTaTrainingConfig(local_files_only=True))


def test_classification_prediction_schema_and_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fake_training(monkeypatch, "binary_classification")
    split_paths = _write_splits(tmp_path, "binary_classification")
    output_dir = tmp_path / "chemberta_classification"

    train_chemberta_model(
        **split_paths,
        config_path=CONFIG_DIR / "bbb_martins.yaml",
        output_dir=output_dir,
        training_config=ChemBERTaTrainingConfig(training_epochs=1, development_row_limit=2),
    )

    predictions = pd.read_csv(output_dir / "predictions_test.csv")
    assert list(predictions.columns) == [
        "molecule_id",
        "canonical_smiles",
        "observed_target",
        "predicted_class",
        "predicted_probability",
    ]
    _assert_required_artifacts(output_dir)
    metadata = _read_json(output_dir / "training_metadata.json")
    assert metadata["development_row_limit"] == 2
    assert metadata["warnings"]


def test_regression_prediction_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fake_training(monkeypatch, "regression")
    split_paths = _write_splits(tmp_path, "regression")
    output_dir = tmp_path / "chemberta_regression"

    train_chemberta_model(
        **split_paths,
        config_path=CONFIG_DIR / "caco2_wang.yaml",
        output_dir=output_dir,
        training_config=ChemBERTaTrainingConfig(training_epochs=1),
    )

    predictions = pd.read_csv(output_dir / "predictions_test.csv")
    assert list(predictions.columns) == [
        "molecule_id",
        "canonical_smiles",
        "observed_target",
        "predicted_value",
        "residual",
    ]


def test_json_safe_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fake_training(monkeypatch, "binary_classification")
    split_paths = _write_splits(tmp_path, "binary_classification")
    output_dir = tmp_path / "json_safe"

    train_chemberta_model(
        **split_paths,
        config_path=CONFIG_DIR / "bbb_martins.yaml",
        output_dir=output_dir,
        training_config=ChemBERTaTrainingConfig(training_epochs=1),
    )

    for name in ["metrics.json", "training_metadata.json", "model_config.json", "training_history.json"]:
        json.dumps(_read_json(output_dir / name), allow_nan=False)


def test_model_and_tokenizer_output_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fake_training(monkeypatch, "binary_classification")
    split_paths = _write_splits(tmp_path, "binary_classification")
    output_dir = tmp_path / "paths"

    train_chemberta_model(
        **split_paths,
        config_path=CONFIG_DIR / "bbb_martins.yaml",
        output_dir=output_dir,
        training_config=ChemBERTaTrainingConfig(training_epochs=1),
    )

    assert (output_dir / "model" / "fake_model.bin").exists()
    assert (output_dir / "tokenizer" / "fake_tokenizer.json").exists()


def test_cli_argument_parsing() -> None:
    parser = chemberta_cli_parser()
    args = parser.parse_args(
        [
            "--train-csv",
            "train.csv",
            "--validation-csv",
            "valid.csv",
            "--test-csv",
            "test.csv",
            "--config",
            "config.yaml",
            "--output-dir",
            "out",
            "--epochs",
            "1",
            "--row-limit",
            "4",
            "--local-files-only",
        ]
    )

    assert args.epochs == 1
    assert args.row_limit == 4
    assert args.local_files_only is True


def test_cli_smoke_with_mocked_import_path(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train_chemberta.py"),
            "--help",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--model-name" in result.stdout


def chemberta_cli_parser():
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    try:
        import train_chemberta

        return train_chemberta.build_parser()
    finally:
        if str(PROJECT_ROOT / "scripts") in sys.path:
            sys.path.remove(str(PROJECT_ROOT / "scripts"))


def _install_fake_transformers(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[dict[str, object]] | None = None,
    fail_load: bool = False,
) -> None:
    calls = calls if calls is not None else []

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            if fail_load:
                raise OSError("missing model")
            calls.append(kwargs)
            return types.SimpleNamespace(_commit_hash="fake-commit")

    class FakeTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            if fail_load:
                raise OSError("missing tokenizer")
            return FakeSavedTokenizer()

    class FakeModelClass:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            if fail_load:
                raise OSError("missing model")
            return FakeSavedModel()

    fake_transformers = types.SimpleNamespace(
        AutoConfig=FakeAutoConfig,
        AutoModelForSequenceClassification=FakeModelClass,
        AutoTokenizer=FakeTokenizer,
        __version__="0.0.fake",
    )
    fake_torch = types.SimpleNamespace(__version__="0.0.fake")
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def _patch_fake_training(monkeypatch: pytest.MonkeyPatch, task_type: str) -> None:
    monkeypatch.setattr(
        chemberta,
        "_load_transformers_components",
        lambda _task_type, _config: {
            "tokenizer": FakeSavedTokenizer(),
            "model": FakeSavedModel(),
            "torch": types.SimpleNamespace(__version__="fake"),
        },
    )
    monkeypatch.setattr(chemberta, "_build_torch_dataset", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        chemberta,
        "_fit_model",
        lambda **kwargs: (
            [{"epoch": 1, "training_loss": 0.1}],
            None,
            "epoch_1",
            0.5,
            [],
        ),
    )

    def fake_predict(_model, _torch, _dataset, df, _task_type, _batch_size=64):
        predictions = df[["molecule_id", "canonical_smiles"]].copy()
        predictions["observed_target"] = df["target"].to_numpy()
        if task_type == "binary_classification":
            predictions["predicted_class"] = 1
            predictions["predicted_probability"] = 0.75
            return predictions, {"roc_auc": None, "pr_auc": None, "accuracy": 1.0}, [
                "ROC AUC unavailable"
            ]
        predictions["predicted_value"] = 1.0
        predictions["residual"] = df["target"].astype(float) - 1.0
        return predictions, {"rmse": 1.0, "mae": 1.0, "r2": None}, []

    monkeypatch.setattr(chemberta, "_predict_and_evaluate", fake_predict)


def _write_splits(tmp_path: Path, task_type: str) -> dict[str, Path]:
    rows = [
        {"molecule_id": "mol_001", "canonical_smiles": "CCO", "target": 1 if task_type == "binary_classification" else -4.8},
        {"molecule_id": "mol_002", "canonical_smiles": "CCN", "target": 0 if task_type == "binary_classification" else -4.5},
        {"molecule_id": "mol_003", "canonical_smiles": "CCOC", "target": 1 if task_type == "binary_classification" else -4.2},
    ]
    paths = {}
    for split in ["train", "validation", "test"]:
        path = tmp_path / ("valid.csv" if split == "validation" else f"{split}.csv")
        frame = pd.DataFrame(rows)
        frame["split"] = split
        frame.to_csv(path, index=False)
        paths["validation_csv" if split == "validation" else f"{split}_csv"] = path
    return paths


def _assert_required_artifacts(output_dir: Path) -> None:
    for name in [
        "model",
        "tokenizer",
        "metrics.json",
        "predictions_validation.csv",
        "predictions_test.csv",
        "training_metadata.json",
        "model_config.json",
        "training_history.json",
        "warnings.json",
    ]:
        assert (output_dir / name).exists()


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


class RecordingTokenizer:
    def __init__(self) -> None:
        self.inputs: list[str] = []

    def __call__(self, inputs, **kwargs):
        self.inputs = list(inputs)
        return {"input_ids": [[1]], "attention_mask": [[1]]}


class FakeTorch:
    long = "long"
    float32 = "float32"

    @staticmethod
    def tensor(values, dtype=None):
        return FakeTensor(values)

    class utils:
        class data:
            class TensorDataset:
                def __init__(self, *args):
                    self.args = args


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def reshape(self, *shape):
        return self


class FakeSavedModel:
    config = types.SimpleNamespace(_commit_hash="fake-commit")

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        Path(path, "fake_model.bin").write_text("fake", encoding="utf-8")


class FakeSavedTokenizer:
    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        Path(path, "fake_tokenizer.json").write_text("fake", encoding="utf-8")
