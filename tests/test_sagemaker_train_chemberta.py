import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from admet_platform.config import load_endpoint_config
from admet_platform.sagemaker import train_chemberta as sm_train


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def test_resolves_train_validation_and_test_channel_paths(tmp_path: Path) -> None:
    layout = _write_sm_layout(tmp_path)
    args = _args()
    env = {
        "SM_CHANNEL_TRAIN": str(layout["train_channel"]),
        "SM_CHANNEL_VALIDATION": str(layout["validation_channel"]),
        "SM_CHANNEL_TEST": str(layout["test_channel"]),
        "SM_MODEL_DIR": str(layout["model_dir"]),
        "SM_OUTPUT_DIR": str(layout["output_dir"]),
    }

    paths = sm_train.resolve_sagemaker_paths(args, env)

    assert paths["train_channel"] == layout["train_channel"]
    assert paths["validation_channel"] == layout["validation_channel"]
    assert paths["test_channel"] == layout["test_channel"]


def test_supports_valid_channel_alias(tmp_path: Path) -> None:
    layout = _write_sm_layout(tmp_path)
    args = _args()
    env = {
        "SM_CHANNEL_TRAIN": str(layout["train_channel"]),
        "SM_CHANNEL_VALID": str(layout["validation_channel"]),
        "SM_CHANNEL_TEST": str(layout["test_channel"]),
        "SM_MODEL_DIR": str(layout["model_dir"]),
    }

    paths = sm_train.resolve_sagemaker_paths(args, env)

    assert paths["validation_channel"] == layout["validation_channel"]


def test_missing_channel_errors(tmp_path: Path) -> None:
    layout = _write_sm_layout(tmp_path)
    args = _args()

    with pytest.raises(ValueError, match="SM_CHANNEL_TRAIN"):
        sm_train.resolve_sagemaker_paths(args, {"SM_MODEL_DIR": str(layout["model_dir"])})


def test_missing_csv_errors(tmp_path: Path) -> None:
    channel = tmp_path / "empty"
    channel.mkdir()

    with pytest.raises(ValueError, match="No CSV"):
        sm_train.resolve_channel_csv(channel)


def test_ambiguous_csv_errors(tmp_path: Path) -> None:
    channel = tmp_path / "channel"
    channel.mkdir()
    (channel / "a.csv").write_text("x\n", encoding="utf-8")
    (channel / "b.csv").write_text("x\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Multiple CSV"):
        sm_train.resolve_channel_csv(channel)


def test_model_output_and_checkpoint_directory_resolution(tmp_path: Path) -> None:
    layout = _write_sm_layout(tmp_path)
    args = _args(checkpoint_dir=str(layout["checkpoint_dir"]))
    env = {
        "SM_CHANNEL_TRAIN": str(layout["train_channel"]),
        "SM_CHANNEL_VALIDATION": str(layout["validation_channel"]),
        "SM_CHANNEL_TEST": str(layout["test_channel"]),
        "SM_MODEL_DIR": str(layout["model_dir"]),
        "SM_OUTPUT_DIR": str(layout["output_dir"]),
    }

    paths = sm_train.resolve_sagemaker_paths(args, env)

    assert paths["model_dir"] == layout["model_dir"]
    assert paths["output_dir"] == layout["output_dir"]
    assert paths["checkpoint_dir"] == layout["checkpoint_dir"]


def test_cli_hyperparameter_and_boolean_parsing() -> None:
    args = _args(
        max_sequence_length="64",
        learning_rate="0.001",
        epochs="2",
        train_batch_size="4",
        evaluation_batch_size="5",
        weight_decay="0.1",
        early_stopping_patience="1",
        random_seed="99",
        development_row_limit="7",
        local_files_only="true",
        cache_dir="null",
    )

    parsed = sm_train.parse_hyperparameters(args)

    assert parsed["max_sequence_length"] == 64
    assert parsed["learning_rate"] == 0.001
    assert parsed["training_epochs"] == 2
    assert parsed["local_files_only"] is True
    assert parsed["development_row_limit"] == 7
    assert parsed["cache_dir"] is None


def test_invalid_hyperparameter_rejection() -> None:
    with pytest.raises(ValueError, match="Invalid integer"):
        sm_train.parse_int("abc", "epochs")
    with pytest.raises(ValueError, match="Invalid boolean"):
        sm_train.parse_bool("maybe", "local_files_only")


def test_packaging_wrapper_imports_without_repo_root_assumptions() -> None:
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "sagemaker" / "train_chemberta.py"), "--help"],
        cwd=PROJECT_ROOT / "sagemaker",
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--endpoint-config" in result.stdout


def test_run_manifest_schema_and_secret_filtering(tmp_path: Path) -> None:
    layout = _write_sm_layout(tmp_path)
    config = load_endpoint_config(CONFIG_DIR / "bbb_martins.yaml")
    manifest = sm_train.build_run_manifest(
        run_id="run-1",
        config=config,
        paths={
            "train_channel": layout["train_channel"],
            "validation_channel": layout["validation_channel"],
            "test_channel": layout["test_channel"],
            "model_dir": layout["model_dir"],
            "output_dir": layout["output_dir"],
            "checkpoint_dir": layout["checkpoint_dir"],
        },
        hyperparameters={"model_name": "model", "development_row_limit": 3},
        result={"training_metadata": {"training_row_count": 1, "validation_row_count": 1, "test_row_count": 1}},
        start=datetime.now(UTC),
        runtime_seconds=1.2,
        status="completed",
        error=None,
    )

    assert {
        "run_id",
        "endpoint_id",
        "task_type",
        "input_channels",
        "model_dir",
        "output_data_dir",
        "checkpoint_dir",
        "hyperparameters",
        "status",
        "development_mode",
    } <= set(manifest)
    assert sm_train.sanitize_message("AWS_SECRET_ACCESS_KEY=abc HF_TOKEN=def") == "[REDACTED]=abc [REDACTED]=def"


def test_successful_simulated_sagemaker_execution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    layout = _write_sm_layout(tmp_path)
    monkeypatch.setattr(sm_train, "train_chemberta_model", _fake_train_success)
    env = {
        "SM_CHANNEL_TRAIN": str(layout["train_channel"]),
        "SM_CHANNEL_VALIDATION": str(layout["validation_channel"]),
        "SM_CHANNEL_TEST": str(layout["test_channel"]),
        "SM_MODEL_DIR": str(layout["model_dir"]),
        "SM_OUTPUT_DIR": str(layout["output_dir"]),
        "SM_CHECKPOINT_DIR": str(layout["checkpoint_dir"]),
    }
    monkeypatch.setattr(sm_train.os, "environ", env)

    exit_code = sm_train.main(
        [
            "--endpoint-config",
            str(CONFIG_DIR / "bbb_martins.yaml"),
            "--epochs",
            "1",
            "--development-row-limit",
            "2",
        ]
    )

    assert exit_code == 0
    assert (layout["model_dir"] / "model" / "fake_model.bin").exists()
    assert (layout["model_dir"] / "tokenizer" / "fake_tokenizer.json").exists()
    assert (layout["output_dir"] / "metrics.json").exists()
    assert (layout["output_dir"] / "run_manifest.json").exists()
    manifest = json.loads((layout["output_dir"] / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["development_mode"] is True


def test_failure_manifest_and_nonzero_exit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    layout = _write_sm_layout(tmp_path)

    def fail_train(**kwargs):
        raise RuntimeError("boom AWS_SECRET_ACCESS_KEY")

    monkeypatch.setattr(sm_train, "train_chemberta_model", fail_train)
    env = {
        "SM_CHANNEL_TRAIN": str(layout["train_channel"]),
        "SM_CHANNEL_VALIDATION": str(layout["validation_channel"]),
        "SM_CHANNEL_TEST": str(layout["test_channel"]),
        "SM_MODEL_DIR": str(layout["model_dir"]),
        "SM_OUTPUT_DIR": str(layout["output_dir"]),
    }
    monkeypatch.setattr(sm_train.os, "environ", env)

    exit_code = sm_train.main(["--endpoint-config", str(CONFIG_DIR / "bbb_martins.yaml")])

    assert exit_code == 1
    manifest = json.loads((layout["output_dir"] / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert "[REDACTED]" in manifest["error"]["message"]


def test_final_model_and_evaluation_outputs_are_separated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    layout = _write_sm_layout(tmp_path)
    monkeypatch.setattr(sm_train, "train_chemberta_model", _fake_train_success)
    env = {
        "SM_CHANNEL_TRAIN": str(layout["train_channel"]),
        "SM_CHANNEL_VALIDATION": str(layout["validation_channel"]),
        "SM_CHANNEL_TEST": str(layout["test_channel"]),
        "SM_MODEL_DIR": str(layout["model_dir"]),
        "SM_OUTPUT_DIR": str(layout["output_dir"]),
    }
    monkeypatch.setattr(sm_train.os, "environ", env)

    sm_train.main(["--endpoint-config", str(CONFIG_DIR / "bbb_martins.yaml")])

    assert (layout["model_dir"] / "model").exists()
    assert not (layout["model_dir"] / "metrics.json").exists()
    assert (layout["output_dir"] / "metrics.json").exists()
    assert not (layout["output_dir"] / "model" / "fake_model.bin").exists()


def test_required_column_validation(tmp_path: Path) -> None:
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("molecule_id,target\nmol,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required column"):
        sm_train.validate_required_columns(bad_csv)


def _args(**overrides) -> argparse.Namespace:
    values = {
        "endpoint_config": str(CONFIG_DIR / "bbb_martins.yaml"),
        "model_name": "model",
        "max_sequence_length": "128",
        "learning_rate": "2e-5",
        "epochs": "3",
        "train_batch_size": "8",
        "evaluation_batch_size": "16",
        "weight_decay": "0.01",
        "early_stopping_patience": "2",
        "random_seed": "42",
        "development_row_limit": None,
        "local_files_only": "false",
        "cache_dir": None,
        "train_channel": None,
        "validation_channel": None,
        "test_channel": None,
        "model_dir": None,
        "output_dir": None,
        "checkpoint_dir": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _write_sm_layout(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "opt" / "ml"
    layout = {
        "train_channel": root / "input" / "data" / "train",
        "validation_channel": root / "input" / "data" / "validation",
        "test_channel": root / "input" / "data" / "test",
        "model_dir": root / "model",
        "output_dir": root / "output" / "data",
        "checkpoint_dir": root / "checkpoints",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    for split, key in [("train", "train_channel"), ("validation", "validation_channel"), ("test", "test_channel")]:
        pd.DataFrame(
            [
                {
                    "molecule_id": f"{split}_001",
                    "canonical_smiles": "CCO",
                    "target": 1,
                    "split": split,
                }
            ]
        ).to_csv(layout[key] / f"{split}.csv", index=False)
    return layout


def _fake_train_success(**kwargs):
    model_dir = Path(kwargs["model_dir"])
    output_dir = Path(kwargs["output_dir"])
    (model_dir / "model").mkdir(parents=True, exist_ok=True)
    (model_dir / "tokenizer").mkdir(parents=True, exist_ok=True)
    (model_dir / "model" / "fake_model.bin").write_text("fake", encoding="utf-8")
    (model_dir / "tokenizer" / "fake_tokenizer.json").write_text("fake", encoding="utf-8")
    (output_dir / "model_config.json").write_text("{}", encoding="utf-8")
    for name in [
        "metrics.json",
        "predictions_validation.csv",
        "predictions_test.csv",
        "training_metadata.json",
        "training_history.json",
        "warnings.json",
    ]:
        (output_dir / name).write_text("{}" if name.endswith(".json") else "molecule_id\n", encoding="utf-8")
    return {
        "training_metadata": {
            "training_row_count": 1,
            "validation_row_count": 1,
            "test_row_count": 1,
            "warnings": [],
        }
    }
