import json
from pathlib import Path

import pandas as pd
import pytest
import torch
import numpy as np
from transformers import AutoTokenizer

from admet_platform.data.multitask import (
    build_task_dataloaders, class_preserving_subset, load_endpoint_datasets,
    load_multitask_config,
)
from admet_platform.training.multitask_run import run_multitask_training


TASKS = ("bbb_martins", "herg_karim", "ames")


def _setup(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "prepared"
    smiles = ["CCO", "CCN", "CCC", "COC", "CNC", "CC=O"]
    for task in TASKS:
        endpoint = root / task
        endpoint.mkdir(parents=True)
        for split, name in (("train", "train.csv"), ("validation", "valid.csv"), ("test", "test.csv")):
            pd.DataFrame({
                "molecule_id": [f"{task}-{split}-{i}" for i in range(6)],
                "smiles": smiles, "canonical_smiles": smiles,
                "target": [0, 1, 0, 1, 0, 1], "split": [split] * 6,
            }).to_csv(endpoint / name, index=False)
    config = tmp_path / "multitask.yaml"
    task_yaml = "\n".join(
        f"  {task}:\n    endpoint_id: {task}\n    tdc_name: {'BBB_Martins' if task == 'bbb_martins' else 'herg' if task == 'herg_karim' else 'AMES'}\n    task_group: {'ADME' if task == 'bbb_martins' else 'Tox'}\n    task_type: binary_classification\n    primary_metric: roc_auc"
        for task in TASKS
    )
    config.write_text(f'''schema_version: "1.0.0"
run_name: smoke
split_track: coordinated_multitask
prepared_root: prepared
tasks:
{task_yaml}
split_files:
  train: train.csv
  validation: valid.csv
  test: test.csv
audit:
  enforce_exact_smiles_exclusion: true
  enforce_scaffold_exclusion: true
training:
  random_seed: 17
  train_batch_size: 2
  evaluation_batch_size: 3
  max_sequence_length: 16
  dropout: 0.0
''', encoding="utf-8")
    return config, root


def test_schema_invalid_labels_and_empty_smiles_fail(tmp_path: Path) -> None:
    config_path, root = _setup(tmp_path)
    path = root / "ames" / "train.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "target"] = 2
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="non-binary"):
        load_endpoint_datasets(load_multitask_config(config_path))
    frame.loc[0, "target"] = 0
    frame.loc[0, "canonical_smiles"] = ""
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="invalid or empty SMILES"):
        load_endpoint_datasets(load_multitask_config(config_path))


def test_class_preserving_subset_and_separate_deterministic_loaders(
    tmp_path: Path, tiny_model_tokenizer_dir: Path
) -> None:
    config_path, _ = _setup(tmp_path)
    datasets = load_endpoint_datasets(load_multitask_config(config_path))
    tokenizer = AutoTokenizer.from_pretrained(tiny_model_tokenizer_dir, local_files_only=True)
    first = build_task_dataloaders(datasets, tokenizer, seed=9, train_batch_size=2,
                                   evaluation_batch_size=2, max_length=16, limit_samples_per_task=4)
    second = build_task_dataloaders(datasets, tokenizer, seed=9, train_batch_size=2,
                                    evaluation_batch_size=2, max_length=16, limit_samples_per_task=4)
    assert set(class_preserving_subset(datasets["ames"].train, 2, 9)["target"]) == {0, 1}
    assert list(first) == list(TASKS)
    for task in TASKS:
        batch1, batch2 = next(iter(first[task]["train"])), next(iter(second[task]["train"]))
        assert set(batch1["task_name"]) == {task}
        assert batch1["molecule_id"] == batch2["molecule_id"]
        assert batch1["input_ids"].equal(batch2["input_ids"])
        assert len(first[task]["train"].dataset) == 4
        assert len(first[task]["validation"].dataset) == 6
        assert len(first[task]["test"].dataset) == 6


def test_offline_cpu_smoke_artifacts_and_resume(
    tmp_path: Path, tiny_model_tokenizer_dir: Path
) -> None:
    config_path, root = _setup(tmp_path)
    output = tmp_path / "run"
    first = run_multitask_training(
        config_path=config_path, prepared_root=root, output_dir=output,
        checkpoint=str(tiny_model_tokenizer_dir), max_steps=3,
        limit_samples_per_task=4, seed=17, device="cpu", offline=True,
    )
    assert first["task_contributions"]["batch_counts"] == {task: 1 for task in TASKS}
    required = {"resolved_config.json", "dataset_manifest.json", "training_metrics.json",
                "task_contributions.json", "checkpoint.pt", "run_manifest.json",
                "training_history.jsonl", "validation_history.jsonl",
                "checkpoint_selection.json", "early_stopping.json",
                "final_run_summary.json", "endpoint_comparison.csv",
                "endpoint_comparison.json", "latest", "best_composite",
                "best_bbb_martins", "best_herg_karim", "best_ames"}
    assert required <= {path.name for path in output.iterdir()}
    assert all((output / f"validation_predictions_{task}.csv").is_file() for task in TASKS)
    for task in TASKS:
        predictions = pd.read_csv(output / f"validation_predictions_{task}.csv")
        assert predictions["molecule_id"].str.contains("-validation-").all()
        assert not predictions["molecule_id"].str.contains("-test-").any()
    assert len((output / "training_history.jsonl").read_text().splitlines()) == 3
    assert len((output / "validation_history.jsonl").read_text().splitlines()) == 3
    manifest = json.loads((output / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["input_hashes"]) == 9
    resumed = run_multitask_training(
        config_path=config_path, prepared_root=root, output_dir=tmp_path / "resumed",
        checkpoint=str(tiny_model_tokenizer_dir), resume_from=output / "checkpoint.pt",
        max_steps=1, limit_samples_per_task=4, seed=17, device="cpu", offline=True,
    )
    assert resumed["global_step"] == 4
    assert resumed["task_contributions"]["batch_counts"] == {
        "bbb_martins": 2, "herg_karim": 1, "ames": 1,
    }


def _run_exact(
    config_path: Path, root: Path, model_path: Path, output: Path,
    steps: int, *, seed: int = 42, resume_from: Path | None = None,
) -> dict:
    return run_multitask_training(
        config_path=config_path, prepared_root=root, output_dir=output,
        checkpoint=str(model_path), resume_from=resume_from, max_steps=steps,
        limit_samples_per_task=4, seed=seed, device="cpu", offline=True,
        deterministic_algorithms=True,
    )


def _assert_nested_exact(left, right) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_exact(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right):
            _assert_nested_exact(left_value, right_value)
    else:
        assert left == right


def _checkpoint(path: Path) -> dict:
    return torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=False)


def test_two_fresh_one_step_runs_are_exact(
    tmp_path: Path, tiny_model_tokenizer_dir: Path
) -> None:
    config_path, root = _setup(tmp_path)
    first_dir, second_dir = tmp_path / "one-a", tmp_path / "one-b"
    first = _run_exact(config_path, root, tiny_model_tokenizer_dir, first_dir, 1)
    second = _run_exact(config_path, root, tiny_model_tokenizer_dir, second_dir, 1)
    first_checkpoint, second_checkpoint = _checkpoint(first_dir), _checkpoint(second_dir)

    assert first["initial_model_state_hash"] == second["initial_model_state_hash"]
    assert first["initial_task_head_hashes"] == second["initial_task_head_hashes"]
    first_record, second_record = first_checkpoint["history"][0], second_checkpoint["history"][0]
    assert first_record["molecule_ids"] == second_record["molecule_ids"]
    assert first_record["batch_hash"] == second_record["batch_hash"]
    assert first_record["combined_loss"] == second_record["combined_loss"]
    assert first_record["gradient_norm_before_clipping"] == second_record["gradient_norm_before_clipping"]
    _assert_nested_exact(first_checkpoint["model_state"], second_checkpoint["model_state"])
    _assert_nested_exact(first_checkpoint["optimizer_state"], second_checkpoint["optimizer_state"])


def test_two_fresh_seven_step_runs_are_exact(
    tmp_path: Path, tiny_model_tokenizer_dir: Path
) -> None:
    config_path, root = _setup(tmp_path)
    first_dir, second_dir = tmp_path / "seven-a", tmp_path / "seven-b"
    _run_exact(config_path, root, tiny_model_tokenizer_dir, first_dir, 7)
    _run_exact(config_path, root, tiny_model_tokenizer_dir, second_dir, 7)
    _assert_nested_exact(_checkpoint(first_dir), _checkpoint(second_dir))
    for task in TASKS:
        assert (first_dir / f"validation_predictions_{task}.csv").read_bytes() == (
            second_dir / f"validation_predictions_{task}.csv"
        ).read_bytes()


def test_uninterrupted_seven_matches_six_plus_one_resume_exactly(
    tmp_path: Path, tiny_model_tokenizer_dir: Path
) -> None:
    config_path, root = _setup(tmp_path)
    full_dir, six_dir, resumed_dir = tmp_path / "full", tmp_path / "six", tmp_path / "resumed-exact"
    _run_exact(config_path, root, tiny_model_tokenizer_dir, full_dir, 7)
    _run_exact(config_path, root, tiny_model_tokenizer_dir, six_dir, 6)
    _run_exact(
        config_path, root, tiny_model_tokenizer_dir, resumed_dir, 1,
        resume_from=six_dir / "checkpoint.pt",
    )
    full_checkpoint, resumed_checkpoint = _checkpoint(full_dir), _checkpoint(resumed_dir)
    for key in (
        "model_state", "optimizer_state", "sampler_state", "loader_states",
        "scheduler_state", "scaler_state", "control_state",
        "history", "global_step", "initial_model_state_hash", "initial_task_head_hashes",
    ):
        _assert_nested_exact(full_checkpoint[key], resumed_checkpoint[key])
    assert json.loads((full_dir / "task_contributions.json").read_text()) == json.loads(
        (resumed_dir / "task_contributions.json").read_text()
    )
    full_metrics = json.loads((full_dir / "training_metrics.json").read_text())
    resumed_metrics = json.loads((resumed_dir / "training_metrics.json").read_text())
    assert full_metrics["history"] == resumed_metrics["history"]
    assert full_metrics["validation"] == resumed_metrics["validation"]
    for task in TASKS:
        assert (full_dir / f"validation_predictions_{task}.csv").read_bytes() == (
            resumed_dir / f"validation_predictions_{task}.csv"
        ).read_bytes()


def test_different_seeds_may_differ(tmp_path: Path, tiny_model_tokenizer_dir: Path) -> None:
    config_path, root = _setup(tmp_path)
    first_dir, second_dir = tmp_path / "seed-42", tmp_path / "seed-43"
    first = _run_exact(config_path, root, tiny_model_tokenizer_dir, first_dir, 1, seed=42)
    second = _run_exact(config_path, root, tiny_model_tokenizer_dir, second_dir, 1, seed=43)
    assert first["initial_model_state_hash"] != second["initial_model_state_hash"]
