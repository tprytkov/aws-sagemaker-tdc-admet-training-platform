from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

import pytest

from admet_platform.chemprop.config import load_chemprop_config
from admet_platform.chemprop.production_manifest import (
    ENDPOINT_ORDER,
    PRODUCTION_SEEDS,
    UNCERTAINTY_INTERPRETATION,
    create_regression_production_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "chemprop" / "multitask_admet_regression.yaml"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_checksums(root: Path, ensemble: Path) -> Path:
    checksums = ensemble / "SHA256SUMS"
    files = sorted(path for path in ensemble.iterdir() if path != checksums)
    lines = [f"{_sha(path)}  {path.relative_to(root).as_posix()}\n" for path in files]
    # The real ECHO inventory also contains historical entries outside this directory.
    lines.append(f"{'f' * 64}  outputs/gpu/pilot/multitask_regression_seed13/console.log\n")
    checksums.write_text("".join(lines), encoding="utf-8")
    return checksums


def _fixture(tmp_path: Path) -> dict[str, object]:
    config = load_chemprop_config(CONFIG)
    config_root = tmp_path / "configs" / "chemprop"
    config_root.mkdir(parents=True)
    for name in ("base.yaml", "multitask_admet_regression.yaml"):
        (config_root / name).write_bytes((CONFIG.parent / name).read_bytes())
    hashes: dict[int, str] = {}
    runs: dict[int, Path] = {}
    scaler = {
        "fit_split": "train",
        "per_endpoint": {
            endpoint: {
                "scientific_transform": config.tasks[endpoint]["target_transform"],
                "mean": float(index),
                "scale": float(index + 1),
                "train_label_count": 100 + index,
            }
            for index, endpoint in enumerate(ENDPOINT_ORDER)
        },
        "validation_statistics_used": False,
        "test_statistics_used": False,
    }
    expected_split_hashes = {
        f"{endpoint}/{split}": hashlib.sha256(f"{endpoint}/{split}".encode()).hexdigest()
        for endpoint in ENDPOINT_ORDER
        for split in ("train", "validation", "test")
    }
    verified_split_hashes = {
        key: value for key, value in expected_split_hashes.items() if not key.endswith("/test")
    }
    for seed in PRODUCTION_SEEDS:
        run = tmp_path / "outputs" / "gpu" / "pilot" / f"multitask_regression_seed{seed}"
        runs[seed] = run.relative_to(tmp_path)
        checkpoint = run / "checkpoints" / "best.ckpt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(f"checkpoint-{seed}".encode())
        hashes[seed] = _sha(checkpoint)
        resolved = {
            **config.raw,
            "resolved_seed": seed,
            "smoke": False,
            "resolved_model": config.model,
            "resolved_training": config.training,
        }
        summary = {
            "endpoint": "multitask_admet_regression",
            "task_type": "regression",
            "seed": seed,
            "checkpoint": f"outputs/gpu/pilot/multitask_regression_seed{seed}/checkpoints/best.ckpt",
            "target_scaler": scaler,
        }
        verification = {
            "loaded_splits": ["train", "validation"],
            "locked_test_opened": False,
            "split_manifest_sha256": config.raw["split_manifest_sha256"],
            "expected_split_hashes": expected_split_hashes,
            "verified_split_hashes": verified_split_hashes,
        }
        _write_json(run / "resolved_config.json", resolved)
        _write_json(run / "run_summary.json", summary)
        _write_json(run / "target_scaler.json", scaler)
        _write_json(run / "data_verification.json", verification)

    ensemble = tmp_path / "outputs" / "gpu" / "pilot" / "multitask_regression_ensemble_validation"
    ensemble.mkdir(parents=True)
    metrics = {
        endpoint: {
            "mae": 1.0 + index,
            "rmse": 2.0 + index,
            "r2": 0.1 + index / 10,
            "spearman": 0.2 + index / 10,
            "median_absolute_error": 0.5 + index,
        }
        for index, endpoint in enumerate(ENDPOINT_ORDER)
    }
    ensemble_summary = {
        "schema_version": "1.0.0",
        "split": "validation",
        "locked_test_opened": False,
        "seeds": list(PRODUCTION_SEEDS),
        "endpoint_order": list(ENDPOINT_ORDER),
        "validation_metrics": metrics,
    }
    summary_path = ensemble / "ensemble_validation_summary.json"
    _write_json(summary_path, ensemble_summary)
    for endpoint in ENDPOINT_ORDER:
        (ensemble / f"{endpoint}.csv").write_text(
            "molecule_id,canonical_smiles,target,ensemble_mean,ensemble_std,seed_count,"
            "uncertainty_interpretation\n"
            f"m1,CC,1.0,1.1,0.2,5,{UNCERTAINTY_INTERPRETATION}\n",
            encoding="utf-8",
        )
    checksums = _write_checksums(tmp_path, ensemble)
    requirements = tmp_path / "requirements-chemprop.txt"
    requirements.write_text(
        "chemprop==2.3.1\nlightning==2.6.5\nrdkit==2026.3.5\n"
        "numpy==2.4.6\npandas==3.0.5\nscikit-learn==1.9.0\n",
        encoding="utf-8",
    )
    environment = tmp_path / "environment-chemprop-gpu.yml"
    environment.write_text(
        "dependencies:\n  - python=3.11.15\n  - pip:\n      - torch==2.6.0+cu124\n",
        encoding="utf-8",
    )
    return {
        "artifact_root": tmp_path,
        "config_path": (config_root / "multitask_admet_regression.yaml").relative_to(tmp_path),
        "run_directories": runs,
        "ensemble_validation_directory": ensemble.relative_to(tmp_path),
        "ensemble_summary_path": summary_path.relative_to(tmp_path),
        "ensemble_checksums_path": checksums.relative_to(tmp_path),
        "requirements_path": requirements.relative_to(tmp_path),
        "environment_path": environment.relative_to(tmp_path),
        "output_path": tmp_path / "release" / "production_manifest.json",
        "git_commit": "a" * 40,
        "expected_checkpoint_hashes": hashes,
        "observed_runtime": {
            "python": "3.11.15",
            "chemprop": "2.3.1",
            "torch": "2.6.0+cu124",
            "cuda_available": True,
            "torch_cuda_version": "12.4",
            "platform": "synthetic-test-platform",
            "python_implementation": "CPython",
            "executable": "python",
        },
    }


def _create(inputs: dict[str, object]) -> dict[str, object]:
    return create_regression_production_manifest(**inputs)  # type: ignore[arg-type]


def test_manifest_freezes_exact_ensemble_endpoint_and_transform_contracts(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    manifest = _create(inputs)
    assert manifest["production_ensemble"] == {
        "seeds": [13, 37, 73, 101, 137],
        "checkpoint_selection": "best.ckpt_per_seed",
        "rule": "unweighted_arithmetic_mean_of_exactly_all_five_seed_predictions",
        "missing_seed_policy": "fail_closed",
        "seed_weighting": "none",
        "calibration": "none",
        "disagreement": {
            "statistic": "sample_standard_deviation_across_seed_predictions",
            "ddof": 1,
            "seed_count": 5,
            "interpretation": UNCERTAINTY_INTERPRETATION,
        },
    }
    assert manifest["endpoint_order"] == list(ENDPOINT_ORDER)
    endpoints = manifest["endpoints"]
    assert endpoints["caco2_wang"]["validation_metrics"]["space"] == (
        "stored_log10_papp_space_not_physical_cm_per_s"
    )
    assert endpoints["caco2_wang"]["internal_model_output_unit"] == "log10(Papp [cm/s])"
    assert endpoints["caco2_wang"]["user_facing_unit"] == "cm/s"
    assert endpoints["vdss_lombardo"]["scientific_transform"] == "log10"
    assert endpoints["vdss_lombardo"]["inverse_transform_for_user_output"] == "10**y"
    assert endpoints["vdss_lombardo"]["user_facing_unit"] == "L/kg"
    assert all(item["clipping"] == "none" for item in endpoints.values())
    assert manifest["provenance"] == {
        "manifest_creation_git_commit": "a" * 40,
        "training_git_commit": {"status": "not_recorded", "value": None},
    }
    assert manifest["environment"]["expected_pinned"]["package_versions"]["chemprop"] == ("2.3.1")
    assert manifest["environment"]["observed_at_manifest_creation"]["platform"] == (
        "synthetic-test-platform"
    )
    assert (
        not {
            "activation",
            "optimizer",
            "scheduler",
        }
        & manifest["model"]["architecture"].keys()
    )


def test_manifest_reads_metrics_and_records_validation_artifact_hashes(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    summary_path = tmp_path / inputs["ensemble_summary_path"]  # type: ignore[operator]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["validation_metrics"]["caco2_wang"]["mae"] = 9.125
    _write_json(summary_path, summary)
    ensemble = summary_path.parent
    _write_checksums(tmp_path, ensemble)
    manifest = _create(inputs)
    assert manifest["validation_evidence"]["metrics"]["caco2_wang"]["mae"] == 9.125
    assert manifest["validation_evidence"]["summary_sha256"] == _sha(summary_path)
    assert len(manifest["validation_evidence"]["artifact_inventory"]) == 6


def test_missing_seed_and_bad_checkpoint_hash_fail_closed(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    missing = copy.deepcopy(inputs)
    del missing["run_directories"][137]  # type: ignore[index]
    with pytest.raises(ValueError, match="exactly seeds"):
        _create(missing)
    checkpoint = tmp_path / inputs["run_directories"][37] / "checkpoints" / "best.ckpt"  # type: ignore[index,operator]
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        _create(inputs)


def test_split_scaler_and_locked_test_inconsistency_fail_closed(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    run37 = tmp_path / inputs["run_directories"][37]  # type: ignore[index,operator]
    scaler_path = run37 / "target_scaler.json"
    scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    scaler["per_endpoint"]["ppbr_az"]["mean"] += 1
    _write_json(scaler_path, scaler)
    summary_path = run37 / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["target_scaler"] = scaler
    _write_json(summary_path, summary)
    with pytest.raises(ValueError, match="target scaler differs"):
        _create(inputs)

    inputs = _fixture(tmp_path / "locked")
    run13 = (tmp_path / "locked") / inputs["run_directories"][13]  # type: ignore[index,operator]
    verification_path = run13 / "data_verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    verification["locked_test_opened"] = True
    _write_json(verification_path, verification)
    with pytest.raises(ValueError, match="locked-test isolation"):
        _create(inputs)


def test_endpoint_order_and_ensemble_summary_contract_fail_closed(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    summary_path = tmp_path / inputs["ensemble_summary_path"]  # type: ignore[operator]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["endpoint_order"] = list(reversed(summary["endpoint_order"]))
    _write_json(summary_path, summary)
    ensemble = summary_path.parent
    _write_checksums(tmp_path, ensemble)
    with pytest.raises(ValueError, match="endpoint order"):
        _create(inputs)


def test_validation_summary_hash_tampering_fails_closed(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    summary_path = tmp_path / inputs["ensemble_summary_path"]  # type: ignore[operator]
    summary_path.write_text(summary_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _create(inputs)


def test_checksums_are_root_relative_and_unrelated_entries_are_not_required(
    tmp_path: Path,
) -> None:
    inputs = _fixture(tmp_path)
    manifest = _create(inputs)
    inventory = manifest["validation_evidence"]["artifact_inventory"]
    assert [Path(item["path"]).name for item in inventory] == sorted(
        [
            "caco2_wang.csv",
            "ensemble_validation_summary.json",
            "lipophilicity_astrazeneca.csv",
            "ppbr_az.csv",
            "solubility_aqsoldb.csv",
            "vdss_lombardo.csv",
        ]
    )

    inputs = _fixture(tmp_path / "bad-relative")
    checksums = (tmp_path / "bad-relative") / inputs["ensemble_checksums_path"]  # type: ignore[operator]
    text = checksums.read_text(encoding="utf-8")
    text = re.sub(
        r"outputs/gpu/pilot/multitask_regression_ensemble_validation/caco2_wang.csv",
        "caco2_wang.csv",
        text,
    )
    checksums.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="absent from SHA256SUMS"):
        _create(inputs)


def test_observed_runtime_must_match_pinned_scientific_versions(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    inputs["observed_runtime"]["chemprop"] = "2.4.0"  # type: ignore[index]
    with pytest.raises(ValueError, match="Observed Chemprop version differs"):
        _create(inputs)


def test_refuses_overwrite_and_manifest_bytes_are_deterministic(tmp_path: Path) -> None:
    first = _fixture(tmp_path / "first")
    _create(first)
    first_bytes = Path(first["output_path"]).read_bytes()  # type: ignore[arg-type]
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _create(first)

    second = _fixture(tmp_path / "second")
    _create(second)
    second_bytes = Path(second["output_path"]).read_bytes()  # type: ignore[arg-type]
    # Artifact-root-relative paths and fixed inputs make independently created manifests identical.
    assert first_bytes == second_bytes
    checksum = Path(second["output_path"]).with_name("production_manifest.json.sha256")  # type: ignore[arg-type]
    assert checksum.read_text(encoding="utf-8").startswith(hashlib.sha256(second_bytes).hexdigest())

    third = _fixture(tmp_path / "sidecar-only")
    third_checksum = Path(third["output_path"]).with_name("production_manifest.json.sha256")  # type: ignore[arg-type]
    third_checksum.parent.mkdir(parents=True)
    third_checksum.write_text("existing\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="manifest checksum"):
        _create(third)
    assert not Path(third["output_path"]).exists()  # type: ignore[arg-type]
