"""Fail-closed production manifest creation for the five-head Chemprop regressor."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from admet_platform.chemprop.config import ChempropExperimentConfig, load_chemprop_config


SCHEMA_VERSION = "1.0.0"
RELEASE_STATUS = "production_manifest_frozen_pending_inference_qualification"
PRODUCTION_SEEDS = (13, 37, 73, 101, 137)
ENDPOINT_ORDER = (
    "caco2_wang",
    "lipophilicity_astrazeneca",
    "solubility_aqsoldb",
    "ppbr_az",
    "vdss_lombardo",
)
CHECKPOINT_SHA256 = {
    13: "9f4b6a234db9cc7d553980d250b6bb86535423ffd1d9f70d18d1b92b6f583da5",
    37: "07a6cbc3a1ffa86667b02e72cceb4d38c2356ff333178075956d07da116445ca",
    73: "805aee18c295c950c08dac2aed137c1063dbc38824a998cbeae5ff47d95f65ce",
    101: "63001c72219d6def35dc09f65337d85cf71e44baa0841b06ba307e0cffd3d313",
    137: "f43e3276721accd55351fc81cce29bea750026bf129d05747c4b47d50f2bcce8",
}
REQUIRED_METRICS = ("mae", "rmse", "r2", "spearman", "median_absolute_error")
UNCERTAINTY_INTERPRETATION = "seed_variability_not_calibrated_confidence_interval"
REQUIRED_ENSEMBLE_ARTIFACTS = (
    "caco2_wang.csv",
    "ensemble_validation_summary.json",
    "lipophilicity_astrazeneca.csv",
    "ppbr_az.csv",
    "solubility_aqsoldb.csv",
    "vdss_lombardo.csv",
)


def create_regression_production_manifest(
    *,
    artifact_root: str | Path,
    config_path: str | Path,
    run_directories: Mapping[int, str | Path],
    ensemble_validation_directory: str | Path,
    ensemble_summary_path: str | Path,
    ensemble_checksums_path: str | Path,
    requirements_path: str | Path,
    environment_path: str | Path,
    output_path: str | Path,
    git_commit: str,
    expected_checkpoint_hashes: Mapping[int, str] | None = None,
    observed_runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate immutable artifacts and create a new deterministic manifest.

    ``expected_checkpoint_hashes`` and ``observed_runtime`` support synthetic tests. Production
    callers must omit both so approved hashes and the actual creation runtime are used.
    No model, prepared dataset, or locked-test CSV is opened by this function.
    """

    root = Path(artifact_root).resolve()
    destination = _within_root(root, output_path, "production manifest output")
    digest_path = destination.with_name(destination.name + ".sha256")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing production manifest: {destination}")
    if digest_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest checksum: {digest_path}")
    hashes = dict(expected_checkpoint_hashes or CHECKPOINT_SHA256)
    _validate_exact_seed_mapping(run_directories, hashes)
    config_file = _within_root(root, config_path, "Chemprop configuration")
    config = load_chemprop_config(config_file)
    _validate_config(config)
    commit = _validate_git_commit(git_commit)

    ensemble_dir = _within_root(
        root, ensemble_validation_directory, "ensemble validation directory"
    )
    summary_path = _within_root(root, ensemble_summary_path, "ensemble summary")
    checksums_path = _within_root(root, ensemble_checksums_path, "ensemble checksums")
    required_summary = (ensemble_dir / "ensemble_validation_summary.json").resolve()
    if summary_path != required_summary:
        raise ValueError(
            "Ensemble summary must be ensemble_validation_summary.json in the "
            "ensemble-validation directory."
        )
    validation_inventory = _verify_checksum_inventory(root, ensemble_dir, checksums_path)
    summary = _read_json(summary_path)
    metrics = _validate_ensemble_summary(summary)
    _validate_ensemble_csvs(ensemble_dir)

    requirements = _within_root(root, requirements_path, "requirements file")
    environment = _within_root(root, environment_path, "environment file")
    package_versions = _validate_environment_contract(requirements, environment, config)
    runtime_versions = dict(observed_runtime or _collect_observed_runtime())
    _validate_observed_runtime(runtime_versions, package_versions)

    checkpoints: list[dict[str, Any]] = []
    reference_scientific: dict[str, Any] | None = None
    reference_scaler: dict[str, Any] | None = None
    reference_scaler_sha: str | None = None
    split_identity: dict[str, Any] | None = None
    for seed in PRODUCTION_SEEDS:
        run_dir = _within_root(root, run_directories[seed], f"seed {seed} run directory")
        checked = _validate_seed_run(root, run_dir, seed, hashes[seed], config)
        if reference_scientific is None:
            reference_scientific = checked["scientific_config"]
            reference_scaler = checked["scaler"]
            reference_scaler_sha = checked["scaler_sha256"]
            split_identity = checked["split_identity"]
        else:
            if checked["scientific_config"] != reference_scientific:
                raise ValueError(
                    f"Seed {seed} resolved configuration is scientifically incompatible."
                )
            if checked["scaler"] != reference_scaler:
                raise ValueError(f"Seed {seed} target scaler differs from the frozen scaler.")
            if checked["scaler_sha256"] != reference_scaler_sha:
                raise ValueError(f"Seed {seed} target scaler SHA-256 differs.")
            if checked["split_identity"] != split_identity:
                raise ValueError(f"Seed {seed} split identity differs.")
        checkpoints.append(checked["checkpoint"])

    assert reference_scientific is not None
    assert reference_scaler is not None
    assert reference_scaler_sha is not None
    assert split_identity is not None
    endpoints = _endpoint_contracts(config, metrics, reference_scaler)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "release_status": RELEASE_STATUS,
        "model": {
            "provider": config.raw["provider"],
            "family": config.raw["model_family"],
            "chemprop_version": config.raw["chemprop_version"],
            "architecture": reference_scientific["model"],
            "task_type": "regression",
            "checkpoint_contents": "five_regression_outputs_no_classification_outputs",
            "configuration_path": _portable_path(root, config_file),
            "configuration_sha256": _sha256(config_file),
        },
        "production_ensemble": {
            "seeds": list(PRODUCTION_SEEDS),
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
        },
        "checkpoints": checkpoints,
        "endpoint_order": list(ENDPOINT_ORDER),
        "endpoints": endpoints,
        "data": {
            "dataset": config.dataset,
            "dataset_version": config.dataset_version,
            "protocol": config.raw["protocol"],
            "preprocessing": config.raw["preprocessing"],
            "split_identity": split_identity,
            "locked_test": {"status": "not_accessed", "used_for_manifest": False},
        },
        "target_scaler": {
            "path": checkpoints[0]["target_scaler_path"],
            "sha256": reference_scaler_sha,
            "fit_split": "train",
            "validation_statistics_used": False,
            "test_statistics_used": False,
            "per_endpoint": reference_scaler["per_endpoint"],
        },
        "validation_evidence": {
            "split": "validation",
            "prediction_recomputation_performed": False,
            "summary_path": _portable_path(root, summary_path),
            "summary_sha256": _sha256(summary_path),
            "checksum_inventory_path": _portable_path(root, checksums_path),
            "checksum_inventory_sha256": _sha256(checksums_path),
            "artifact_inventory": validation_inventory,
            "metrics": metrics,
        },
        "applicability_domain": {
            **config.raw["applicability_domain"],
            "maximum_similarity": "maximum_morgan_tanimoto_to_endpoint_training_set",
            "scaffold_familiarity": "murcko_scaffold_present_in_endpoint_training_set",
            "label_rule": "in_domain_only_if_similarity_at_or_above_threshold_and_scaffold_familiar",
        },
        "environment": {
            "expected_pinned": {
                "package_versions": package_versions,
                "requirements_path": _portable_path(root, requirements),
                "requirements_sha256": _sha256(requirements),
                "environment_path": _portable_path(root, environment),
                "environment_sha256": _sha256(environment),
            },
            "observed_at_manifest_creation": runtime_versions,
        },
        "provenance": {
            "manifest_creation_git_commit": commit,
            "training_git_commit": {"status": "not_recorded", "value": None},
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_json(manifest)
    try:
        with destination.open("xb") as handle:
            handle.write(encoded)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Refusing to overwrite existing production manifest: {destination}"
        ) from exc
    try:
        with digest_path.open("x", encoding="utf-8") as handle:
            handle.write(f"{hashlib.sha256(encoded).hexdigest()}  {destination.name}\n")
    except FileExistsError as exc:
        destination.unlink()
        raise FileExistsError(
            f"Refusing to overwrite existing manifest checksum: {digest_path}"
        ) from exc
    return manifest


def current_git_commit(repository_root: str | Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(repository_root),
        check=True,
        capture_output=True,
        text=True,
    )
    return _validate_git_commit(result.stdout.strip())


def _validate_seed_run(
    root: Path,
    run_dir: Path,
    seed: int,
    expected_checkpoint_hash: str,
    config: ChempropExperimentConfig,
) -> dict[str, Any]:
    required = {
        "checkpoint": run_dir / "checkpoints" / "best.ckpt",
        "resolved": run_dir / "resolved_config.json",
        "summary": run_dir / "run_summary.json",
        "scaler": run_dir / "target_scaler.json",
        "verification": run_dir / "data_verification.json",
    }
    for label, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"Seed {seed} is missing required {label}: {path}")
    checkpoint_sha = _sha256(required["checkpoint"])
    if checkpoint_sha != expected_checkpoint_hash:
        raise ValueError(f"Seed {seed} checkpoint SHA-256 mismatch.")
    resolved = _read_json(required["resolved"])
    summary = _read_json(required["summary"])
    scaler = _read_json(required["scaler"])
    verification = _read_json(required["verification"])
    if resolved.get("resolved_seed") != seed or summary.get("seed") != seed:
        raise ValueError(f"Seed {seed} run metadata has the wrong seed identity.")
    if Path(str(summary.get("checkpoint", ""))).name != "best.ckpt":
        raise ValueError(f"Seed {seed} run summary does not select best.ckpt.")
    if resolved.get("smoke") is not False:
        raise ValueError(f"Seed {seed} is not a full non-smoke run.")
    if (
        summary.get("endpoint") != "multitask_admet_regression"
        or summary.get("task_type") != "regression"
    ):
        raise ValueError(f"Seed {seed} is not the five-head regression run.")
    _validate_scaler(scaler, config)
    if summary.get("target_scaler") != scaler:
        raise ValueError(f"Seed {seed} run-summary scaler does not match target_scaler.json.")
    split_identity = _validate_data_verification(verification, config)
    scientific = _scientific_config(resolved)
    _validate_scientific_against_config(scientific, config)
    scaler_sha = _sha256(required["scaler"])
    return {
        "checkpoint": {
            "seed": seed,
            "path": _portable_path(root, required["checkpoint"]),
            "sha256": checkpoint_sha,
            "resolved_config_path": _portable_path(root, required["resolved"]),
            "resolved_config_sha256": _sha256(required["resolved"]),
            "run_summary_path": _portable_path(root, required["summary"]),
            "run_summary_sha256": _sha256(required["summary"]),
            "target_scaler_path": _portable_path(root, required["scaler"]),
            "target_scaler_sha256": scaler_sha,
            "data_verification_path": _portable_path(root, required["verification"]),
            "data_verification_sha256": _sha256(required["verification"]),
        },
        "scientific_config": scientific,
        "scaler": scaler,
        "scaler_sha256": scaler_sha,
        "split_identity": split_identity,
    }


def _scientific_config(resolved: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "provider",
        "model_family",
        "chemprop_version",
        "dataset",
        "dataset_version",
        "protocol",
        "preprocessing",
        "target_scaling",
        "task_weighting",
        "tasks",
    )
    result = {key: resolved.get(key) for key in keys}
    result["model"] = resolved.get("resolved_model", resolved.get("model"))
    result["training"] = resolved.get("resolved_training", resolved.get("training"))
    return result


def _validate_scientific_against_config(
    scientific: Mapping[str, Any], config: ChempropExperimentConfig
) -> None:
    expected = _scientific_config(
        {
            **config.raw,
            "resolved_model": config.model,
            "resolved_training": config.training,
        }
    )
    if scientific != expected:
        raise ValueError("Resolved run configuration differs from the authoritative configuration.")


def _validate_scaler(scaler: Mapping[str, Any], config: ChempropExperimentConfig) -> None:
    if scaler.get("fit_split") != "train":
        raise ValueError("Target scaler was not fitted on training data only.")
    if (
        scaler.get("validation_statistics_used") is not False
        or scaler.get("test_statistics_used") is not False
    ):
        raise ValueError("Target scaler indicates validation or test statistics were used.")
    per_endpoint = scaler.get("per_endpoint")
    if not isinstance(per_endpoint, dict) or tuple(per_endpoint) != ENDPOINT_ORDER:
        raise ValueError("Target scaler endpoint order differs from the frozen endpoint order.")
    for endpoint in ENDPOINT_ORDER:
        item = per_endpoint[endpoint]
        if item.get("scientific_transform") != config.tasks[endpoint]["target_transform"]:
            raise ValueError(f"Target scaler transform differs for {endpoint}.")
        for field in ("mean", "scale", "train_label_count"):
            if field not in item:
                raise ValueError(f"Target scaler is missing {endpoint}.{field}.")
        if not math.isfinite(float(item["mean"])) or not math.isfinite(float(item["scale"])):
            raise ValueError(f"Target scaler contains non-finite values for {endpoint}.")
        if float(item["scale"]) <= 0 or int(item["train_label_count"]) <= 0:
            raise ValueError(f"Target scaler contains invalid values for {endpoint}.")


def _validate_data_verification(
    verification: Mapping[str, Any], config: ChempropExperimentConfig
) -> dict[str, Any]:
    if verification.get("locked_test_opened") is not False:
        raise ValueError("Run metadata does not prove locked-test isolation.")
    if verification.get("loaded_splits") != ["train", "validation"]:
        raise ValueError("Run loaded splits other than train and validation.")
    if verification.get("split_manifest_sha256") != config.raw["split_manifest_sha256"]:
        raise ValueError("Run split-manifest SHA-256 differs from the frozen configuration.")
    expected = verification.get("expected_split_hashes")
    actual = verification.get("verified_split_hashes")
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        raise ValueError("Run is missing split hash verification metadata.")
    train_validation = {
        f"{endpoint}/{split}": expected[f"{endpoint}/{split}"]
        for endpoint in ENDPOINT_ORDER
        for split in ("train", "validation")
    }
    if actual != train_validation:
        raise ValueError("Verified train/validation split hashes differ from expected hashes.")
    return {
        "split_manifest_sha256": config.raw["split_manifest_sha256"],
        "train_validation_sha256": train_validation,
    }


def _endpoint_contracts(
    config: ChempropExperimentConfig,
    metrics: Mapping[str, Any],
    scaler: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for endpoint in ENDPOINT_ORDER:
        source = config.tasks[endpoint]
        transform = source["target_transform"]
        inverse = "identity"
        internal_unit = source["units"]
        user_unit = source["units"]
        user_representation = source["units"]
        validation_space = "user_facing_output_space"
        if endpoint == "caco2_wang":
            inverse = "identity_in_stored_label_space; physical_conversion=10**y"
            user_unit = source["underlying_physical_unit"]
            user_representation = "Papp expressed in cm/s"
            validation_space = "stored_log10_papp_space_not_physical_cm_per_s"
        elif endpoint == "vdss_lombardo":
            inverse = "10**y"
            internal_unit = "log10(L/kg)"
            user_unit = "L/kg"
            user_representation = "volume of distribution at steady state in L/kg"
        result[endpoint] = {
            "order_index": ENDPOINT_ORDER.index(endpoint),
            "tdc_name": source["tdc_name"],
            "task_type": "regression",
            "target_definition": source["target_definition"],
            "scientific_transform": transform,
            "inverse_transform_for_user_output": inverse,
            "internal_model_output_unit": internal_unit,
            "user_facing_unit": user_unit,
            "user_facing_output_representation": user_representation,
            "clipping": "none",
            "training_only_scaler": scaler["per_endpoint"][endpoint],
            "validation_metrics": {
                "space": validation_space,
                "unit": source["units"],
                **metrics[endpoint],
            },
        }
    return result


def _validate_ensemble_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    if summary.get("locked_test_opened", summary.get("locked_test_accessed")) is not False:
        raise ValueError("Ensemble summary does not prove locked-test isolation.")
    if "split" in summary and summary["split"] != "validation":
        raise ValueError("Ensemble summary is not validation evidence.")
    if tuple(summary.get("seeds", ())) != PRODUCTION_SEEDS:
        raise ValueError("Ensemble summary seed set differs from production seeds.")
    if tuple(summary.get("endpoint_order", ())) != ENDPOINT_ORDER:
        raise ValueError("Ensemble summary endpoint order differs.")
    declared_rule = summary.get("ensemble_rule")
    if declared_rule is not None and declared_rule not in {
        "unweighted_arithmetic_mean",
        "unweighted_arithmetic_mean_of_exactly_all_five_seed_predictions",
    }:
        raise ValueError("Ensemble summary declares an incompatible ensemble rule.")
    declared_ddof = summary.get("ensemble_standard_deviation_ddof")
    if declared_ddof is not None and declared_ddof != 1:
        raise ValueError("Ensemble summary declares a non-sample standard deviation.")
    raw_metrics = summary.get("validation_metrics", summary.get("metrics"))
    if raw_metrics is None:
        raw_metrics = summary.get("endpoints")
    if not isinstance(raw_metrics, dict) or tuple(raw_metrics) != ENDPOINT_ORDER:
        raise ValueError("Ensemble summary metrics must use the exact endpoint order.")
    metrics: dict[str, Any] = {}
    for endpoint in ENDPOINT_ORDER:
        item = raw_metrics[endpoint]
        if isinstance(item, dict) and isinstance(item.get("metrics"), dict):
            item = item["metrics"]
        if not isinstance(item, dict):
            raise ValueError(f"Ensemble metrics for {endpoint} must be a mapping.")
        missing = set(REQUIRED_METRICS) - set(item)
        if missing:
            raise ValueError(f"Ensemble metrics for {endpoint} are missing: {sorted(missing)}")
        selected = {name: float(item[name]) for name in REQUIRED_METRICS}
        if not all(math.isfinite(value) for value in selected.values()):
            raise ValueError(f"Ensemble metrics for {endpoint} must be finite.")
        metrics[endpoint] = selected
    return metrics


def _validate_ensemble_csvs(directory: Path) -> None:
    import csv

    for endpoint in ENDPOINT_ORDER:
        path = directory / f"{endpoint}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen ensemble validation artifact: {path}")
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"seed_count", "uncertainty_interpretation", "ensemble_mean", "ensemble_std"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(f"Ensemble artifact has an incompatible schema: {path}")
            found = False
            for row in reader:
                found = True
                if int(row["seed_count"]) != 5:
                    raise ValueError(f"Ensemble artifact does not use five seeds: {path}")
                if row["uncertainty_interpretation"] != UNCERTAINTY_INTERPRETATION:
                    raise ValueError(f"Ensemble artifact has the wrong SD interpretation: {path}")
                mean = float(row["ensemble_mean"])
                standard_deviation = float(row["ensemble_std"])
                if (
                    not math.isfinite(mean)
                    or not math.isfinite(standard_deviation)
                    or standard_deviation < 0
                ):
                    raise ValueError(f"Ensemble artifact contains non-finite predictions: {path}")
            if not found:
                raise ValueError(f"Ensemble artifact is empty: {path}")


def _verify_checksum_inventory(
    root: Path, directory: Path, checksum_path: Path
) -> list[dict[str, Any]]:
    if not checksum_path.is_file():
        raise FileNotFoundError(f"Missing frozen ensemble checksum inventory: {checksum_path}")
    entries: dict[str, str] = {}
    pattern = re.compile(r"^([0-9a-fA-F]{64})\s+\*?(.+?)\s*$")
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        match = pattern.match(line)
        if not match:
            raise ValueError(f"Invalid SHA256SUMS line: {line!r}")
        token = Path(match.group(2).replace("\\", "/"))
        if token.is_absolute():
            raise ValueError("SHA256SUMS paths must be repository-root-relative.")
        path = (root / token).resolve()
        if path != root and root not in path.parents:
            raise ValueError("SHA256SUMS path escapes the artifact root.")
        portable = _portable_path(root, path)
        if portable in entries:
            raise ValueError(f"Duplicate SHA256SUMS entry: {portable}")
        entries[portable] = match.group(1).lower()
    inventory = []
    for name in REQUIRED_ENSEMBLE_ARTIFACTS:
        path = (directory / name).resolve()
        portable = _portable_path(root, path)
        if portable not in entries:
            raise ValueError(f"Required ensemble artifact is absent from SHA256SUMS: {portable}")
        if not path.is_file():
            raise FileNotFoundError(f"Missing required ensemble validation artifact: {path}")
        digest = _sha256(path)
        if digest != entries[portable]:
            raise ValueError(f"Ensemble-validation SHA-256 mismatch: {path}")
        inventory.append({"path": portable, "sha256": digest})
    return inventory


def _collect_observed_runtime() -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to record the observed manifest runtime.") from exc
    try:
        chemprop_version = importlib.metadata.version("chemprop")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("Chemprop is required to record the observed manifest runtime.") from exc
    return {
        "python": platform.python_version(),
        "chemprop": chemprop_version,
        "torch": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
        "platform": platform.platform(),
        "python_implementation": platform.python_implementation(),
        "executable": Path(sys.executable).name,
    }


def _validate_observed_runtime(observed: Mapping[str, Any], expected: Mapping[str, str]) -> None:
    required = {
        "python",
        "chemprop",
        "torch",
        "cuda_available",
        "torch_cuda_version",
        "platform",
        "python_implementation",
        "executable",
    }
    missing = sorted(required - set(observed))
    if missing:
        raise ValueError(f"Observed runtime is missing fields: {missing}")
    if not str(observed["python"]).startswith("3.11"):
        raise ValueError("Observed runtime must use Python 3.11.")
    if observed["chemprop"] != expected["chemprop"]:
        raise ValueError("Observed Chemprop version differs from the pinned contract.")
    if observed["torch"] != expected["torch"]:
        raise ValueError("Observed PyTorch version differs from the pinned contract.")


def _validate_environment_contract(
    requirements: Path, environment: Path, config: ChempropExperimentConfig
) -> dict[str, str]:
    req_versions: dict[str, str] = {}
    for line in requirements.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("--"):
            continue
        if "==" in stripped:
            name, version = stripped.split("==", 1)
            req_versions[name.lower()] = version
    env_text = environment.read_text(encoding="utf-8")
    python_match = re.search(r"(?m)^\s*-\s*python=([^\s]+)", env_text)
    torch_match = re.search(r"(?m)^\s*-\s*torch==([^\s]+)", env_text)
    if req_versions.get("chemprop") != config.raw["chemprop_version"]:
        raise ValueError("Requirements Chemprop version differs from the model contract.")
    if not python_match or not python_match.group(1).startswith("3.11"):
        raise ValueError("Environment must pin a Python 3.11 runtime.")
    if not torch_match:
        raise ValueError("Environment does not pin the GPU PyTorch runtime.")
    required_packages = ("chemprop", "lightning", "rdkit", "numpy", "pandas", "scikit-learn")
    missing = [name for name in required_packages if name not in req_versions]
    if missing:
        raise ValueError(f"Requirements are missing runtime packages: {missing}")
    return {
        "python": python_match.group(1),
        "torch": torch_match.group(1),
        **{name: req_versions[name] for name in sorted(req_versions)},
    }


def _validate_config(config: ChempropExperimentConfig) -> None:
    if config.task_type != "regression" or tuple(config.tasks) != ENDPOINT_ORDER:
        raise ValueError(
            "Production manifest requires the exact five-head regression configuration."
        )
    if tuple(config.raw.get("seeds", ())) != PRODUCTION_SEEDS:
        raise ValueError("Configuration does not contain the exact frozen production seeds.")


def _validate_exact_seed_mapping(runs: Mapping[int, str | Path], hashes: Mapping[int, str]) -> None:
    if tuple(sorted(runs)) != PRODUCTION_SEEDS:
        raise ValueError("Run directories must contain exactly seeds 13, 37, 73, 101, and 137.")
    if tuple(sorted(hashes)) != PRODUCTION_SEEDS:
        raise ValueError("Checkpoint hash contract must contain exactly the five production seeds.")
    if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes.values()):
        raise ValueError("Every checkpoint SHA-256 must be a lowercase hexadecimal digest.")


def _within_root(root: Path, value: str | Path, label: str) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{label} must remain inside the artifact root.")
    return resolved


def _portable_path(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _validate_git_commit(value: str) -> str:
    normalized = value.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", normalized):
        raise ValueError("Git commit must be a full 40-character hexadecimal identity.")
    return normalized


__all__ = [
    "CHECKPOINT_SHA256",
    "ENDPOINT_ORDER",
    "PRODUCTION_SEEDS",
    "RELEASE_STATUS",
    "create_regression_production_manifest",
    "current_git_commit",
]
