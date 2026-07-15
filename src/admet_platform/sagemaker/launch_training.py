"""SageMaker ChemBERTa training-job launcher.

The dry-run path is intentionally self-contained: it validates the requested
job and writes a launch plan without constructing boto3 or SageMaker clients.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - requirements include PyYAML.
    yaml = None  # type: ignore[assignment]

from admet_platform.config import EndpointConfig, load_endpoint_config
from admet_platform.models.artifacts import write_json


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "sagemaker_training.yaml"
JOB_NAME_MAX_LENGTH = 63
ALLOWED_MODEL_TYPES = {"chemberta"}
HYPERPARAMETER_KEYS = {
    "endpoint_config",
    "model_name",
    "max_sequence_length",
    "learning_rate",
    "epochs",
    "train_batch_size",
    "evaluation_batch_size",
    "weight_decay",
    "early_stopping_patience",
    "random_seed",
    "development_row_limit",
    "local_files_only",
    "cache_dir",
}
SENSITIVE_PATTERNS = (
    re.compile(r"aws_secret_access_key\s*=\s*[^,\s]+", re.IGNORECASE),
    re.compile(r"aws_access_key_id\s*=\s*[^,\s]+", re.IGNORECASE),
    re.compile(r"aws_session_token\s*=\s*[^,\s]+", re.IGNORECASE),
    re.compile(r"hf_token\s*=\s*[^,\s]+", re.IGNORECASE),
)
S3_URI_PATTERN = re.compile(r"^s3://([a-z0-9][a-z0-9.-]{1,61}[a-z0-9])/(.+)$")


@dataclass(frozen=True)
class LaunchResult:
    """Result from a dry-run or submitted launch."""

    status: str
    job_name: str
    manifest_path: Path
    manifest: dict[str, Any]


class SageMakerSubmissionError(RuntimeError):
    """Raised when real SageMaker submission fails."""


def load_training_yaml(path: str | Path) -> dict[str, Any]:
    """Load SageMaker execution YAML as a mapping."""

    config_path = Path(path)
    raw_text = config_path.read_text(encoding="utf-8")
    if yaml is None:
        raise RuntimeError("PyYAML is required to load SageMaker training configuration.")
    payload = yaml.safe_load(raw_text) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"SageMaker training config {config_path} must contain a YAML mapping.")
    return payload


def run_launch(
    training_config_path: str | Path,
    *,
    dry_run: bool,
    launch_plan_output: str | Path | None = None,
    wait: bool = True,
    cli_overrides: dict[str, Any] | None = None,
    now: datetime | None = None,
    suffix: str | None = None,
    sagemaker_session_factory: Any | None = None,
    estimator_class: Any | None = None,
) -> LaunchResult:
    """Validate, dry-run, or submit a SageMaker ChemBERTa training job."""

    raw_config = load_training_yaml(training_config_path)
    effective = build_effective_config(raw_config, cli_overrides or {})
    endpoint_config = load_endpoint_config(effective["endpoint_config"])
    validate_effective_config(effective, endpoint_config)
    job_name = generate_job_name(
        effective["job_name_prefix"],
        endpoint_config.endpoint_id,
        effective["model_type"],
        now=now,
        suffix=suffix,
    )
    hyperparameters = build_hyperparameters(effective)
    image_strategy = resolve_image_strategy(effective["image"])
    channels = build_training_inputs(effective, dry_run=dry_run)
    source_description = validate_source_package_inputs(
        effective["source_dir"],
        effective["entry_point"],
        effective["endpoint_config"],
        project_root=PROJECT_ROOT,
    )
    warnings = build_warnings(effective)
    plan = build_launch_manifest(
        status="dry_run" if dry_run else "submitted",
        job_name=job_name,
        endpoint_config=endpoint_config,
        effective=effective,
        image_strategy=image_strategy,
        channels=channels,
        hyperparameters=hyperparameters,
        source_description=source_description,
        warnings=warnings,
    )

    output_path = Path(launch_plan_output or effective.get("launch_plan_output") or "launch_plan.json")
    if dry_run:
        write_json(output_path, plan)
        return LaunchResult(status="dry_run", job_name=job_name, manifest_path=output_path, manifest=plan)

    try:
        package_dir = prepare_source_package(
            source_dir=effective["source_dir"],
            entry_point=effective["entry_point"],
            endpoint_config_path=effective["endpoint_config"],
            job_name=job_name,
            project_root=PROJECT_ROOT,
        )
        estimator_args = build_estimator_args(
            effective=effective,
            image_strategy=image_strategy,
            hyperparameters=hyperparameters,
            source_dir=package_dir,
        )
        session = create_sagemaker_session(effective["region"], sagemaker_session_factory)
        estimator_args["sagemaker_session"] = session
        estimator = create_huggingface_estimator(estimator_args, estimator_class)
        estimator.fit(channels, job_name=job_name, wait=wait)
        plan["status"] = "submitted"
        plan["wait"] = wait
        write_json(output_path, plan)
        return LaunchResult(status="submitted", job_name=job_name, manifest_path=output_path, manifest=plan)
    except Exception as exc:  # noqa: BLE001 - CLI must convert submission failure to nonzero.
        failure = dict(plan)
        failure["status"] = "failed"
        failure["error"] = sanitize_text(str(exc))
        write_json(output_path, failure)
        raise SageMakerSubmissionError(sanitize_text(str(exc))) from exc


def build_effective_config(raw_config: dict[str, Any], cli_overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply CLI-over-YAML-over-default precedence without mutating endpoint YAML."""

    aws = _mapping(raw_config.get("aws"))
    s3 = _mapping(raw_config.get("s3"))
    compute = _mapping(raw_config.get("compute"))
    job = _mapping(raw_config.get("job"))
    source = _mapping(raw_config.get("source"))
    security = _mapping(raw_config.get("security"))

    effective: dict[str, Any] = {
        "endpoint_config": raw_config.get("endpoint_config"),
        "model_type": raw_config.get("model_type", "chemberta"),
        "train_s3_uri": s3.get("train"),
        "validation_s3_uri": s3.get("validation"),
        "test_s3_uri": s3.get("test"),
        "region": aws.get("region"),
        "role_arn": aws.get("role_arn"),
        "output_s3_path": s3.get("output"),
        "checkpoint_s3_uri": s3.get("checkpoint"),
        "instance_type": compute.get("instance_type", "ml.g5.xlarge"),
        "instance_count": compute.get("instance_count", 1),
        "volume_size": compute.get("volume_size", 100),
        "max_runtime": compute.get("max_runtime", 3600),
        "input_mode": raw_config.get("input_mode", "File"),
        "job_name_prefix": job.get("name_prefix", "admet"),
        "source_dir": source.get("source_dir", "sagemaker"),
        "entry_point": source.get("entry_point", "train_chemberta.py"),
        "image": _mapping(raw_config.get("image")),
        "hyperparameters": _mapping(raw_config.get("hyperparameters")),
        "tags": raw_config.get("tags", []),
        "kms_key_arn": security.get("kms_key_arn"),
        "enable_network_isolation": security.get("enable_network_isolation", False),
        "vpc_subnets": security.get("vpc_subnets", []),
        "vpc_security_group_ids": security.get("vpc_security_group_ids", []),
        "launch_plan_output": raw_config.get("launch_plan_output"),
    }
    for key, value in cli_overrides.items():
        if value is None:
            continue
        if key == "hyperparameters":
            effective["hyperparameters"] = {**effective["hyperparameters"], **value}
        elif key == "tags":
            effective["tags"] = value
        elif key == "image":
            effective["image"] = {**effective["image"], **value}
        else:
            effective[key] = value
    if effective.get("endpoint_config"):
        effective["endpoint_config"] = _public_path((PROJECT_ROOT / effective["endpoint_config"]).resolve(), PROJECT_ROOT)
    return effective


def validate_effective_config(config: dict[str, Any], endpoint_config: EndpointConfig) -> None:
    required = {
        "endpoint_config": "endpoint config path",
        "model_type": "model type",
        "train_s3_uri": "train S3 URI",
        "validation_s3_uri": "validation S3 URI",
        "test_s3_uri": "test S3 URI",
        "region": "AWS region",
        "role_arn": "SageMaker execution role ARN",
        "output_s3_path": "output S3 path",
        "instance_type": "instance type",
        "instance_count": "instance count",
        "volume_size": "volume size",
        "max_runtime": "maximum runtime",
        "job_name_prefix": "job-name prefix",
        "source_dir": "source directory",
        "entry_point": "entry-point filename",
    }
    missing = [label for key, label in required.items() if config.get(key) in (None, "")]
    if missing:
        raise ValueError(f"Missing required SageMaker launcher field(s): {', '.join(missing)}.")
    if config["model_type"] not in ALLOWED_MODEL_TYPES:
        raise ValueError("Only model_type 'chemberta' is supported for SageMaker launch.")
    for key in ("train_s3_uri", "validation_s3_uri", "test_s3_uri", "output_s3_path"):
        validate_s3_uri(config[key], key)
    if config.get("checkpoint_s3_uri"):
        validate_s3_uri(config["checkpoint_s3_uri"], "checkpoint_s3_uri")
    if not re.match(r"^arn:aws[a-zA-Z-]*:iam::\d{12}:role\/[A-Za-z0-9+=,.@_/-]+$", config["role_arn"]):
        raise ValueError("role_arn must be a syntactically valid IAM role ARN.")
    if int(config["instance_count"]) < 1:
        raise ValueError("instance_count must be at least 1.")
    if int(config["volume_size"]) < 1:
        raise ValueError("volume_size must be at least 1.")
    if int(config["max_runtime"]) < 1:
        raise ValueError("max_runtime must be at least 1 second.")
    if config.get("input_mode", "File") != "File":
        raise ValueError("Only SageMaker File input mode is supported by this launcher.")
    resolve_image_strategy(config["image"])
    validate_tags(config.get("tags", []))
    validate_network_config(config)
    if config["hyperparameters"].get("endpoint_config") not in (None, config["endpoint_config"]):
        raise ValueError("hyperparameters.endpoint_config must match the selected endpoint config path.")
    _ = endpoint_config


def resolve_image_strategy(image_config: dict[str, Any]) -> dict[str, Any]:
    strategy = image_config.get("strategy")
    image_uri = image_config.get("image_uri")
    managed_fields = {
        "transformers_version": image_config.get("transformers_version"),
        "pytorch_version": image_config.get("pytorch_version"),
        "py_version": image_config.get("py_version"),
    }
    has_managed = any(value not in (None, "") for value in managed_fields.values())
    if strategy == "managed":
        missing = [key for key, value in managed_fields.items() if value in (None, "")]
        if missing:
            raise ValueError(f"Managed Hugging Face image configuration is missing: {', '.join(missing)}.")
        if image_uri:
            raise ValueError("Managed image mode cannot also specify image_uri.")
        return {"strategy": "managed", **managed_fields}
    if strategy == "custom":
        if not image_uri:
            raise ValueError("Custom image mode requires image_uri.")
        if has_managed:
            raise ValueError("Custom image mode cannot include managed framework fields.")
        return {"strategy": "custom", "image_uri": image_uri}
    if image_uri and has_managed:
        raise ValueError("Conflicting image configuration: choose managed fields or image_uri, not both.")
    raise ValueError("Image configuration must set strategy to either 'managed' or 'custom'.")


def validate_s3_uri(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not S3_URI_PATTERN.match(value):
        raise ValueError(f"{field_name} must be a valid s3://bucket/key URI.")


def build_training_inputs(config: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    channels = {
        "train": config["train_s3_uri"],
        "validation": config["validation_s3_uri"],
        "test": config["test_s3_uri"],
    }
    if dry_run:
        return {
            name: {"s3_data": uri, "content_type": "text/csv", "input_mode": config.get("input_mode", "File")}
            for name, uri in channels.items()
        }

    try:
        from sagemaker.inputs import TrainingInput
    except ModuleNotFoundError as exc:  # pragma: no cover - covered by dry-run tests.
        raise RuntimeError("sagemaker is required for real submission. Install requirements.txt.") from exc
    return {
        name: TrainingInput(s3_data=uri, content_type="text/csv", input_mode=config.get("input_mode", "File"))
        for name, uri in channels.items()
    }


def build_hyperparameters(config: dict[str, Any]) -> dict[str, str]:
    raw = dict(config.get("hyperparameters", {}))
    raw["endpoint_config"] = config["endpoint_config"]
    converted: dict[str, str] = {}
    for key in HYPERPARAMETER_KEYS:
        value = raw.get(key)
        if value is None:
            continue
        converted[key.replace("_", "-")] = _stringify_hyperparameter(value)
    return dict(sorted(converted.items()))


def build_estimator_args(
    *,
    effective: dict[str, Any],
    image_strategy: dict[str, Any],
    hyperparameters: dict[str, str],
    source_dir: Path,
) -> dict[str, Any]:
    args: dict[str, Any] = {
        "entry_point": effective["entry_point"],
        "source_dir": str(source_dir),
        "role": effective["role_arn"],
        "instance_type": effective["instance_type"],
        "instance_count": int(effective["instance_count"]),
        "hyperparameters": hyperparameters,
        "output_path": effective["output_s3_path"],
        "max_run": int(effective["max_runtime"]),
        "volume_size": int(effective["volume_size"]),
        "tags": effective.get("tags", []),
    }
    if effective.get("checkpoint_s3_uri"):
        args["checkpoint_s3_uri"] = effective["checkpoint_s3_uri"]
    if effective.get("kms_key_arn"):
        args["output_kms_key"] = effective["kms_key_arn"]
    if effective.get("enable_network_isolation"):
        args["enable_network_isolation"] = True
    if effective.get("vpc_subnets"):
        args["subnets"] = effective["vpc_subnets"]
    if effective.get("vpc_security_group_ids"):
        args["security_group_ids"] = effective["vpc_security_group_ids"]
    if image_strategy["strategy"] == "custom":
        args["image_uri"] = image_strategy["image_uri"]
    else:
        args.update(
            {
                "transformers_version": image_strategy["transformers_version"],
                "pytorch_version": image_strategy["pytorch_version"],
                "py_version": image_strategy["py_version"],
            }
        )
    return args


def create_sagemaker_session(region: str, session_factory: Any | None = None) -> Any:
    if session_factory is not None:
        return session_factory(region)
    try:
        import boto3
        import sagemaker
    except ModuleNotFoundError as exc:  # pragma: no cover - covered by dry-run tests.
        raise RuntimeError("boto3 and sagemaker are required for real submission.") from exc
    boto_session = boto3.Session(region_name=region)
    return sagemaker.Session(boto_session=boto_session)


def create_huggingface_estimator(estimator_args: dict[str, Any], estimator_class: Any | None = None) -> Any:
    if estimator_class is None:
        try:
            from sagemaker.huggingface import HuggingFace
        except ModuleNotFoundError as exc:  # pragma: no cover - covered by dry-run tests.
            raise RuntimeError("sagemaker is required for real submission. Install requirements.txt.") from exc
        estimator_class = HuggingFace
    return estimator_class(**estimator_args)


def generate_job_name(
    prefix: str,
    endpoint_id: str,
    model_type: str,
    *,
    now: datetime | None = None,
    suffix: str | None = None,
) -> str:
    timestamp = (now or datetime.now(UTC)).strftime("%Y%m%d-%H%M%S")
    suffix_value = sanitize_job_component(suffix or uuid.uuid4().hex[:8])
    base = "-".join(
        sanitize_job_component(part)
        for part in [prefix, endpoint_id, model_type, timestamp, suffix_value]
        if part
    )
    base = re.sub(r"-+", "-", base).strip("-")
    if len(base) <= JOB_NAME_MAX_LENGTH:
        return base
    overflow = len(base) - JOB_NAME_MAX_LENGTH
    endpoint_part = sanitize_job_component(endpoint_id)
    shortened_endpoint = endpoint_part[: max(6, len(endpoint_part) - overflow)]
    shortened = "-".join(
        sanitize_job_component(part)
        for part in [prefix, shortened_endpoint, model_type, timestamp, suffix_value]
        if part
    )
    return shortened[:JOB_NAME_MAX_LENGTH].rstrip("-")


def sanitize_job_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9-]+", "-", str(value).lower()).strip("-")
    return sanitized or "job"


def validate_source_package_inputs(
    source_dir: str | Path,
    entry_point: str,
    endpoint_config_path: str | Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    source_path = (project_root / source_dir).resolve()
    entry_path = source_path / entry_point
    package_path = (project_root / "src" / "admet_platform").resolve()
    requirements_path = source_path / "requirements.txt"
    endpoint_path = (project_root / endpoint_config_path).resolve()
    _relative_project_path(endpoint_path, project_root)
    missing = [
        str(path)
        for path in [source_path, entry_path, package_path, requirements_path, endpoint_path]
        if not path.exists()
    ]
    if missing:
        raise ValueError(f"Source package input(s) are missing: {', '.join(missing)}.")
    return {
        "strategy": "copy_wrapper_package_and_requirements",
        "source_dir": _public_path(source_path, project_root),
        "entry_point": entry_point,
        "includes": [
            _public_path(entry_path, project_root),
            _public_path(package_path, project_root),
            _public_path(requirements_path, project_root),
            _public_path(endpoint_path, project_root),
        ],
        "entry_point_at_package_root": True,
    }


def prepare_source_package(
    *,
    source_dir: str | Path,
    entry_point: str,
    endpoint_config_path: str | Path,
    job_name: str,
    project_root: Path,
) -> Path:
    package_root = Path(tempfile.mkdtemp(prefix=f"{job_name}-source-"))
    source_path = (project_root / source_dir).resolve()
    shutil.copy2(source_path / entry_point, package_root / entry_point)
    shutil.copytree(project_root / "src" / "admet_platform", package_root / "src" / "admet_platform")
    shutil.copy2(source_path / "requirements.txt", package_root / "requirements.txt")
    endpoint_source = (project_root / endpoint_config_path).resolve()
    endpoint_destination = package_root / _relative_project_path(endpoint_source, project_root)
    endpoint_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(endpoint_source, endpoint_destination)
    return package_root


def build_launch_manifest(
    *,
    status: str,
    job_name: str,
    endpoint_config: EndpointConfig,
    effective: dict[str, Any],
    image_strategy: dict[str, Any],
    channels: dict[str, Any],
    hyperparameters: dict[str, str],
    source_description: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "status": status,
        "job_name": job_name,
        "region": effective["region"],
        "role_arn": sanitize_role_arn(effective["role_arn"]),
        "endpoint_id": endpoint_config.endpoint_id,
        "source_dataset": endpoint_config.tdc_name,
        "task_type": endpoint_config.task_type,
        "model_type": effective["model_type"],
        "image_strategy": image_strategy["strategy"],
        "image": redact_mapping(image_strategy),
        "instance_settings": {
            "instance_type": effective["instance_type"],
            "instance_count": int(effective["instance_count"]),
            "volume_size": int(effective["volume_size"]),
        },
        "s3_channels": redact_mapping(channels),
        "output_s3_path": effective["output_s3_path"],
        "checkpoint_s3_uri": effective.get("checkpoint_s3_uri"),
        "hyperparameters": redact_mapping(hyperparameters),
        "tags": redact_mapping({"tags": effective.get("tags", [])})["tags"],
        "source_packaging": source_description,
        "maximum_runtime": int(effective["max_runtime"]),
        "development_mode": hyperparameters.get("development-row-limit") is not None,
        "network": {
            "enable_network_isolation": bool(effective.get("enable_network_isolation")),
            "vpc_subnets": effective.get("vpc_subnets", []),
            "vpc_security_group_ids": effective.get("vpc_security_group_ids", []),
        },
        "kms_key_arn": effective.get("kms_key_arn"),
        "effective_configuration": redact_mapping(_public_effective_config(effective)),
        "warnings": warnings,
    }


def build_warnings(config: dict[str, Any]) -> list[str]:
    warnings: list[str] = [
        "Dry run validates configuration only and does not verify remote S3 object existence.",
        "This launcher submits a single-split baseline fine-tuning job; do not treat results as production-ready.",
    ]
    if config.get("hyperparameters", {}).get("development_row_limit") is not None:
        warnings.append("development_row_limit is set; this is development-only and not a full benchmark run.")
    return warnings


def validate_tags(tags: Any) -> None:
    if not isinstance(tags, list):
        raise ValueError("tags must be a list of Key/Value mappings.")
    for tag in tags:
        if not isinstance(tag, dict) or not tag.get("Key") or not tag.get("Value"):
            raise ValueError("Each tag must contain non-empty Key and Value fields.")


def validate_network_config(config: dict[str, Any]) -> None:
    for key in ("vpc_subnets", "vpc_security_group_ids"):
        if not isinstance(config.get(key, []), list):
            raise ValueError(f"{key} must be a list.")


def sanitize_role_arn(value: str) -> str:
    return re.sub(r"(?<=:iam::)\d{12}(?=:role/)", "************", value)


def redact_mapping(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: redact_mapping(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_mapping(item) for item in value]
    if isinstance(value, str):
        return sanitize_text(value)
    return value


def sanitize_text(value: str) -> str:
    redacted = value
    for pattern in SENSITIVE_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _public_effective_config(config: dict[str, Any]) -> dict[str, Any]:
    public_config = dict(config)
    public_config["role_arn"] = sanitize_role_arn(public_config["role_arn"])
    public_config["source_dir"] = str(public_config["source_dir"])
    return public_config


def _public_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve())).replace("\\", "/")
    except ValueError:
        return path.name


def _relative_project_path(path: Path, project_root: Path) -> Path:
    try:
        return path.resolve().relative_to(project_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Path must be inside the project root for SageMaker packaging: {path}.") from exc


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("Expected YAML mapping.")
    return value


def _stringify_hyperparameter(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def parse_key_value(items: list[str] | None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE override, got {item!r}.")
        key, value = item.split("=", 1)
        if not key:
            raise ValueError(f"Expected non-empty KEY in override {item!r}.")
        parsed[key] = value
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch or dry-run a SageMaker ChemBERTa training job.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="SageMaker execution YAML.")
    parser.add_argument("--endpoint-config")
    parser.add_argument("--model-type")
    parser.add_argument("--train-s3-uri")
    parser.add_argument("--validation-s3-uri")
    parser.add_argument("--test-s3-uri")
    parser.add_argument("--region")
    parser.add_argument("--role-arn")
    parser.add_argument("--output-s3-path")
    parser.add_argument("--checkpoint-s3-uri")
    parser.add_argument("--instance-type")
    parser.add_argument("--instance-count", type=int)
    parser.add_argument("--volume-size", type=int)
    parser.add_argument("--max-runtime", type=int)
    parser.add_argument("--job-name-prefix")
    parser.add_argument("--source-dir")
    parser.add_argument("--entry-point")
    parser.add_argument("--image-strategy", choices=["managed", "custom"])
    parser.add_argument("--image-uri")
    parser.add_argument("--transformers-version")
    parser.add_argument("--pytorch-version")
    parser.add_argument("--py-version")
    parser.add_argument("--hyperparameter", action="append", default=[])
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--kms-key-arn")
    parser.add_argument("--enable-network-isolation", action="store_true")
    parser.add_argument("--vpc-subnet", action="append", default=[])
    parser.add_argument("--vpc-security-group", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--launch-plan-output", default="launch_plan.json")
    parser.add_argument("--wait", dest="wait", action="store_true", default=True)
    parser.add_argument("--no-wait", dest="wait", action="store_false")
    return parser


def cli_overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    image = {
        key: value
        for key, value in {
            "strategy": args.image_strategy,
            "image_uri": args.image_uri,
            "transformers_version": args.transformers_version,
            "pytorch_version": args.pytorch_version,
            "py_version": args.py_version,
        }.items()
        if value is not None
    }
    overrides = {
        "endpoint_config": args.endpoint_config,
        "model_type": args.model_type,
        "train_s3_uri": args.train_s3_uri,
        "validation_s3_uri": args.validation_s3_uri,
        "test_s3_uri": args.test_s3_uri,
        "region": args.region,
        "role_arn": args.role_arn,
        "output_s3_path": args.output_s3_path,
        "checkpoint_s3_uri": args.checkpoint_s3_uri,
        "instance_type": args.instance_type,
        "instance_count": args.instance_count,
        "volume_size": args.volume_size,
        "max_runtime": args.max_runtime,
        "job_name_prefix": args.job_name_prefix,
        "source_dir": args.source_dir,
        "entry_point": args.entry_point,
        "image": image or None,
        "hyperparameters": parse_key_value(args.hyperparameter),
        "tags": [{"Key": key, "Value": value} for key, value in parse_key_value(args.tag).items()] or None,
        "kms_key_arn": args.kms_key_arn,
        "enable_network_isolation": True if args.enable_network_isolation else None,
        "vpc_subnets": args.vpc_subnet or None,
        "vpc_security_group_ids": args.vpc_security_group or None,
    }
    return {key: value for key, value in overrides.items() if value not in (None, {}, [])}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_launch(
            args.config,
            dry_run=args.dry_run,
            launch_plan_output=args.launch_plan_output,
            wait=args.wait,
            cli_overrides=cli_overrides_from_args(args),
        )
    except Exception as exc:  # noqa: BLE001 - CLI returns nonzero with clear message.
        parser.exit(1, f"SageMaker launch failed: {sanitize_text(str(exc))}\n")
    print(f"{result.status}: {result.job_name}")
    print(f"Manifest: {result.manifest_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
