"""SageMaker Processing launcher for ADMET TDC dataset preparation."""

from __future__ import annotations

import argparse
import shutil
import tempfile
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
from admet_platform.sagemaker.launch_training import (
    JOB_NAME_MAX_LENGTH,
    _public_path,
    generate_job_name,
    redact_mapping,
    sanitize_role_arn,
    sanitize_text,
    validate_s3_uri,
    validate_tags,
)
from admet_platform.sagemaker.prepare_tdc_dataset import PROCESSING_MODES


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "sagemaker_processing.yaml"


@dataclass(frozen=True)
class ProcessingLaunchResult:
    """Result from a dry-run or submitted Processing job."""

    status: str
    job_name: str
    manifest_path: Path
    manifest: dict[str, Any]


class SageMakerProcessingSubmissionError(RuntimeError):
    """Raised when real SageMaker Processing submission fails."""


def load_processing_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if yaml is None:
        raise RuntimeError("PyYAML is required to load SageMaker Processing configuration.")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"SageMaker Processing config {config_path} must contain a YAML mapping.")
    return payload


def run_processing_launch(
    processing_config_path: str | Path,
    *,
    dry_run: bool,
    launch_plan_output: str | Path | None = None,
    wait: bool = True,
    cli_overrides: dict[str, Any] | None = None,
    now: datetime | None = None,
    suffix: str | None = None,
    sagemaker_session_factory: Any | None = None,
    processor_class: Any | None = None,
) -> ProcessingLaunchResult:
    raw_config = load_processing_yaml(processing_config_path)
    effective = build_effective_config(raw_config, cli_overrides or {})
    endpoint_config = load_endpoint_config(effective["endpoint_config"])
    validate_effective_config(effective, endpoint_config)
    job_name = generate_processing_job_name(
        effective["job_name_prefix"],
        endpoint_config.endpoint_id,
        now=now,
        suffix=suffix,
    )
    inputs = build_processing_inputs(effective, dry_run=dry_run)
    outputs = build_processing_outputs(effective, dry_run=dry_run)
    arguments = build_container_arguments(effective)
    source_description = validate_source_package_inputs(
        effective["source_dir"],
        effective["entry_point"],
        effective["endpoint_config"],
        project_root=PROJECT_ROOT,
    )
    warnings = build_warnings(effective)
    plan = build_launch_plan(
        status="dry_run" if dry_run else "submitted",
        job_name=job_name,
        endpoint_config=endpoint_config,
        effective=effective,
        inputs=inputs,
        outputs=outputs,
        arguments=arguments,
        source_description=source_description,
        warnings=warnings,
    )
    output_path = Path(launch_plan_output or effective.get("launch_plan_output") or "processing_launch_plan.json")

    if dry_run:
        write_json(output_path, plan)
        return ProcessingLaunchResult("dry_run", job_name, output_path, plan)

    try:
        package_dir = prepare_source_package(
            source_dir=effective["source_dir"],
            entry_point=effective["entry_point"],
            endpoint_config_path=effective["endpoint_config"],
            job_name=job_name,
            project_root=PROJECT_ROOT,
        )
        processor_args = build_processor_args(effective=effective, source_dir=package_dir)
        session = create_sagemaker_session(effective["region"], sagemaker_session_factory)
        processor_args["sagemaker_session"] = session
        processor = create_processor(processor_args, processor_class)
        processor.run(
            inputs=inputs,
            outputs=outputs,
            arguments=arguments,
            job_name=job_name,
            wait=wait,
        )
        plan["status"] = "submitted"
        plan["wait"] = wait
        write_json(output_path, plan)
        return ProcessingLaunchResult("submitted", job_name, output_path, plan)
    except Exception as exc:  # noqa: BLE001 - CLI must return nonzero.
        failure = dict(plan)
        failure["status"] = "failed"
        failure["error"] = sanitize_text(str(exc))
        write_json(output_path, failure)
        raise SageMakerProcessingSubmissionError(sanitize_text(str(exc))) from exc


def build_effective_config(raw_config: dict[str, Any], cli_overrides: dict[str, Any]) -> dict[str, Any]:
    aws = _mapping(raw_config.get("aws"))
    s3 = _mapping(raw_config.get("s3"))
    compute = _mapping(raw_config.get("compute"))
    job = _mapping(raw_config.get("job"))
    source = _mapping(raw_config.get("source"))
    security = _mapping(raw_config.get("security"))
    effective: dict[str, Any] = {
        "endpoint_config": raw_config.get("endpoint_config"),
        "processing_mode": raw_config.get("processing_mode"),
        "source_csv_s3_uri": s3.get("source_csv"),
        "output_s3_prefix": s3.get("output_prefix"),
        "region": aws.get("region"),
        "role_arn": aws.get("role_arn"),
        "image_uri": raw_config.get("image_uri"),
        "instance_type": compute.get("instance_type", "ml.m5.xlarge"),
        "instance_count": compute.get("instance_count", 1),
        "volume_size": compute.get("volume_size", 50),
        "max_runtime": compute.get("max_runtime", 1800),
        "job_name_prefix": job.get("name_prefix", "admet-processing"),
        "source_dir": source.get("source_dir", "sagemaker"),
        "entry_point": source.get("entry_point", "prepare_tdc_dataset.py"),
        "tags": raw_config.get("tags", []),
        "kms_key_arn": security.get("kms_key_arn"),
        "vpc_subnets": security.get("vpc_subnets", []),
        "vpc_security_group_ids": security.get("vpc_security_group_ids", []),
        "development_row_limit": raw_config.get("development_row_limit"),
        "launch_plan_output": raw_config.get("launch_plan_output"),
    }
    for key, value in cli_overrides.items():
        if value is not None:
            effective[key] = value
    if effective.get("endpoint_config"):
        effective["endpoint_config"] = _public_path((PROJECT_ROOT / effective["endpoint_config"]).resolve(), PROJECT_ROOT)
    return effective


def validate_effective_config(config: dict[str, Any], endpoint_config: EndpointConfig) -> None:
    required = {
        "endpoint_config": "endpoint config path",
        "processing_mode": "processing mode",
        "output_s3_prefix": "output S3 prefix",
        "region": "AWS region",
        "role_arn": "SageMaker execution role ARN",
        "image_uri": "processing image URI",
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
        raise ValueError(f"Missing required SageMaker Processing field(s): {', '.join(missing)}.")
    if config["processing_mode"] not in PROCESSING_MODES:
        allowed = ", ".join(sorted(PROCESSING_MODES))
        raise ValueError(f"processing_mode must be one of: {allowed}.")
    if config["processing_mode"] == "supplied_csv" and not config.get("source_csv_s3_uri"):
        raise ValueError("source_csv_s3_uri is required in supplied_csv mode.")
    if config.get("source_csv_s3_uri"):
        validate_s3_uri(config["source_csv_s3_uri"], "source_csv_s3_uri")
    validate_s3_uri(config["output_s3_prefix"], "output_s3_prefix")
    if not str(config["image_uri"]).strip():
        raise ValueError("processing image_uri is required.")
    if not str(config["role_arn"]).startswith("arn:aws:iam::"):
        raise ValueError("role_arn must be a syntactically valid IAM role ARN.")
    if int(config["instance_count"]) < 1:
        raise ValueError("instance_count must be at least 1.")
    if int(config["volume_size"]) < 1:
        raise ValueError("volume_size must be at least 1.")
    if int(config["max_runtime"]) < 1:
        raise ValueError("max_runtime must be at least 1 second.")
    validate_tags(config.get("tags", []))
    for key in ("vpc_subnets", "vpc_security_group_ids"):
        if not isinstance(config.get(key, []), list):
            raise ValueError(f"{key} must be a list.")
    _ = endpoint_config


def build_container_arguments(config: dict[str, Any]) -> list[str]:
    args = ["--mode", config["processing_mode"], "--endpoint-config", config["endpoint_config"]]
    if config.get("development_row_limit") is not None:
        args.extend(["--development-row-limit", str(config["development_row_limit"])])
    return args


def build_processing_inputs(config: dict[str, Any], *, dry_run: bool) -> list[Any]:
    inputs = [
        _processing_input(
            source=config["endpoint_config"],
            destination="/opt/ml/processing/input/config",
            input_name="config",
            dry_run=dry_run,
        )
    ]
    if config["processing_mode"] == "supplied_csv":
        inputs.append(
            _processing_input(
                source=config["source_csv_s3_uri"],
                destination="/opt/ml/processing/input/data",
                input_name="data",
                dry_run=dry_run,
            )
        )
    return inputs


def build_processing_outputs(config: dict[str, Any], *, dry_run: bool) -> list[Any]:
    return [
        _processing_output(
            source="/opt/ml/processing/output/train",
            destination=_join_s3(config["output_s3_prefix"], "train"),
            output_name="train",
            dry_run=dry_run,
        ),
        _processing_output(
            source="/opt/ml/processing/output/validation",
            destination=_join_s3(config["output_s3_prefix"], "validation"),
            output_name="validation",
            dry_run=dry_run,
        ),
        _processing_output(
            source="/opt/ml/processing/output/test",
            destination=_join_s3(config["output_s3_prefix"], "test"),
            output_name="test",
            dry_run=dry_run,
        ),
        _processing_output(
            source="/opt/ml/processing/output/metadata",
            destination=_join_s3(config["output_s3_prefix"], "metadata"),
            output_name="metadata",
            dry_run=dry_run,
        ),
    ]


def build_processor_args(*, effective: dict[str, Any], source_dir: Path) -> dict[str, Any]:
    args: dict[str, Any] = {
        "role": effective["role_arn"],
        "image_uri": effective["image_uri"],
        "instance_type": effective["instance_type"],
        "instance_count": int(effective["instance_count"]),
        "volume_size_in_gb": int(effective["volume_size"]),
        "max_runtime_in_seconds": int(effective["max_runtime"]),
        "entrypoint": ["python", effective["entry_point"]],
        "base_job_name": effective["job_name_prefix"],
        "source_dir": str(source_dir),
        "tags": effective.get("tags", []),
    }
    if effective.get("kms_key_arn"):
        args["output_kms_key"] = effective["kms_key_arn"]
    if effective.get("vpc_subnets"):
        args["subnets"] = effective["vpc_subnets"]
    if effective.get("vpc_security_group_ids"):
        args["security_group_ids"] = effective["vpc_security_group_ids"]
    return args


def create_sagemaker_session(region: str, session_factory: Any | None = None) -> Any:
    if session_factory is not None:
        return session_factory(region)
    try:
        import boto3
        import sagemaker
    except ModuleNotFoundError as exc:  # pragma: no cover - dry-run avoids this path.
        raise RuntimeError("boto3 and sagemaker are required for real Processing submission.") from exc
    boto_session = boto3.Session(region_name=region)
    return sagemaker.Session(boto_session=boto_session)


def create_processor(processor_args: dict[str, Any], processor_class: Any | None = None) -> Any:
    if processor_class is None:
        try:
            from sagemaker.processing import Processor
        except ModuleNotFoundError as exc:  # pragma: no cover - dry-run avoids this path.
            raise RuntimeError("sagemaker is required for real Processing submission.") from exc
        processor_class = Processor
    return processor_class(**processor_args)


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
    requirements_path = source_path / "processing_requirements.txt"
    endpoint_path = (project_root / endpoint_config_path).resolve()
    missing = [
        str(path)
        for path in [source_path, entry_path, package_path, requirements_path, endpoint_path]
        if not path.exists()
    ]
    if missing:
        raise ValueError(f"Processing source package input(s) are missing: {', '.join(missing)}.")
    return {
        "strategy": "copy_processing_wrapper_package_config_and_requirements",
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
    package_root = Path(tempfile.mkdtemp(prefix=f"{job_name}-processing-source-"))
    source_path = (project_root / source_dir).resolve()
    shutil.copy2(source_path / entry_point, package_root / entry_point)
    shutil.copytree(project_root / "src" / "admet_platform", package_root / "src" / "admet_platform")
    shutil.copy2(source_path / "processing_requirements.txt", package_root / "processing_requirements.txt")
    endpoint_source = (project_root / endpoint_config_path).resolve()
    endpoint_destination = package_root / Path(endpoint_config_path)
    endpoint_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(endpoint_source, endpoint_destination)
    return package_root


def build_launch_plan(
    *,
    status: str,
    job_name: str,
    endpoint_config: EndpointConfig,
    effective: dict[str, Any],
    inputs: list[Any],
    outputs: list[Any],
    arguments: list[str],
    source_description: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "status": status,
        "job_name": job_name,
        "endpoint_id": endpoint_config.endpoint_id,
        "task_type": endpoint_config.task_type,
        "source_dataset": endpoint_config.tdc_name,
        "processing_mode": effective["processing_mode"],
        "region": effective["region"],
        "role_arn": sanitize_role_arn(effective["role_arn"]),
        "image_uri": effective["image_uri"],
        "instance_settings": {
            "instance_type": effective["instance_type"],
            "instance_count": int(effective["instance_count"]),
            "volume_size": int(effective["volume_size"]),
        },
        "inputs": redact_mapping(inputs),
        "outputs": redact_mapping(outputs),
        "container_arguments": arguments,
        "source_package": source_description,
        "output_s3_prefix": effective["output_s3_prefix"],
        "tags": redact_mapping({"tags": effective.get("tags", [])})["tags"],
        "maximum_runtime": int(effective["max_runtime"]),
        "development_mode": effective.get("development_row_limit") is not None,
        "kms_key_arn": effective.get("kms_key_arn"),
        "network": {
            "vpc_subnets": effective.get("vpc_subnets", []),
            "vpc_security_group_ids": effective.get("vpc_security_group_ids", []),
        },
        "effective_configuration": redact_mapping(_public_effective_config(effective)),
        "warnings": warnings,
    }


def generate_processing_job_name(
    prefix: str,
    endpoint_id: str,
    *,
    now: datetime | None = None,
    suffix: str | None = None,
) -> str:
    return generate_job_name(prefix, endpoint_id, "processing", now=now, suffix=suffix)


def build_warnings(config: dict[str, Any]) -> list[str]:
    warnings = ["Dry run validates configuration only and does not check S3 object existence."]
    if config.get("development_row_limit") is not None:
        warnings.append("development_row_limit is set; this is development-only and not a full dataset preparation.")
    return warnings


def parse_key_value(items: list[str] | None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE item, got {item!r}.")
        key, value = item.split("=", 1)
        parsed[key] = value
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch or dry-run a SageMaker Processing ADMET prep job.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--endpoint-config")
    parser.add_argument("--processing-mode", choices=sorted(PROCESSING_MODES))
    parser.add_argument("--source-csv-s3-uri")
    parser.add_argument("--output-s3-prefix")
    parser.add_argument("--region")
    parser.add_argument("--role-arn")
    parser.add_argument("--image-uri")
    parser.add_argument("--instance-type")
    parser.add_argument("--instance-count", type=int)
    parser.add_argument("--volume-size", type=int)
    parser.add_argument("--max-runtime", type=int)
    parser.add_argument("--job-name-prefix")
    parser.add_argument("--kms-key-arn")
    parser.add_argument("--vpc-subnet", action="append", default=[])
    parser.add_argument("--vpc-security-group", action="append", default=[])
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--development-row-limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--launch-plan-output", default="processing_launch_plan.json")
    parser.add_argument("--wait", dest="wait", action="store_true", default=True)
    parser.add_argument("--no-wait", dest="wait", action="store_false")
    return parser


def cli_overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    tags = [{"Key": key, "Value": value} for key, value in parse_key_value(args.tag).items()]
    overrides = {
        "endpoint_config": args.endpoint_config,
        "processing_mode": args.processing_mode,
        "source_csv_s3_uri": args.source_csv_s3_uri,
        "output_s3_prefix": args.output_s3_prefix,
        "region": args.region,
        "role_arn": args.role_arn,
        "image_uri": args.image_uri,
        "instance_type": args.instance_type,
        "instance_count": args.instance_count,
        "volume_size": args.volume_size,
        "max_runtime": args.max_runtime,
        "job_name_prefix": args.job_name_prefix,
        "kms_key_arn": args.kms_key_arn,
        "vpc_subnets": args.vpc_subnet or None,
        "vpc_security_group_ids": args.vpc_security_group or None,
        "tags": tags or None,
        "development_row_limit": args.development_row_limit,
    }
    return {key: value for key, value in overrides.items() if value is not None}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_processing_launch(
            args.config,
            dry_run=args.dry_run,
            launch_plan_output=args.launch_plan_output,
            wait=args.wait,
            cli_overrides=cli_overrides_from_args(args),
        )
    except Exception as exc:  # noqa: BLE001 - CLI returns nonzero with clear message.
        parser.exit(1, f"SageMaker Processing launch failed: {sanitize_text(str(exc))}\n")
    print(f"{result.status}: {result.job_name}")
    print(f"Manifest: {result.manifest_path}")
    return 0


def _processing_input(*, source: str, destination: str, input_name: str, dry_run: bool) -> Any:
    if dry_run:
        return {"input_name": input_name, "source": source, "destination": destination}
    try:
        from sagemaker.processing import ProcessingInput
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("sagemaker is required for real Processing submission.") from exc
    return ProcessingInput(source=source, destination=destination, input_name=input_name)


def _processing_output(*, source: str, destination: str, output_name: str, dry_run: bool) -> Any:
    if dry_run:
        return {"output_name": output_name, "source": source, "destination": destination}
    try:
        from sagemaker.processing import ProcessingOutput
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("sagemaker is required for real Processing submission.") from exc
    return ProcessingOutput(source=source, destination=destination, output_name=output_name)


def _join_s3(prefix: str, suffix: str) -> str:
    return f"{prefix.rstrip('/')}/{suffix}/"


def _public_effective_config(config: dict[str, Any]) -> dict[str, Any]:
    public_config = dict(config)
    public_config["role_arn"] = sanitize_role_arn(public_config["role_arn"])
    return public_config


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("Expected YAML mapping.")
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
