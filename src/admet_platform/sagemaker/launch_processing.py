"""SageMaker Processing launcher for ADMET TDC dataset preparation."""

from __future__ import annotations

import argparse
import math
import re
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
    manifest_inputs = build_processing_inputs(effective, dry_run=True)
    manifest_outputs = build_processing_outputs(effective, dry_run=True)
    run_inputs = manifest_inputs if dry_run else build_processing_inputs(effective, dry_run=False)
    run_outputs = manifest_outputs if dry_run else build_processing_outputs(effective, dry_run=False)
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
        inputs=manifest_inputs,
        outputs=manifest_outputs,
        arguments=arguments,
        source_description=source_description,
        warnings=warnings,
    )
    output_path = Path(launch_plan_output or effective.get("launch_plan_output") or "processing_launch_plan.json")

    if dry_run:
        write_manifest_json(output_path, plan)
        return ProcessingLaunchResult("dry_run", job_name, output_path, plan)

    code_path = build_processing_code_path(effective)
    validate_processing_code_path(code_path, project_root=PROJECT_ROOT)
    processor = None
    try:
        processor_args = build_processor_args(effective=effective)
        session = create_sagemaker_session(effective["region"], sagemaker_session_factory)
        processor_args["sagemaker_session"] = session
        processor = create_processor(processor_args, processor_class)
        run_kwargs = build_processor_run_kwargs(
            code_path=code_path,
            inputs=run_inputs,
            outputs=run_outputs,
            arguments=arguments,
            job_name=job_name,
            wait=wait,
            kms_key_arn=effective.get("kms_key_arn"),
        )
        processor.run(**run_kwargs)
        result_manifest = build_submission_result_manifest(plan, processor=processor, wait=wait)
        write_manifest_json(output_path, result_manifest)
        return ProcessingLaunchResult("submitted", job_name, output_path, result_manifest)
    except Exception as exc:  # noqa: BLE001 - CLI must return nonzero.
        failure = build_failure_result_manifest(plan, processor=processor, wait=wait, error=exc)
        write_manifest_json(output_path, failure)
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
        _processing_input(**input_spec, dry_run=dry_run)
        for input_spec in build_processing_input_specs(config)
    ]
    return inputs


def build_processing_input_specs(config: dict[str, Any]) -> list[dict[str, str]]:
    inputs = [
        {
            "input_name": "config",
            "source": config["endpoint_config"],
            "destination": "/opt/ml/processing/input/config",
        }
    ]
    if config["processing_mode"] == "supplied_csv":
        inputs.append(
            {
                "input_name": "data",
                "source": config["source_csv_s3_uri"],
                "destination": "/opt/ml/processing/input/data",
            }
        )
    return inputs


def build_processing_outputs(config: dict[str, Any], *, dry_run: bool) -> list[Any]:
    return [
        _processing_output(**output_spec, dry_run=dry_run)
        for output_spec in build_processing_output_specs(config)
    ]


def build_processing_output_specs(config: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "output_name": "train",
            "source": "/opt/ml/processing/output/train",
            "destination": _join_s3(config["output_s3_prefix"], "train"),
        },
        {
            "output_name": "validation",
            "source": "/opt/ml/processing/output/validation",
            "destination": _join_s3(config["output_s3_prefix"], "validation"),
        },
        {
            "output_name": "test",
            "source": "/opt/ml/processing/output/test",
            "destination": _join_s3(config["output_s3_prefix"], "test"),
        },
        {
            "output_name": "metadata",
            "source": "/opt/ml/processing/output/metadata",
            "destination": _join_s3(config["output_s3_prefix"], "metadata"),
        },
    ]


def build_processor_args(*, effective: dict[str, Any]) -> dict[str, Any]:
    args: dict[str, Any] = {
        "role": effective["role_arn"],
        "image_uri": effective["image_uri"],
        "command": ["python"],
        "instance_type": effective["instance_type"],
        "instance_count": int(effective["instance_count"]),
        "volume_size_in_gb": int(effective["volume_size"]),
        "max_runtime_in_seconds": int(effective["max_runtime"]),
        "base_job_name": effective["job_name_prefix"],
        "tags": effective.get("tags", []),
    }
    if effective.get("kms_key_arn"):
        args["output_kms_key"] = effective["kms_key_arn"]
    if effective.get("vpc_subnets") or effective.get("vpc_security_group_ids"):
        args["network_config"] = build_network_config(effective)
    return args


def build_processor_run_kwargs(
    *,
    code_path: str,
    inputs: list[Any],
    outputs: list[Any],
    arguments: list[str],
    job_name: str,
    wait: bool,
    kms_key_arn: str | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "code": code_path,
        "inputs": inputs,
        "outputs": outputs,
        "arguments": arguments,
        "wait": wait,
        "logs": True,
        "job_name": job_name,
    }
    if kms_key_arn:
        kwargs["kms_key"] = kms_key_arn
    return kwargs


def build_processing_code_path(config: dict[str, Any]) -> str:
    return (Path(config["source_dir"]) / config["entry_point"]).as_posix()


def validate_processing_code_path(code_path: str, *, project_root: Path) -> None:
    local_path = (project_root / code_path).resolve()
    if not local_path.exists() or not local_path.is_file():
        raise ValueError(f"Configured SageMaker Processing code file does not exist: {code_path}.")


def build_network_config(effective: dict[str, Any]) -> Any:
    try:
        from sagemaker.network import NetworkConfig
    except ModuleNotFoundError as exc:  # pragma: no cover - dry-run avoids this path.
        raise RuntimeError("sagemaker is required to configure Processing network settings.") from exc
    return NetworkConfig(
        subnets=effective.get("vpc_subnets") or None,
        security_group_ids=effective.get("vpc_security_group_ids") or None,
    )


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
            from sagemaker.processing import ScriptProcessor
        except ModuleNotFoundError as exc:  # pragma: no cover - dry-run avoids this path.
            raise RuntimeError("sagemaker is required for real Processing submission.") from exc
        processor_class = ScriptProcessor
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


def build_submission_result_manifest(plan: dict[str, Any], *, processor: Any, wait: bool) -> dict[str, Any]:
    job_description = describe_processing_job(processor)
    result = dict(plan)
    result["status"] = "submitted"
    result["submitted"] = True
    result["wait"] = wait
    result["processing_job_arn"] = sanitize_optional_arn(job_description.get("ProcessingJobArn"))
    if wait:
        result["final_status"] = job_description.get("ProcessingJobStatus")
        failure_reason = job_description.get("FailureReason")
        if failure_reason:
            result["failure_reason"] = sanitize_text(str(failure_reason))
    return result


def build_failure_result_manifest(
    plan: dict[str, Any],
    *,
    processor: Any | None,
    wait: bool,
    error: Exception,
) -> dict[str, Any]:
    job_description = describe_processing_job(processor)
    result = dict(plan)
    result["status"] = "failed"
    result["submitted"] = False
    result["wait"] = wait
    result["processing_job_arn"] = sanitize_optional_arn(job_description.get("ProcessingJobArn"))
    result["final_status"] = job_description.get("ProcessingJobStatus")
    result["failure_reason"] = sanitize_text(str(job_description.get("FailureReason") or error))
    result["error"] = sanitize_text(str(error))
    return result


def describe_processing_job(processor: Any | None) -> dict[str, Any]:
    if processor is None:
        return {}
    latest_job = getattr(processor, "latest_job", None)
    if latest_job is None:
        return {}
    describe = getattr(latest_job, "describe", None)
    if not callable(describe):
        return {}
    try:
        description = describe()
    except Exception:  # noqa: BLE001 - best effort metadata only.
        return {}
    return description if isinstance(description, dict) else {}


def sanitize_optional_arn(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return re.sub(r"(?<=:)\d{12}(?=:)", "************", sanitize_text(str(value)))


def write_manifest_json(path: str | Path, payload: dict[str, Any]) -> None:
    validate_json_safe(payload)
    write_json(path, payload)


def validate_json_safe(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, str | bool | int):
        return
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise TypeError(f"Manifest value at {path} must be a finite JSON number.")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            validate_json_safe(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"Manifest key at {path} must be a string.")
            validate_json_safe(item, f"{path}.{key}")
        return
    raise TypeError(f"Manifest value at {path} is not JSON serializable: {type(value).__name__}.")


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
