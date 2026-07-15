"""SageMaker Processing launcher for model evaluation jobs."""

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

from admet_platform.models.artifacts import write_json
from admet_platform.sagemaker.launch_processing import (
    _join_s3,
    _mapping,
    _processing_input,
    _processing_output,
    create_processor,
    create_sagemaker_session,
)
from admet_platform.sagemaker.launch_training import (
    _public_path,
    generate_job_name,
    redact_mapping,
    sanitize_role_arn,
    sanitize_text,
    validate_s3_uri,
    validate_tags,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "sagemaker_evaluation.yaml"


@dataclass(frozen=True)
class EvaluationLaunchResult:
    """Result from a dry-run or submitted SageMaker evaluation Processing job."""

    status: str
    job_name: str
    manifest_path: Path
    manifest: dict[str, Any]


class SageMakerEvaluationSubmissionError(RuntimeError):
    """Raised when real SageMaker evaluation submission fails."""


def load_evaluation_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if yaml is None:
        raise RuntimeError("PyYAML is required to load SageMaker evaluation configuration.")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"SageMaker evaluation config {config_path} must contain a YAML mapping.")
    return payload


def run_evaluation_launch(
    evaluation_config_path: str | Path,
    *,
    dry_run: bool,
    launch_plan_output: str | Path | None = None,
    wait: bool = True,
    cli_overrides: dict[str, Any] | None = None,
    now: datetime | None = None,
    suffix: str | None = None,
    sagemaker_session_factory: Any | None = None,
    processor_class: Any | None = None,
) -> EvaluationLaunchResult:
    raw = load_evaluation_yaml(evaluation_config_path)
    effective = build_effective_config(raw, cli_overrides or {})
    validate_effective_config(effective)
    endpoint_component = effective.get("endpoint_id") or "multi"
    job_name = generate_job_name(effective["job_name_prefix"], endpoint_component, "evaluation", now=now, suffix=suffix)
    inputs = build_processing_inputs(effective, dry_run=dry_run)
    outputs = build_processing_outputs(effective, dry_run=dry_run)
    arguments = build_container_arguments(effective)
    source_description = validate_source_package_inputs(
        effective["source_dir"],
        effective["entry_point"],
        effective.get("local_evaluation_config"),
        project_root=PROJECT_ROOT,
    )
    warnings = build_warnings(effective)
    plan = build_launch_plan(
        status="dry_run" if dry_run else "submitted",
        job_name=job_name,
        effective=effective,
        inputs=inputs,
        outputs=outputs,
        arguments=arguments,
        source_description=source_description,
        warnings=warnings,
    )
    output_path = Path(launch_plan_output or effective.get("launch_plan_output") or "evaluation_launch_plan.json")

    if dry_run:
        write_json(output_path, plan)
        return EvaluationLaunchResult("dry_run", job_name, output_path, plan)

    try:
        package_dir = prepare_source_package(
            source_dir=effective["source_dir"],
            entry_point=effective["entry_point"],
            local_evaluation_config=effective.get("local_evaluation_config"),
            job_name=job_name,
            project_root=PROJECT_ROOT,
        )
        processor_args = build_processor_args(effective=effective, source_dir=package_dir)
        session = create_sagemaker_session(effective["region"], sagemaker_session_factory)
        processor_args["sagemaker_session"] = session
        processor = create_processor(processor_args, processor_class)
        processor.run(inputs=inputs, outputs=outputs, arguments=arguments, job_name=job_name, wait=wait)
        plan["status"] = "submitted"
        plan["wait"] = wait
        write_json(output_path, plan)
        return EvaluationLaunchResult("submitted", job_name, output_path, plan)
    except Exception as exc:  # noqa: BLE001 - CLI converts to nonzero.
        failure = dict(plan)
        failure["status"] = "failed"
        failure["error"] = sanitize_text(str(exc))
        write_json(output_path, failure)
        raise SageMakerEvaluationSubmissionError(sanitize_text(str(exc))) from exc


def build_effective_config(raw: dict[str, Any], cli_overrides: dict[str, Any]) -> dict[str, Any]:
    aws = _mapping(raw.get("aws"))
    s3 = _mapping(raw.get("s3"))
    compute = _mapping(raw.get("compute"))
    job = _mapping(raw.get("job"))
    source = _mapping(raw.get("source"))
    security = _mapping(raw.get("security"))
    evaluation = _mapping(raw.get("evaluation"))
    effective: dict[str, Any] = {
        "candidate_runs_s3_uri": s3.get("candidate_runs"),
        "evaluation_config_s3_uri": s3.get("evaluation_config"),
        "local_evaluation_config": evaluation.get("local_config"),
        "output_s3_prefix": s3.get("output_prefix"),
        "region": aws.get("region"),
        "role_arn": aws.get("role_arn"),
        "image_uri": raw.get("image_uri"),
        "instance_type": compute.get("instance_type", "ml.m5.xlarge"),
        "instance_count": compute.get("instance_count", 1),
        "volume_size": compute.get("volume_size", 30),
        "max_runtime": compute.get("max_runtime", 1800),
        "job_name_prefix": job.get("name_prefix", "admet-evaluation"),
        "source_dir": source.get("source_dir", "sagemaker"),
        "entry_point": source.get("entry_point", "evaluate_models.py"),
        "endpoint_id": evaluation.get("endpoint_id"),
        "near_tie_tolerance": evaluation.get("near_tie_tolerance", 0.01),
        "primary_metric_override": evaluation.get("primary_metric_override"),
        "include_development_runs": evaluation.get("include_development_runs", False),
        "registry_schema_version": evaluation.get("registry_schema_version", "1.0.0"),
        "tags": raw.get("tags", []),
        "kms_key_arn": security.get("kms_key_arn"),
        "vpc_subnets": security.get("vpc_subnets", []),
        "vpc_security_group_ids": security.get("vpc_security_group_ids", []),
        "launch_plan_output": raw.get("launch_plan_output"),
    }
    for key, value in cli_overrides.items():
        if value is not None:
            effective[key] = value
    if effective.get("local_evaluation_config"):
        effective["local_evaluation_config"] = _public_path(
            (PROJECT_ROOT / effective["local_evaluation_config"]).resolve(),
            PROJECT_ROOT,
        )
    return effective


def validate_effective_config(config: dict[str, Any]) -> None:
    required = {
        "candidate_runs_s3_uri": "candidate-runs S3 URI",
        "output_s3_prefix": "output S3 prefix",
        "region": "AWS region",
        "role_arn": "SageMaker execution role ARN",
        "image_uri": "Processing image URI",
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
        raise ValueError(f"Missing required SageMaker evaluation field(s): {', '.join(missing)}.")
    validate_s3_uri(config["candidate_runs_s3_uri"], "candidate_runs_s3_uri")
    validate_s3_uri(config["output_s3_prefix"], "output_s3_prefix")
    if config.get("evaluation_config_s3_uri"):
        validate_s3_uri(config["evaluation_config_s3_uri"], "evaluation_config_s3_uri")
    if not str(config["role_arn"]).startswith("arn:aws:iam::"):
        raise ValueError("role_arn must be a syntactically valid IAM role ARN.")
    if int(config["instance_count"]) < 1 or int(config["volume_size"]) < 1 or int(config["max_runtime"]) < 1:
        raise ValueError("instance_count, volume_size, and max_runtime must be positive.")
    validate_tags(config.get("tags", []))
    for key in ("vpc_subnets", "vpc_security_group_ids"):
        if not isinstance(config.get(key, []), list):
            raise ValueError(f"{key} must be a list.")


def build_container_arguments(config: dict[str, Any]) -> list[str]:
    args = [
        "--runs-dir",
        "/opt/ml/processing/input/runs",
        "--output-dir",
        "/opt/ml/processing/output",
    ]
    if config.get("evaluation_config_s3_uri") or config.get("local_evaluation_config"):
        args.extend(["--config-dir", "/opt/ml/processing/input/config"])
    if config.get("endpoint_id"):
        args.extend(["--endpoint-id", str(config["endpoint_id"])])
    if config.get("near_tie_tolerance") is not None:
        args.extend(["--near-tie-tolerance", str(config["near_tie_tolerance"])])
    if config.get("primary_metric_override"):
        args.extend(["--primary-metric", str(config["primary_metric_override"])])
    if config.get("include_development_runs"):
        args.append("--include-development-runs")
    if config.get("registry_schema_version"):
        args.extend(["--registry-schema-version", str(config["registry_schema_version"])])
    return args


def build_processing_inputs(config: dict[str, Any], *, dry_run: bool) -> list[Any]:
    inputs = [
        _processing_input(
            source=config["candidate_runs_s3_uri"],
            destination="/opt/ml/processing/input/runs",
            input_name="candidate-runs",
            dry_run=dry_run,
        )
    ]
    config_source = config.get("evaluation_config_s3_uri") or config.get("local_evaluation_config")
    if config_source:
        inputs.append(
            _processing_input(
                source=config_source,
                destination="/opt/ml/processing/input/config",
                input_name="evaluation-config",
                dry_run=dry_run,
            )
        )
    return inputs


def build_processing_outputs(config: dict[str, Any], *, dry_run: bool) -> list[Any]:
    return [
        _processing_output(
            source="/opt/ml/processing/output/evaluation",
            destination=_join_s3(config["output_s3_prefix"], "evaluation"),
            output_name="evaluation",
            dry_run=dry_run,
        ),
        _processing_output(
            source="/opt/ml/processing/output/model_card",
            destination=_join_s3(config["output_s3_prefix"], "model_card"),
            output_name="model-card",
            dry_run=dry_run,
        ),
        _processing_output(
            source="/opt/ml/processing/output/registry",
            destination=_join_s3(config["output_s3_prefix"], "registry"),
            output_name="registry",
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


def validate_source_package_inputs(
    source_dir: str | Path,
    entry_point: str,
    local_evaluation_config: str | Path | None,
    *,
    project_root: Path,
) -> dict[str, Any]:
    source_path = (project_root / source_dir).resolve()
    entry_path = source_path / entry_point
    package_path = (project_root / "src" / "admet_platform").resolve()
    requirements_path = source_path / "evaluation_requirements.txt"
    paths = [source_path, entry_path, package_path, requirements_path]
    if local_evaluation_config:
        paths.append((project_root / local_evaluation_config).resolve())
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise ValueError(f"Evaluation source package input(s) are missing: {', '.join(missing)}.")
    includes = [
        _public_path(entry_path, project_root),
        _public_path(package_path, project_root),
        _public_path(requirements_path, project_root),
    ]
    if local_evaluation_config:
        includes.append(_public_path((project_root / local_evaluation_config).resolve(), project_root))
    return {
        "strategy": "copy_evaluation_wrapper_package_and_requirements",
        "source_dir": _public_path(source_path, project_root),
        "entry_point": entry_point,
        "includes": includes,
        "entry_point_at_package_root": True,
    }


def prepare_source_package(
    *,
    source_dir: str | Path,
    entry_point: str,
    local_evaluation_config: str | Path | None,
    job_name: str,
    project_root: Path,
) -> Path:
    package_root = Path(tempfile.mkdtemp(prefix=f"{job_name}-evaluation-source-"))
    source_path = (project_root / source_dir).resolve()
    shutil.copy2(source_path / entry_point, package_root / entry_point)
    shutil.copytree(project_root / "src" / "admet_platform", package_root / "src" / "admet_platform")
    shutil.copy2(source_path / "evaluation_requirements.txt", package_root / "evaluation_requirements.txt")
    if local_evaluation_config:
        destination = package_root / "evaluation_config.yaml"
        shutil.copy2(project_root / local_evaluation_config, destination)
    return package_root


def build_launch_plan(
    *,
    status: str,
    job_name: str,
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
        "endpoint_id": effective.get("endpoint_id"),
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
        "evaluation_settings": {
            "near_tie_tolerance": effective["near_tie_tolerance"],
            "primary_metric_override": effective.get("primary_metric_override"),
            "include_development_runs": effective["include_development_runs"],
            "registry_schema_version": effective["registry_schema_version"],
        },
        "source_package": source_description,
        "output_s3_prefix": effective["output_s3_prefix"],
        "tags": redact_mapping({"tags": effective.get("tags", [])})["tags"],
        "maximum_runtime": int(effective["max_runtime"]),
        "warnings": warnings,
    }


def build_warnings(config: dict[str, Any]) -> list[str]:
    warnings = [
        "Dry run validates configuration only and does not check remote S3 object existence.",
        "Evaluation recommendations use validation metrics only; test metrics remain descriptive.",
    ]
    if config.get("include_development_runs"):
        warnings.append("Development/smoke runs are explicitly included in eligibility.")
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
    parser = argparse.ArgumentParser(description="Launch or dry-run a SageMaker model-evaluation Processing job.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--candidate-runs-s3-uri")
    parser.add_argument("--evaluation-config-s3-uri")
    parser.add_argument("--local-evaluation-config")
    parser.add_argument("--output-s3-prefix")
    parser.add_argument("--region")
    parser.add_argument("--role-arn")
    parser.add_argument("--image-uri")
    parser.add_argument("--instance-type")
    parser.add_argument("--instance-count", type=int)
    parser.add_argument("--volume-size", type=int)
    parser.add_argument("--max-runtime", type=int)
    parser.add_argument("--job-name-prefix")
    parser.add_argument("--endpoint-id")
    parser.add_argument("--near-tie-tolerance", type=float)
    parser.add_argument("--primary-metric")
    parser.add_argument("--include-development-runs", action="store_true")
    parser.add_argument("--registry-schema-version")
    parser.add_argument("--kms-key-arn")
    parser.add_argument("--vpc-subnet", action="append", default=[])
    parser.add_argument("--vpc-security-group", action="append", default=[])
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--launch-plan-output", default="evaluation_launch_plan.json")
    parser.add_argument("--wait", dest="wait", action="store_true", default=True)
    parser.add_argument("--no-wait", dest="wait", action="store_false")
    return parser


def cli_overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    tags = [{"Key": key, "Value": value} for key, value in parse_key_value(args.tag).items()]
    overrides = {
        "candidate_runs_s3_uri": args.candidate_runs_s3_uri,
        "evaluation_config_s3_uri": args.evaluation_config_s3_uri,
        "local_evaluation_config": args.local_evaluation_config,
        "output_s3_prefix": args.output_s3_prefix,
        "region": args.region,
        "role_arn": args.role_arn,
        "image_uri": args.image_uri,
        "instance_type": args.instance_type,
        "instance_count": args.instance_count,
        "volume_size": args.volume_size,
        "max_runtime": args.max_runtime,
        "job_name_prefix": args.job_name_prefix,
        "endpoint_id": args.endpoint_id,
        "near_tie_tolerance": args.near_tie_tolerance,
        "primary_metric_override": args.primary_metric,
        "include_development_runs": True if args.include_development_runs else None,
        "registry_schema_version": args.registry_schema_version,
        "kms_key_arn": args.kms_key_arn,
        "vpc_subnets": args.vpc_subnet or None,
        "vpc_security_group_ids": args.vpc_security_group or None,
        "tags": tags or None,
    }
    return {key: value for key, value in overrides.items() if value is not None}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_evaluation_launch(
            args.config,
            dry_run=args.dry_run,
            launch_plan_output=args.launch_plan_output,
            wait=args.wait,
            cli_overrides=cli_overrides_from_args(args),
        )
    except Exception as exc:  # noqa: BLE001
        parser.exit(1, f"SageMaker evaluation launch failed: {sanitize_text(str(exc))}\n")
    print(f"{result.status}: {result.job_name}")
    print(f"Manifest: {result.manifest_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
