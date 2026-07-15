"""Render local SageMaker execution YAML files from Terraform output JSON.

This script makes no AWS calls and does not invoke Terraform. It expects the
JSON shape produced by `terraform output -json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - requirements include PyYAML.
    yaml = None  # type: ignore[assignment]


REQUIRED_OUTPUTS = {
    "aws_region",
    "artifact_bucket_name",
    "sagemaker_execution_role_arn",
    "processing_ecr_repository_url",
    "evaluation_ecr_repository_url",
    "project_s3_prefixes",
}


def load_terraform_outputs(path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Terraform output JSON must contain an object.")
    outputs = {key: value.get("value") if isinstance(value, dict) and "value" in value else value for key, value in raw.items()}
    missing = sorted(REQUIRED_OUTPUTS - set(outputs))
    if missing:
        raise ValueError(f"Missing required Terraform output(s): {', '.join(missing)}.")
    return outputs


def render_configs(
    terraform_outputs_json: str | Path,
    output_dir: str | Path,
    *,
    endpoint_id: str = "bbb_martins",
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    outputs = load_terraform_outputs(terraform_outputs_json)
    rendered = build_rendered_configs(outputs, endpoint_id)
    output_path = Path(output_dir)
    targets = {
        "processing": output_path / "generated_sagemaker_processing.yaml",
        "training": output_path / "generated_sagemaker_training.yaml",
        "evaluation": output_path / "generated_sagemaker_evaluation.yaml",
    }
    existing = [str(path) for path in targets.values() if path.exists()]
    manifest_path = output_path / "aws_config_generation_manifest.json"
    if existing and not force and not dry_run:
        raise FileExistsError(
            "Generated config file(s) already exist; pass --force to overwrite: "
            + ", ".join(existing)
        )

    manifest = {
        "status": "dry_run" if dry_run else "written",
        "endpoint_id": endpoint_id,
        "generated_files": {name: str(path) for name, path in targets.items()},
        "terraform_outputs_used": sorted(REQUIRED_OUTPUTS | {"kms_key_arn", "training_ecr_repository_url"}),
        "created_at": datetime.now(UTC).isoformat(),
        "redacted_preview": redact(rendered),
    }
    if dry_run:
        return manifest

    output_path.mkdir(parents=True, exist_ok=True)
    if yaml is None:
        raise RuntimeError("PyYAML is required to render AWS config YAML files.")
    for name, path in targets.items():
        path.write_text(yaml.safe_dump(rendered[name], sort_keys=False), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def build_rendered_configs(outputs: dict[str, Any], endpoint_id: str) -> dict[str, dict[str, Any]]:
    bucket = outputs["artifact_bucket_name"]
    region = outputs["aws_region"]
    role = outputs["sagemaker_execution_role_arn"]
    processing_image = outputs["processing_ecr_repository_url"]
    evaluation_image = outputs["evaluation_ecr_repository_url"]
    training_image = outputs.get("training_ecr_repository_url")
    kms_key = outputs.get("kms_key_arn")

    processing = {
        "endpoint_config": f"configs/{endpoint_id}.yaml",
        "processing_mode": "tdc_download",
        "image_uri": f"{processing_image}:latest",
        "development_row_limit": None,
        "aws": {"region": region, "role_arn": role},
        "s3": {
            "source_csv": None,
            "output_prefix": f"s3://{bucket}/processed/{endpoint_id}/",
        },
        "compute": {"instance_type": "ml.m5.xlarge", "instance_count": 1, "volume_size": 50, "max_runtime": 1800},
        "job": {"name_prefix": "admet-processing"},
        "source": {"source_dir": "sagemaker", "entry_point": "prepare_tdc_dataset.py"},
        "tags": [{"Key": "Project", "Value": "ADMET"}, {"Key": "JobType", "Value": "Processing"}],
        "security": {"kms_key_arn": kms_key, "vpc_subnets": [], "vpc_security_group_ids": []},
    }
    training = {
        "endpoint_config": f"configs/{endpoint_id}.yaml",
        "model_type": "chemberta",
        "aws": {"region": region, "role_arn": role},
        "s3": {
            "train": f"s3://{bucket}/processed/{endpoint_id}/train/train.csv",
            "validation": f"s3://{bucket}/processed/{endpoint_id}/validation/valid.csv",
            "test": f"s3://{bucket}/processed/{endpoint_id}/test/test.csv",
            "output": f"s3://{bucket}/training/{endpoint_id}/",
            "checkpoint": f"s3://{bucket}/checkpoints/{endpoint_id}/",
        },
        "compute": {"instance_type": "ml.g5.xlarge", "instance_count": 1, "volume_size": 100, "max_runtime": 3600},
        "job": {"name_prefix": "admet"},
        "source": {"source_dir": "sagemaker", "entry_point": "train_chemberta.py"},
        "image": (
            {"strategy": "custom", "image_uri": f"{training_image}:latest"}
            if training_image
            else {"strategy": "managed", "transformers_version": "4.37", "pytorch_version": "2.1", "py_version": "py310"}
        ),
        "input_mode": "File",
        "hyperparameters": {
            "model_name": "seyonec/ChemBERTa-zinc-base-v1",
            "max_sequence_length": 128,
            "learning_rate": 0.00002,
            "epochs": 3,
            "train_batch_size": 8,
            "evaluation_batch_size": 16,
            "weight_decay": 0.01,
            "early_stopping_patience": 2,
            "random_seed": 42,
            "development_row_limit": None,
            "local_files_only": False,
            "cache_dir": None,
        },
        "tags": [{"Key": "Project", "Value": "ADMET"}, {"Key": "ModelType", "Value": "ChemBERTa"}],
        "security": {"kms_key_arn": kms_key, "enable_network_isolation": False, "vpc_subnets": [], "vpc_security_group_ids": []},
    }
    evaluation = {
        "image_uri": f"{evaluation_image}:latest",
        "aws": {"region": region, "role_arn": role},
        "s3": {
            "candidate_runs": f"s3://{bucket}/training/{endpoint_id}/",
            "evaluation_config": None,
            "output_prefix": f"s3://{bucket}/evaluation/{endpoint_id}/",
        },
        "compute": {"instance_type": "ml.m5.xlarge", "instance_count": 1, "volume_size": 30, "max_runtime": 1800},
        "job": {"name_prefix": "admet-evaluation"},
        "source": {"source_dir": "sagemaker", "entry_point": "evaluate_models.py"},
        "evaluation": {
            "endpoint_id": endpoint_id,
            "near_tie_tolerance": 0.01,
            "primary_metric_override": None,
            "include_development_runs": False,
            "registry_schema_version": "1.0.0",
            "local_config": None,
        },
        "tags": [{"Key": "Project", "Value": "ADMET"}, {"Key": "JobType", "Value": "Evaluation"}],
        "security": {"kms_key_arn": kms_key, "vpc_subnets": [], "vpc_security_group_ids": []},
    }
    return {"processing": processing, "training": training, "evaluation": evaluation}


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str) and value.startswith("arn:aws:iam::"):
        parts = value.split(":")
        if len(parts) > 4:
            parts[4] = "************"
            return ":".join(parts)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render local SageMaker YAML configs from Terraform outputs.")
    parser.add_argument("--terraform-outputs-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--endpoint-id", default="bbb_martins")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = render_configs(
            args.terraform_outputs_json,
            args.output_dir,
            endpoint_id=args.endpoint_id,
            force=args.force,
            dry_run=args.dry_run,
        )
    except Exception as exc:  # noqa: BLE001 - CLI returns nonzero with clear error.
        parser.exit(1, f"AWS config rendering failed: {exc}\n")
    print(json.dumps(redact(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
