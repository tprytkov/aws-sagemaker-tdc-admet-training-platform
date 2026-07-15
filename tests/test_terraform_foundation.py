import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TERRAFORM_DIR = PROJECT_ROOT / "infra" / "terraform"
REQUIRED_TF_FILES = {
    "versions.tf",
    "providers.tf",
    "variables.tf",
    "locals.tf",
    "s3.tf",
    "iam.tf",
    "ecr.tf",
    "logging.tf",
    "budget.tf",
    "outputs.tf",
    "terraform.tfvars.example",
    "README.md",
}
RENDERER = PROJECT_ROOT / "scripts" / "render_aws_configs.py"


def test_terraform_directory_and_required_file_presence() -> None:
    assert TERRAFORM_DIR.exists()
    assert REQUIRED_TF_FILES <= {path.name for path in TERRAFORM_DIR.iterdir()}


def test_versions_and_aws_provider_declarations() -> None:
    versions = _read("versions.tf")
    assert 'required_version = ">= 1.6, < 2.0"' in versions
    assert 'source  = "hashicorp/aws"' in versions
    assert 'version = ">= 5.50, < 6.0"' in versions


def test_s3_security_versioning_encryption_https_and_lifecycle() -> None:
    s3 = _read("s3.tf")
    assert "aws_s3_bucket_public_access_block" in s3
    assert "block_public_acls       = true" in s3
    assert "restrict_public_buckets = true" in s3
    assert "BucketOwnerEnforced" in s3
    assert "aws_s3_bucket_versioning" in s3
    assert 'status = "Enabled"' in s3
    assert "aws_s3_bucket_server_side_encryption_configuration" in s3
    assert '"aws:SecureTransport"' in s3
    assert '"false"' in s3
    assert "aws_s3_bucket_lifecycle_configuration" in s3
    assert 'prefix = "temporary/"' in s3
    assert 'prefix = "checkpoints/"' in s3


def test_terraform_state_ignore_rules() -> None:
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("**/.terraform/", "*.tfstate", "*.tfstate.*", "crash.log", "*.tfvars", "!terraform.tfvars.example", "*.tfplan", "override.tf"):
        assert pattern in gitignore


def test_iam_trust_permissions_and_absences() -> None:
    iam = _read("iam.tf")
    assert "sagemaker.amazonaws.com" in iam
    assert "AdministratorAccess" not in iam
    assert '"iam:*"' not in iam
    assert '"kms:*"' not in iam
    assert '"s3:*"' not in iam
    assert "aws_s3_bucket.artifacts.arn" in iam
    assert '"${aws_s3_bucket.artifacts.arn}/*"' in iam
    assert "aws_ecr_repository.processing.arn" in iam
    assert "aws_ecr_repository.evaluation.arn" in iam


def test_ecr_scan_lifecycle_and_optional_training_repo() -> None:
    ecr = _read("ecr.tf")
    assert "scan_on_push = true" in ecr
    assert 'image_tag_mutability = "IMMUTABLE"' in ecr
    assert "aws_ecr_lifecycle_policy" in ecr
    assert 'tagStatus   = "untagged"' in ecr
    assert "var.enable_training_ecr_repository ? 1 : 0" in ecr


def test_optional_kms_budget_and_excluded_expensive_resources() -> None:
    all_tf = _all_tf()
    assert "var.enable_customer_managed_kms_key ? 1 : 0" in all_tf
    assert "enable_key_rotation     = true" in all_tf
    assert "aws_budgets_budget" in all_tf
    assert "var.enable_budget ? 1 : 0" in all_tf
    forbidden = [
        "aws_sagemaker_endpoint",
        "aws_sagemaker_notebook_instance",
        "aws_sagemaker_domain",
        "aws_nat_gateway",
        "aws_vpc",
        "aws_db_instance",
    ]
    for resource_name in forbidden:
        assert resource_name not in all_tf


def test_expected_outputs_and_example_tfvars_are_public_safe() -> None:
    outputs = _read("outputs.tf")
    for name in (
        "aws_region",
        "artifact_bucket_name",
        "artifact_bucket_arn",
        "sagemaker_execution_role_arn",
        "processing_ecr_repository_url",
        "evaluation_ecr_repository_url",
        "training_ecr_repository_url",
        "kms_key_arn",
        "project_s3_prefixes",
    ):
        assert f'output "{name}"' in outputs
    example = _read("terraform.tfvars.example")
    assert "AWS_SECRET_ACCESS_KEY" not in example
    assert "aws_access_key_id" not in example.lower()
    assert "example-not-real@example.com" in example


def test_terraform_output_json_parsing_and_missing_output_validation(tmp_path: Path) -> None:
    renderer = _load_renderer()
    output_json = _write_outputs(tmp_path)
    parsed = renderer.load_terraform_outputs(output_json)
    assert parsed["artifact_bucket_name"] == "example-admet-bucket"

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"aws_region": {"value": "us-west-2"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="Missing required Terraform output"):
        renderer.load_terraform_outputs(bad)


def test_processing_training_and_evaluation_config_rendering(tmp_path: Path) -> None:
    renderer = _load_renderer()
    manifest = renderer.render_configs(_write_outputs(tmp_path), tmp_path / "generated", endpoint_id="bbb_martins")
    assert manifest["status"] == "written"

    processing = yaml.safe_load((tmp_path / "generated" / "generated_sagemaker_processing.yaml").read_text(encoding="utf-8"))
    training = yaml.safe_load((tmp_path / "generated" / "generated_sagemaker_training.yaml").read_text(encoding="utf-8"))
    evaluation = yaml.safe_load((tmp_path / "generated" / "generated_sagemaker_evaluation.yaml").read_text(encoding="utf-8"))

    assert processing["image_uri"] == "111111111111.dkr.ecr.us-west-2.amazonaws.com/admet-processing:latest"
    assert processing["s3"]["output_prefix"] == "s3://example-admet-bucket/processed/bbb_martins/"
    assert training["aws"]["role_arn"].endswith("role/admet-demo-sagemaker-execution")
    assert training["image"]["strategy"] == "managed"
    assert evaluation["image_uri"] == "111111111111.dkr.ecr.us-west-2.amazonaws.com/admet-evaluation:latest"
    assert evaluation["s3"]["output_prefix"] == "s3://example-admet-bucket/evaluation/bbb_martins/"
    generation_manifest = json.loads((tmp_path / "generated" / "aws_config_generation_manifest.json").read_text(encoding="utf-8"))
    assert {"status", "endpoint_id", "generated_files", "terraform_outputs_used", "created_at", "redacted_preview"} <= set(generation_manifest)


def test_overwrite_protection_force_and_dry_run(tmp_path: Path) -> None:
    renderer = _load_renderer()
    outputs = _write_outputs(tmp_path)
    out_dir = tmp_path / "generated"
    renderer.render_configs(outputs, out_dir)
    with pytest.raises(FileExistsError):
        renderer.render_configs(outputs, out_dir)
    dry = renderer.render_configs(outputs, out_dir, dry_run=True)
    assert dry["status"] == "dry_run"
    forced = renderer.render_configs(outputs, out_dir, force=True)
    assert forced["status"] == "written"


def test_renderer_cli_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "cli"
    result = subprocess.run(
        [
            sys.executable,
            str(RENDERER),
            "--terraform-outputs-json",
            str(_write_outputs(tmp_path)),
            "--output-dir",
            str(out_dir),
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"status": "dry_run"' in result.stdout
    assert "arn:aws:iam::************:role/" in result.stdout
    assert not out_dir.exists()


def _read(name: str) -> str:
    return (TERRAFORM_DIR / name).read_text(encoding="utf-8")


def _all_tf() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in TERRAFORM_DIR.glob("*.tf"))


def _load_renderer():
    spec = importlib.util.spec_from_file_location("render_aws_configs", RENDERER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_outputs(tmp_path: Path) -> Path:
    payload = {
        "aws_region": {"value": "us-west-2"},
        "artifact_bucket_name": {"value": "example-admet-bucket"},
        "artifact_bucket_arn": {"value": "arn:aws:s3:::example-admet-bucket"},
        "sagemaker_execution_role_arn": {"value": "arn:aws:iam::111111111111:role/admet-demo-sagemaker-execution"},
        "processing_ecr_repository_url": {"value": "111111111111.dkr.ecr.us-west-2.amazonaws.com/admet-processing"},
        "evaluation_ecr_repository_url": {"value": "111111111111.dkr.ecr.us-west-2.amazonaws.com/admet-evaluation"},
        "training_ecr_repository_url": {"value": None},
        "kms_key_arn": {"value": None},
        "project_s3_prefixes": {"value": {"raw": "raw/", "processed": "processed/"}},
    }
    path = tmp_path / "terraform_outputs.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
