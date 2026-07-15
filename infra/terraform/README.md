# Terraform Foundation

This root module defines a low-cost AWS foundation for the SageMaker TDC ADMET training platform. It is intended for local review and planning before any manually approved deployment.

## Resources Created

- One private, versioned S3 artifact bucket for `raw/`, `processed/`, `training/`, `checkpoints/`, `evaluation/`, `models/`, `manifests/`, `source/`, and `temporary/` prefixes.
- Optional customer-managed KMS key with rotation.
- SageMaker execution role trusted only by `sagemaker.amazonaws.com`.
- ECR repositories for dataset Processing and evaluation Processing images.
- Optional custom training ECR repository, disabled by default.
- CloudWatch log retention for SageMaker Training and Processing log groups.
- Optional monthly AWS Budget.

## Resources Deliberately Not Created

This module does not create SageMaker endpoints, notebook instances, Studio domains, VPCs, subnets, NAT gateways, VPC endpoints, EC2 instances, RDS databases, dashboards, alarms, datasets, model artifacts, or container images.

## Expected Cost Profile

The default configuration is intended to be low cost when idle. Costs can still occur for S3 storage and versions, ECR image storage, CloudWatch logs, KMS keys if enabled, and AWS Budgets. Budget email subscriptions may require confirmation.

## Required Local Tools

- Terraform CLI 1.6 or newer, below 2.0
- AWS CLI or another supported AWS authentication method

## AWS Authentication Options

Use an AWS profile, environment variables, AWS SSO, or your normal local credential chain. Do not commit credentials, `.tfvars`, Terraform state, or plans.

## Commands

```bash
terraform -chdir=infra/terraform init
terraform -chdir=infra/terraform fmt
terraform -chdir=infra/terraform validate
terraform -chdir=infra/terraform plan -var-file=local.tfvars
```

Do not run `terraform apply` without manual review and approval.

For local validation without configuring a backend:

```bash
terraform -chdir=infra/terraform init -backend=false
terraform -chdir=infra/terraform validate
```

## Configuration Mapping

Terraform outputs map into local launcher YAML files:

- `aws_region` -> `aws.region`
- `artifact_bucket_name` -> S3 paths under `s3://<bucket>/...`
- `sagemaker_execution_role_arn` -> `aws.role_arn`
- `processing_ecr_repository_url` -> dataset Processing `image_uri`
- `evaluation_ecr_repository_url` -> evaluation Processing `image_uri`
- `training_ecr_repository_url` -> optional custom training image URI
- `kms_key_arn` -> `security.kms_key_arn` when customer-managed KMS is enabled

Use `scripts/render_aws_configs.py` with `terraform output -json` content to generate local override files without invoking Terraform or AWS.

## Destroying Resources

After a demonstration, run `terraform destroy` only after reviewing the plan. Versioned S3 objects and delete markers can prevent bucket deletion; remove object versions manually if necessary.

## Cost-Safety Warnings

Keep SageMaker endpoints, notebook instances, Studio domains, NAT gateways, and long-running jobs out of this foundation until there is a clear need and an explicit cost review.
