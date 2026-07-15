output "aws_region" {
  description = "AWS region used by this Terraform deployment."
  value       = var.aws_region
}

output "artifact_bucket_name" {
  description = "Private S3 bucket for ADMET platform artifacts."
  value       = aws_s3_bucket.artifacts.bucket
}

output "artifact_bucket_arn" {
  description = "ARN of the ADMET artifact bucket."
  value       = aws_s3_bucket.artifacts.arn
}

output "sagemaker_execution_role_arn" {
  description = "SageMaker execution role ARN."
  value       = aws_iam_role.sagemaker_execution.arn
}

output "processing_ecr_repository_url" {
  description = "ECR repository URL for dataset Processing image."
  value       = aws_ecr_repository.processing.repository_url
}

output "evaluation_ecr_repository_url" {
  description = "ECR repository URL for evaluation Processing image."
  value       = aws_ecr_repository.evaluation.repository_url
}

output "training_ecr_repository_url" {
  description = "Optional ECR repository URL for custom training image."
  value       = var.enable_training_ecr_repository ? aws_ecr_repository.training[0].repository_url : null
}

output "kms_key_arn" {
  description = "Customer-managed KMS key ARN when enabled."
  value       = var.enable_customer_managed_kms_key ? aws_kms_key.project[0].arn : null
}

output "project_s3_prefixes" {
  description = "Expected S3 prefix layout for project artifacts."
  value       = local.s3_prefixes
}
