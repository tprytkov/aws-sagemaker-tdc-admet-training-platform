variable "aws_region" {
  description = "AWS region for all resources."
  type        = string
  default     = "us-west-2"
}

variable "aws_profile" {
  description = "Optional local AWS CLI profile for terraform plan/apply."
  type        = string
  default     = null
}

variable "project_name" {
  description = "Short project name used in resource naming and tags."
  type        = string
  default     = "admet-platform"
}

variable "environment" {
  description = "Deployment environment name, such as dev or demo."
  type        = string
  default     = "dev"
}

variable "resource_prefix" {
  description = "Optional resource-name prefix. Defaults to project-environment."
  type        = string
  default     = null
}

variable "artifact_bucket_name" {
  description = "Globally unique S3 bucket name. If null, Terraform creates a name using random suffix."
  type        = string
  default     = null
}

variable "common_tags" {
  description = "Additional tags applied to all supported resources."
  type        = map(string)
  default     = {}
}

variable "enable_customer_managed_kms_key" {
  description = "Create and use a customer-managed KMS key for S3 and ECR encryption."
  type        = bool
  default     = false
}

variable "kms_key_alias" {
  description = "Optional KMS alias name without alias/ prefix."
  type        = string
  default     = null
}

variable "log_retention_days" {
  description = "CloudWatch log retention for SageMaker Training and Processing logs."
  type        = number
  default     = 30
}

variable "enable_training_ecr_repository" {
  description = "Create an optional custom training ECR repository. Disabled by default because managed Hugging Face images are supported."
  type        = bool
  default     = false
}

variable "enable_budget" {
  description = "Create an optional AWS monthly cost budget."
  type        = bool
  default     = false
}

variable "monthly_budget_amount" {
  description = "Monthly budget limit."
  type        = string
  default     = "25"
}

variable "budget_currency" {
  description = "Budget currency."
  type        = string
  default     = "USD"
}

variable "budget_alert_percentages" {
  description = "Budget alert thresholds as percentages of the monthly limit."
  type        = list(number)
  default     = [50, 80, 100]
}

variable "budget_notification_email" {
  description = "Optional email address for budget notifications. Use a real address only in private local tfvars."
  type        = string
  default     = null
}
