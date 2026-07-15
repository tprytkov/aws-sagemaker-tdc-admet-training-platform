resource "random_id" "bucket_suffix" {
  byte_length = 4
}

data "aws_caller_identity" "current" {}

data "aws_partition" "current" {}

locals {
  name_prefix = var.resource_prefix != null ? var.resource_prefix : "${var.project_name}-${var.environment}"

  artifact_bucket_name = var.artifact_bucket_name != null ? var.artifact_bucket_name : "${local.name_prefix}-${random_id.bucket_suffix.hex}"

  common_tags = merge(
    {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
      Repository  = "aws-sagemaker-tdc-admet-training-platform"
    },
    var.common_tags,
  )

  s3_prefixes = {
    raw         = "raw/"
    processed   = "processed/"
    training    = "training/"
    checkpoints = "checkpoints/"
    evaluation  = "evaluation/"
    models      = "models/"
    manifests   = "manifests/"
    source      = "source/"
    temporary   = "temporary/"
  }

  ecr_repository_names = {
    processing = "${local.name_prefix}-processing"
    evaluation = "${local.name_prefix}-evaluation"
    training   = "${local.name_prefix}-training"
  }
}
