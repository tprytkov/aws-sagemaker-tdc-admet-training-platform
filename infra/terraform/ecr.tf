resource "aws_ecr_repository" "processing" {
  name                 = local.ecr_repository_names.processing
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = var.enable_customer_managed_kms_key ? "KMS" : "AES256"
    kms_key         = var.enable_customer_managed_kms_key ? aws_kms_key.project[0].arn : null
  }
}

resource "aws_ecr_repository" "evaluation" {
  name                 = local.ecr_repository_names.evaluation
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = var.enable_customer_managed_kms_key ? "KMS" : "AES256"
    kms_key         = var.enable_customer_managed_kms_key ? aws_kms_key.project[0].arn : null
  }
}

resource "aws_ecr_repository" "training" {
  count = var.enable_training_ecr_repository ? 1 : 0

  name                 = local.ecr_repository_names.training
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = var.enable_customer_managed_kms_key ? "KMS" : "AES256"
    kms_key         = var.enable_customer_managed_kms_key ? aws_kms_key.project[0].arn : null
  }
}

resource "aws_ecr_lifecycle_policy" "processing" {
  repository = aws_ecr_repository.processing.name
  policy     = local.ecr_lifecycle_policy
}

resource "aws_ecr_lifecycle_policy" "evaluation" {
  repository = aws_ecr_repository.evaluation.name
  policy     = local.ecr_lifecycle_policy
}

resource "aws_ecr_lifecycle_policy" "training" {
  count = var.enable_training_ecr_repository ? 1 : 0

  repository = aws_ecr_repository.training[0].name
  policy     = local.ecr_lifecycle_policy
}

locals {
  ecr_lifecycle_policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Expire old untagged images"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 14
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}
