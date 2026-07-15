data "aws_iam_policy_document" "sagemaker_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "sagemaker_execution" {
  name               = "${local.name_prefix}-sagemaker-execution"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_trust.json
}

data "aws_iam_policy_document" "sagemaker_execution" {
  statement {
    sid    = "ProjectBucketAccess"
    effect = "Allow"
    actions = [
      "s3:AbortMultipartUpload",
      "s3:DeleteObject",
      "s3:GetBucketLocation",
      "s3:GetObject",
      "s3:ListBucket",
      "s3:PutObject",
    ]

    resources = [
      aws_s3_bucket.artifacts.arn,
      "${aws_s3_bucket.artifacts.arn}/*",
    ]
  }

  statement {
    sid    = "ProjectEcrAccess"
    effect = "Allow"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:CompleteLayerUpload",
      "ecr:DescribeImages",
      "ecr:DescribeRepositories",
      "ecr:GetDownloadUrlForLayer",
      "ecr:InitiateLayerUpload",
      "ecr:ListImages",
      "ecr:PutImage",
      "ecr:UploadLayerPart",
    ]

    resources = concat(
      [
        aws_ecr_repository.processing.arn,
        aws_ecr_repository.evaluation.arn,
      ],
      var.enable_training_ecr_repository ? [aws_ecr_repository.training[0].arn] : [],
    )
  }

  statement {
    sid       = "EcrAuthorizationToken"
    effect    = "Allow"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid    = "SageMakerLogs"
    effect = "Allow"
    actions = [
      "logs:CreateLogStream",
      "logs:DescribeLogStreams",
      "logs:PutLogEvents",
    ]

    resources = [
      "${aws_cloudwatch_log_group.sagemaker_training.arn}:*",
      "${aws_cloudwatch_log_group.sagemaker_processing.arn}:*",
    ]
  }

  statement {
    sid    = "SageMakerJobOperations"
    effect = "Allow"
    actions = [
      "sagemaker:DescribeProcessingJob",
      "sagemaker:DescribeTrainingJob",
      "sagemaker:StopProcessingJob",
      "sagemaker:StopTrainingJob",
    ]

    resources = [
      "arn:${data.aws_partition.current.partition}:sagemaker:${var.aws_region}:${data.aws_caller_identity.current.account_id}:processing-job/${local.name_prefix}*",
      "arn:${data.aws_partition.current.partition}:sagemaker:${var.aws_region}:${data.aws_caller_identity.current.account_id}:training-job/${local.name_prefix}*",
    ]
  }

  dynamic "statement" {
    for_each = var.enable_customer_managed_kms_key ? [1] : []

    content {
      sid    = "ProjectKmsUse"
      effect = "Allow"
      actions = [
        "kms:Decrypt",
        "kms:DescribeKey",
        "kms:Encrypt",
        "kms:GenerateDataKey",
        "kms:ReEncryptFrom",
        "kms:ReEncryptTo",
      ]

      resources = [aws_kms_key.project[0].arn]
    }
  }
}

resource "aws_iam_policy" "sagemaker_execution" {
  name        = "${local.name_prefix}-sagemaker-execution"
  description = "Least-privilege SageMaker execution policy for the ADMET platform."
  policy      = data.aws_iam_policy_document.sagemaker_execution.json
}

resource "aws_iam_role_policy_attachment" "sagemaker_execution" {
  role       = aws_iam_role.sagemaker_execution.name
  policy_arn = aws_iam_policy.sagemaker_execution.arn
}
