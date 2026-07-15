data "aws_iam_policy_document" "project_kms" {
  count = var.enable_customer_managed_kms_key ? 1 : 0

  statement {
    sid     = "AllowAccountRootAdministration"
    effect  = "Allow"
    actions = ["kms:*"]

    principals {
      type        = "AWS"
      identifiers = ["arn:${data.aws_partition.current.partition}:iam::${data.aws_caller_identity.current.account_id}:root"]
    }

    resources = ["*"]
  }

  statement {
    sid    = "AllowSageMakerRoleUse"
    effect = "Allow"
    actions = [
      "kms:Decrypt",
      "kms:DescribeKey",
      "kms:Encrypt",
      "kms:GenerateDataKey",
      "kms:ReEncryptFrom",
      "kms:ReEncryptTo",
    ]

    principals {
      type        = "AWS"
      identifiers = [aws_iam_role.sagemaker_execution.arn]
    }

    resources = ["*"]
  }
}

resource "aws_kms_key" "project" {
  count = var.enable_customer_managed_kms_key ? 1 : 0

  description             = "KMS key for ${local.name_prefix} ADMET platform artifacts"
  deletion_window_in_days = 30
  enable_key_rotation     = true
  policy                  = data.aws_iam_policy_document.project_kms[0].json
}

resource "aws_kms_alias" "project" {
  count = var.enable_customer_managed_kms_key ? 1 : 0

  name          = "alias/${var.kms_key_alias != null ? var.kms_key_alias : local.name_prefix}"
  target_key_id = aws_kms_key.project[0].key_id
}
