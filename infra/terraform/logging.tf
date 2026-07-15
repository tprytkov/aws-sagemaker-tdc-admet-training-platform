resource "aws_cloudwatch_log_group" "sagemaker_training" {
  name              = "/aws/sagemaker/TrainingJobs"
  retention_in_days = var.log_retention_days
}

resource "aws_cloudwatch_log_group" "sagemaker_processing" {
  name              = "/aws/sagemaker/ProcessingJobs"
  retention_in_days = var.log_retention_days
}
