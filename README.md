# AWS SageMaker TDC ADMET Training Platform

This project is a public-safe scaffold for an AWS SageMaker training platform for Therapeutics Data Commons (TDC) ADMET models. The goal is to organize data preparation, model training, evaluation, model registry metadata, and infrastructure definitions into a reproducible ML platform structure.

The first planned endpoints are:

- `BBB_Martins`
- `Caco2_Wang`
- `hERG_Karim`

Planned AWS services include S3, SageMaker Processing, SageMaker Training, CloudWatch, IAM, Step Functions, and Terraform-managed infrastructure.

MolOptima integration is planned through lightweight model registry JSON files in `model_registry/`. These files will describe trained model metadata and evaluation summaries without storing private molecules or large model artifacts in the repository.

Public-safe scope:

- No private molecules or proprietary compound data.
- No AWS credentials, account IDs, secrets, or private S3 bucket names.
- No clinical, medical, or safety claims.
- No large datasets or trained model binaries committed to Git.

## Project Status

Initial scaffold only. Full SageMaker data processing, training, evaluation, inference, and Terraform logic will be added incrementally.
