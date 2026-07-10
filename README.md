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

## Implemented Configuration Layer

The first endpoint configuration layer defines YAML metadata files for `BBB_Martins`, `Caco2_Wang`, and `hERG_Karim` under `configs/`. The Python loader in `src/admet_platform/config.py` reads these files, validates required fields, validates supported task types, and returns a typed `EndpointConfig` object for downstream pipeline code.

## Implemented Local Data Schema Validation

The local data validation layer checks normalized ADMET CSV files before any AWS, SageMaker, or TDC download workflow is introduced. It validates required columns, non-empty SMILES values, split labels, endpoint-specific target values, and returns a compact dataset summary for downstream processing.

## Implemented Local Dataset Preparation CLI

The local dataset preparation CLI reads a public-safe CSV, loads an endpoint config, validates the schema, normalizes column order, removes exact duplicate rows, writes a cleaned CSV, and writes a compact dataset summary JSON. This mirrors the future SageMaker Processing pattern without using AWS services.

## Implemented Local Baseline Training

The local baseline training layer fits simple scikit-learn models from prepared ADMET CSV files. It uses deterministic character n-gram TF-IDF features from SMILES strings, trains logistic regression for binary endpoints and ridge regression for regression endpoints, then writes a local `joblib` model artifact and metrics JSON.

## Implemented Model Registry Entry Generator

The model registry entry generator creates public-safe JSON metadata for local trained artifacts. It combines endpoint configuration, baseline metrics, artifact references, validation status, input/output schema fields, limitations, and MolOptima integration flags without storing credentials, private data, or large model binaries.

## Implemented Optional TDC Dataset Download Layer

The optional TDC download layer can load public TDC ADMET datasets through PyTDC, apply the configured split strategy, normalize raw TDC columns into the project schema, validate the dataset, and write cleaned CSV plus summary JSON outputs. PyTDC is imported only at runtime, so local tests and non-download workflows do not require it.
