# AWS SageMaker TDC ADMET Training Platform — Project Goals

## Project Name

`aws-sagemaker-tdc-admet-training-platform`

## Working Directory

Local Windows path:

```text
C:\aws-sagemaker-tdc-admet-training-platform
```

## One-Sentence Goal

Build an enterprise-style AWS machine-learning training platform for fine-tuning and evaluating molecular property prediction models on public Therapeutics Data Commons (TDC) ADMET datasets, then expose trained model metadata for downstream use in MolOptima.

## Why This Project Exists

This project is designed to demonstrate practical AWS AI/ML platform experience, not only model experimentation. It should show that a scientific AI workflow can be organized as a reproducible cloud ML system using public-safe data, infrastructure-as-code, managed AWS training jobs, model evaluation, model registry metadata, and application integration.

The project should support employability for roles such as:

- Senior Applied AI Scientist
- Senior Data Scientist
- Senior ML Scientist
- Scientific AI Engineer
- AWS AI/ML Engineer
- GenAI / RAG / AI Platform Engineer, when combined with other AWS RAG projects

## Core Technical Goals

1. Use public TDC ADMET datasets for endpoint-specific molecular property prediction.
2. Fine-tune ChemBERTa-family models from SMILES strings.
3. Compare fine-tuned transformer models against simple baselines, including RDKit descriptor/rule baselines and classical ML baselines where practical.
4. Use AWS SageMaker Processing jobs for data preparation and evaluation workflows.
5. Use AWS SageMaker Training jobs for model training and fine-tuning.
6. Store input data, processed datasets, trained model artifacts, metrics, and registry metadata in S3.
7. Use CloudWatch logs for operational visibility.
8. Add Terraform infrastructure definitions for the AWS resources needed by the project.
9. Add model-card and evaluation-report templates for each trained endpoint model.
10. Produce model registry JSON files that MolOptima can read later.

## First ADMET Endpoints

Start with three public TDC endpoints:

| Endpoint | Task Type | Reason |
|---|---|---|
| `BBB_Martins` | Binary classification | CNS / BBB permeability; fits MolOptima BBB model direction |
| `Caco2_Wang` | Regression | Absorption / permeability example |
| `hERG_Karim` | Binary classification | Toxicity / cardiotoxicity screening signal |

Additional endpoints can be added later only after the first three have reproducible data preparation, training, evaluation, and registry output.

## Public-Safety Rules

This repository must use only public or synthetic data. Do not commit:

- AWS credentials
- `.env` files containing secrets
- proprietary compound structures
- unpublished lead rankings
- private SAR tables
- internal candidate IDs
- patent-sensitive hypotheses
- trained model binaries if they are large
- raw SageMaker output folders

Commit only lightweight, public-safe artifacts such as source code, small sample files, documentation, metrics summaries, model cards, and registry JSON metadata.

## Intended Architecture

```text
TDC public ADMET dataset
        ↓
SageMaker Processing / local fallback preprocessing
        ↓
S3 train/validation/test CSV layout
        ↓
SageMaker Training job
        ↓
S3 model artifact
        ↓
Evaluation job or local evaluation script
        ↓
metrics.json + evaluation_report.md + model_card.md
        ↓
model_registry/*.json
        ↓
MolOptima reads selected model metadata and loads model if available
```

## Repository Standards

The repository should be organized for technical review:

```text
aws-sagemaker-tdc-admet-training-platform/
  README.md
  PROJECT_GOALS.md
  AGENTS.md
  requirements.txt
  .gitignore
  configs/
  src/
    data_processing/
    training/
    evaluation/
    inference/
    registry/
  scripts/
  terraform/
  docs/
    architecture.md
    security_design.md
    cost_control.md
    model_cards/
    evaluation_reports/
  model_registry/
  tests/
```

## AWS Resources to Demonstrate

Minimum AWS components:

- S3 for raw data, processed data, model artifacts, metrics, and registry files
- SageMaker Processing for dataset preparation and evaluation
- SageMaker Training for ChemBERTa and baseline model training
- IAM execution role with least-privilege design notes
- CloudWatch logs for training and processing jobs

Second phase AWS components:

- Terraform-managed infrastructure
- Step Functions workflow for prepare → train → evaluate → register
- KMS encryption notes or optional module
- Batch Transform or batch prediction pattern

## Model Training Scope

The first ChemBERTa implementation should be modest and reproducible. It does not need to outperform all existing ADMET methods. The goal is to demonstrate:

- correct dataset handling
- clean SMILES input schema
- reproducible splits
- fine-tuning workflow
- metrics reporting
- baseline comparison
- model-card documentation
- safe integration path into MolOptima

## Required Output Artifacts Per Endpoint

For each endpoint, the project should eventually produce:

```text
model_registry/<endpoint>_chemberta_v1.json
docs/evaluation_reports/<endpoint>_chemberta_v1.md
docs/model_cards/<endpoint>_chemberta_v1.md
outputs/examples/<endpoint>_metrics_example.json
```

The actual trained model artifact should live in S3 or another model store, not directly in GitHub.

## MolOptima Integration Goal

MolOptima should not train these models. MolOptima should later consume registry metadata and optionally load local model artifacts.

MolOptima should be able to:

1. List available trained ADMET models.
2. Display endpoint, task type, base model, training dataset, metrics, validation status, and limitations.
3. Allow a user to select one or more ADMET models for molecular prioritization.
4. Add model prediction columns to the prioritization output.
5. Preserve fallback behavior if a trained model is unavailable.

## Scientific and Ethical Framing

All model outputs are computational screening signals only. They are not experimental ADMET evidence, toxicity evidence, safety validation, clinical evidence, regulatory conclusions, or legal conclusions.

Use wording such as:

- endpoint-specific computational ADMET screening signal
- model-based triage output
- public benchmark evaluation
- experimental model artifact

Avoid wording such as:

- validated toxicity predictor
- clinical safety model
- drug-likeness decision engine
- patentability or freedom-to-operate conclusion

## Success Criteria

The first successful version of this project should include:

- A clean repository scaffold
- Public-safe documentation
- A working local data preparation script for at least one TDC endpoint
- A training script that can run locally on a small sample and is structured for SageMaker
- A SageMaker launch script template
- A metrics JSON output schema
- A model registry JSON schema
- Unit tests for schema and data validation
- Clear README instructions
- No secrets or large model binaries committed

The second successful version should include:

- Real SageMaker run for one endpoint
- S3 artifact path recorded in registry JSON
- Evaluation report and model card
- Terraform skeleton or minimal deployable Terraform
- MolOptima registry integration plan or example registry entry
