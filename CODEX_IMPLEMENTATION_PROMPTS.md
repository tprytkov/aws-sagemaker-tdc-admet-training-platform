# Codex Implementation Prompts — AWS SageMaker TDC ADMET Training Platform

Use these prompts in order inside Codex for the local repository:

```text
C:\aws-sagemaker-tdc-admet-training-platform
```

Do not ask Codex to push to GitHub or run AWS commands until you explicitly approve it.

---

## Prompt 0 — Inspect Repository State

```text
You are working in the local repository C:\aws-sagemaker-tdc-admet-training-platform.

First, inspect the current repository state. Show:
1. current working directory,
2. git status,
3. top-level file tree,
4. whether README.md, PROJECT_GOALS.md, requirements.txt, .gitignore, src/, tests/, configs/, docs/, terraform/, scripts/, and model_registry/ already exist.

Do not modify files yet. After inspection, propose the smallest safe first implementation step.
```

---

## Prompt 1 — Create Repository Scaffold

```text
Create the initial repository scaffold for an AWS SageMaker TDC ADMET training platform.

Required structure:
- README.md
- PROJECT_GOALS.md if missing
- AGENTS.md
- requirements.txt
- requirements-dev.txt
- .gitignore
- configs/
  - bbb_martins.yaml
  - caco2_wang.yaml
  - herg_karim.yaml
- src/
  - data_processing/
  - training/
  - evaluation/
  - inference/
  - registry/
- scripts/
- terraform/
- docs/
  - architecture.md
  - security_design.md
  - cost_control.md
  - model_cards/
  - evaluation_reports/
- model_registry/
- tests/

Keep all files public-safe. Do not include AWS credentials, private data, or large model files. Add minimal placeholder files only where needed to preserve folders. Use clean Python packaging conventions. After changes, show git diff summary and explain what was added.
```

---

## Prompt 2 — Write or Improve PROJECT_GOALS.md

```text
Create or update PROJECT_GOALS.md for this repository.

The project goal is to build an enterprise-style AWS SageMaker ML training platform for fine-tuning and evaluating ChemBERTa-family models on public TDC ADMET datasets. The platform should produce trained model artifacts, metrics, model cards, and model_registry JSON entries that MolOptima can later consume.

Include:
- one-sentence project goal,
- target employability value,
- first ADMET endpoints: BBB_Martins, Caco2_Wang, hERG_Karim,
- AWS services demonstrated: S3, SageMaker Processing, SageMaker Training, IAM, CloudWatch, Terraform, later Step Functions,
- public-safety rules,
- repository structure,
- success criteria,
- scientific limitations and disclaimer wording.

Keep formatting professional and recruiter-readable.
```

---

## Prompt 3 — Add Configuration Schema

```text
Implement endpoint configuration files and a Python config loader.

Create:
- configs/bbb_martins.yaml
- configs/caco2_wang.yaml
- configs/herg_karim.yaml
- src/registry/config.py or src/config.py
- tests/test_config_schema.py

Each YAML should include:
- endpoint_name
- tdc_group: ADME or Tox
- tdc_dataset_name
- task_type: binary_classification or regression
- smiles_column
- label_column
- split_strategy: scaffold by default where supported
- base_model: DeepChem/ChemBERTa-77M-MLM
- model_family: chemberta
- metrics list
- s3_prefix placeholders
- local_output_dir

The config loader should validate required fields and fail with clear errors. Add tests for missing fields and valid configs. Do not call AWS yet.
```

---

## Prompt 4 — Implement TDC Data Preparation Script

```text
Implement a local-first TDC data preparation module.

Create:
- src/data_processing/prepare_tdc_dataset.py
- scripts/prepare_tdc_dataset.py
- tests/test_prepare_tdc_dataset_schema.py

Requirements:
1. Load endpoint config from YAML.
2. Use TDC single-prediction dataset APIs when available.
3. Create train/valid/test CSV files with standardized columns:
   - molecule_id
   - smiles
   - label
   - split
   - endpoint
4. Include a --sample-size option for quick local testing.
5. Include a --dry-run option that validates config and prints planned outputs.
6. Make the script robust if TDC is not installed: show a clear install message instead of crashing obscurely.
7. Do not upload to S3 in this first step.
8. Add tests using a tiny synthetic dataframe so tests do not depend on internet access.

After implementation, run pytest if possible and report results.
```

---

## Prompt 5 — Add S3 Upload Script

```text
Add a controlled S3 upload script for prepared endpoint datasets.

Create:
- scripts/upload_prepared_data_to_s3.py
- src/data_processing/s3_upload.py
- tests/test_s3_upload_dry_run.py

Requirements:
1. Accept local prepared-data directory.
2. Accept S3 bucket and prefix from CLI arguments or config.
3. Support --dry-run as the default recommended mode.
4. Never upload if bucket is missing.
5. Print exact planned S3 paths.
6. Use boto3 only when not dry-run.
7. Do not include credentials in code.
8. Add .gitignore rules for local outputs.

This script should support the professional workflow: GitHub stores code; S3 stores training data and artifacts.
```

---

## Prompt 6 — Implement Baseline Model Training

```text
Implement a baseline model training module before ChemBERTa fine-tuning.

Create:
- src/training/train_baseline.py
- scripts/train_baseline.py
- src/evaluation/metrics.py
- tests/test_metrics_classification.py
- tests/test_metrics_regression.py

Baseline requirements:
1. For binary classification, train a simple scikit-learn baseline using Morgan fingerprints if RDKit is available, otherwise a simple fallback featurizer for tests.
2. For regression, train a simple RandomForestRegressor or similar baseline.
3. Save metrics.json.
4. Save predictions.csv.
5. Metrics:
   - classification: AUROC when possible, AUPRC when possible, F1, balanced accuracy, accuracy
   - regression: MAE, RMSE, R2, Spearman if scipy is available
6. Keep tests independent of RDKit by using small synthetic feature arrays.

Do not implement SageMaker yet. First make local baseline evaluation reproducible.
```

---

## Prompt 7 — Implement ChemBERTa Training Script for SageMaker Script Mode

```text
Implement a ChemBERTa training script suitable for SageMaker script mode and local smoke testing.

Create:
- src/training/train_chemberta.py
- scripts/train_chemberta_local_smoke.py
- tests/test_training_argument_parser.py

Requirements:
1. Accept train, validation, and test CSV paths.
2. Accept task_type: binary_classification or regression.
3. Accept model_id, output_dir, epochs, batch_size, learning_rate, max_length.
4. Use Hugging Face transformers AutoTokenizer and AutoModelForSequenceClassification.
5. Use SMILES as text input.
6. Save model artifacts to output_dir.
7. Write metrics.json and predictions.csv.
8. Detect SageMaker environment variables if present, including SM_MODEL_DIR and SM_CHANNEL_TRAIN/VALIDATION/TEST.
9. Keep local smoke test small and optional; tests should not download model weights.
10. Add clear error messages if transformers or torch are missing.

Do not run a full model download unless explicitly requested.
```

---

## Prompt 8 — Implement SageMaker Training Launcher

```text
Create a SageMaker training launcher for ChemBERTa ADMET fine-tuning.

Create:
- scripts/launch_sagemaker_training.py
- src/training/sagemaker_launcher.py
- docs/sagemaker_training.md

Requirements:
1. Use SageMaker Python SDK HuggingFace estimator or PyTorch estimator, whichever is more appropriate and available.
2. Accept endpoint config YAML.
3. Accept AWS region, role ARN, S3 train/validation/test paths, output S3 path, instance type, and job name prefix.
4. Default to a modest instance type placeholder and require the user to confirm real GPU instance selection in docs.
5. Support --dry-run that prints the estimator/job configuration without launching.
6. Never hard-code AWS credentials.
7. Add documentation explaining that the user should run this from AWS CloudShell, SageMaker Studio, or a configured local environment.

Do not call .fit() unless --execute is explicitly passed.
```

---

## Prompt 9 — Implement Evaluation Report and Model Card Generation

```text
Add automated generation of evaluation reports and model cards.

Create:
- src/evaluation/reporting.py
- scripts/generate_evaluation_report.py
- scripts/generate_model_card.py
- docs/model_cards/model_card_template.md
- docs/evaluation_reports/evaluation_report_template.md
- tests/test_report_generation.py

Requirements:
1. Read metrics.json and endpoint config.
2. Generate Markdown evaluation report.
3. Generate Markdown model card.
4. Include dataset, endpoint, task type, model ID, training date placeholder, metrics, baseline comparison, limitations, and public-safety disclaimer.
5. Use careful wording: computational screening signal only, not clinical/safety validation.
6. Keep generated reports lightweight and commit-safe.
```

---

## Prompt 10 — Implement Model Registry JSON

```text
Implement model registry JSON generation for MolOptima integration.

Create:
- src/registry/model_registry.py
- scripts/generate_model_registry_entry.py
- model_registry/example_bbb_martins_chemberta_v1.json
- tests/test_model_registry_schema.py

Each registry entry should include:
- model_id
- endpoint_name
- task_type
- base_model
- model_family
- training_dataset
- training_repo
- artifact_location
- s3_artifact_uri placeholder
- local_cache_path placeholder
- input_schema
- output_schema
- metrics
- validation_status
- intended_use
- limitations
- moloptima_enabled

The schema should be strict enough that MolOptima can read it later. Do not put real AWS account IDs in committed examples.
```

---

## Prompt 11 — Add Terraform Skeleton

```text
Add a Terraform skeleton for the AWS ADMET training platform.

Create:
- terraform/providers.tf
- terraform/variables.tf
- terraform/main.tf
- terraform/outputs.tf
- terraform/modules/s3/
- terraform/modules/iam/
- terraform/modules/cloudwatch/
- terraform/modules/sagemaker/
- docs/terraform_deployment.md

Requirements:
1. S3 bucket module for data and artifacts.
2. IAM role module for SageMaker execution role with least-privilege notes.
3. CloudWatch log group naming convention.
4. Variables for project_name, environment, aws_region, bucket_prefix.
5. No hard-coded account IDs.
6. No secrets.
7. Documentation must instruct the user to run terraform plan before apply.

Do not run terraform apply. Only write code and docs.
```

---

## Prompt 12 — Add Step Functions Design, Not Full Deployment Yet

```text
Add a Step Functions workflow design document and optional ASL template for the future prepare → train → evaluate → register workflow.

Create:
- terraform/modules/step_functions/
- docs/step_functions_workflow.md
- docs/architecture.md update

The workflow should include conceptual states:
1. Prepare TDC dataset with SageMaker Processing.
2. Train ChemBERTa with SageMaker Training.
3. Evaluate model with SageMaker Processing.
4. Generate registry metadata.
5. Store output under S3 artifacts path.

Use placeholders where exact ARNs are unknown. Do not deploy the state machine yet unless explicitly requested.
```

---

## Prompt 13 — Add GitHub Actions CI

```text
Add a GitHub Actions CI workflow for code quality and tests.

Create:
- .github/workflows/ci.yml

Requirements:
1. Run on pull_request and push to main.
2. Set up Python 3.11.
3. Install requirements-dev.txt.
4. Run pytest.
5. Do not require AWS credentials.
6. Do not download large Hugging Face models.
7. Make tests local and fast.

Update README with a CI badge placeholder only if repository URL is known.
```

---

## Prompt 14 — README Polish for Recruiters and Engineers

```text
Rewrite README.md so it is clear to both recruiters and technical reviewers.

Include:
1. Project summary.
2. Architecture diagram in text form.
3. What is implemented vs planned.
4. Local quickstart.
5. AWS execution path.
6. SageMaker training workflow.
7. S3 layout.
8. Endpoint configs.
9. Testing.
10. Public-safety and scientific limitations.
11. MolOptima integration plan.
12. Employability keywords, but phrased naturally.

Keep it truthful. Do not claim completed AWS training until scripts and outputs exist.
```

---

## Prompt 15 — Final Review Before First GitHub Push

```text
Review the repository before first GitHub push.

Check:
1. No AWS credentials or secrets.
2. No .env files committed.
3. No large model files.
4. No private data.
5. README is accurate.
6. PROJECT_GOALS.md exists.
7. Tests pass or failing tests are clearly reported.
8. .gitignore covers outputs, models, cache, and credentials.
9. Git status is clean except intended files.

Then produce a concise summary suitable for a GitHub commit message and a longer summary suitable for a LinkedIn/GitHub project update.
Do not push unless I explicitly ask.
```
