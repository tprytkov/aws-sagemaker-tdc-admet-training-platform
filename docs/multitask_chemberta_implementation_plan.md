# Multi-Task ChemBERTa Implementation Plan

## Purpose

This document defines a new multi-task ChemBERTa research track for the ADMET training platform. It extends, rather than replaces, the existing endpoint-specific TDC preparation, classical benchmark, ChemBERTa training, SageMaker, evaluation, model-card, and registry workflows.

The first multi-task model will share one ChemBERTa encoder across three binary-classification endpoints:

```text
                           +-- BBB_Martins binary head
SMILES -> shared ChemBERTa +-- hERG_Karim binary head
                           +-- AMES binary head
```

`Caco2_Wang` will be added later as a regression head only after the classification-only model is correct, reproducible, and evaluated against the single-task baselines.

The scientific question is not whether a shared model can be made to run. It is whether shared representation learning improves or preserves endpoint-specific performance while reducing model storage and inference cost. Multi-task results must therefore be compared with the existing descriptor, Morgan-fingerprint, and single-task ChemBERTa results for every endpoint.

## Objectives

1. Implement one platform-independent Python training system shared by local, work-cluster, and SageMaker execution.
2. Use one pretrained ChemBERTa encoder with small endpoint-specific prediction heads.
3. Preserve separate DataLoaders, losses, metrics, thresholds, and provenance for every endpoint.
4. Prevent large datasets from dominating optimization through explicit task sampling.
5. Audit exact-molecule and scaffold leakage within and across endpoints before training.
6. Maintain two clearly labeled evaluation tracks:
   - official endpoint-specific TDC benchmark splits;
   - coordinated leakage-safe multi-task splits.
7. Save resumable checkpoints and standards-compliant artifacts suitable for the existing evaluation, model-card, and registry layers.
8. Validate the implementation locally before using a work GPU cluster, then reproduce a selected run in SageMaker.
9. Keep all data public-safe and keep datasets, model weights, caches, credentials, and account-specific paths out of Git.

## Non-Goals

The first multi-task milestone will not:

- add `Caco2_Wang` regression;
- replace the existing single-task models or official TDC benchmark results;
- claim that multi-task learning is inherently better than Morgan fingerprints or single-task ChemBERTa;
- perform hyperparameter optimization or automated architecture search;
- use multi-node or multi-GPU distributed training;
- launch SageMaker jobs, push Docker images, or create AWS infrastructure;
- deploy an inference endpoint or modify MolOptima;
- merge independently fine-tuned transformer weights;
- use test metrics for checkpoint selection, loss weighting, or architecture decisions;
- commit downloaded TDC datasets, Hugging Face caches, checkpoints, or generated predictions.

## Relationship to the Existing Platform

The multi-task track must reuse current components wherever their contracts remain appropriate:

```text
TDC endpoint configs and loaders
        |
        v
Existing canonicalization and validation
        |
        +--> official single-task prepared splits and benchmarks
        |
        v
New cross-task leakage audit and coordinated split manifest
        |
        v
Shared multi-task ChemBERTa trainer
        |
        v
Existing-style metrics, predictions, model card, and registry metadata
        |
        +--> local execution
        +--> direct work-cluster execution
        +--> thin SageMaker entry point (later phase)
```

Existing modules remain the source of truth for endpoint configuration, SMILES preparation, metric semantics, JSON-safe artifact writing, and scientific warnings. Multi-task code must not duplicate those implementations merely to create a new execution path.

## Endpoint Scope

### Phase 1: classification-only

| Endpoint ID | TDC family | Task | Head | Primary selection metric |
|---|---|---|---|---|
| `bbb_martins` | ADME | Binary classification | Linear output, one logit | Validation ROC-AUC |
| `herg_karim` | Tox | Binary classification | Linear output, one logit | Validation ROC-AUC |
| `ames` | Tox | Binary classification | Linear output, one logit | Validation ROC-AUC |

The project-level endpoint ID and the TDC dataset name must remain separate. The multi-task endpoint retains `herg_karim` as its stable platform identifier and uses the exact TDC dataset name `hERG_Karim`. TDC's smaller `hERG` dataset is distinct and is not part of this multi-task track.

### Later phase: mixed classification and regression

| Endpoint ID | TDC family | Task | Head | Primary selection metric |
|---|---|---|---|---|
| `caco2_wang` | ADME | Regression | Linear output, one value | Validation RMSE |

Adding Caco-2 requires training-only target normalization, inverse transformation for reporting, a regression loss, mixed metric aggregation, and explicit loss-scale controls. None of those concerns should complicate the first classification-only milestone.

## Execution Strategy

### Local Windows development

Local runs are for implementation, unit tests, artifact verification, and tiny CPU smoke training. Use the project Conda environment and Python 3.11. Because the machine may resolve plain `python` to an unrelated interpreter, commands should use an activated environment or an explicit Conda invocation.

Illustrative commands:

```powershell
conda activate admet-platform

python -m pytest -q

python .\scripts\audit_multitask_splits.py `
  --config .\configs\multitask_classification.yaml `
  --output-dir .\outputs\local\multitask\split_audit

python .\scripts\train_multitask.py `
  --config .\configs\multitask_classification.yaml `
  --output-dir .\outputs\local\multitask\smoke `
  --limit-samples-per-task 100 `
  --max-steps 12
```

The local smoke run must prove that every task receives batches, every head participates in training, losses remain finite, evaluation artifacts are written, and checkpoints can be reloaded. It is not a scientific performance run.

### Work GPU cluster without SLURM

Use the same Python entry point directly. Cluster-specific concerns belong in shell commands and documentation, not in the model or trainer.

Start with one explicitly selected GPU:

```bash
conda activate admet-multitask
nvidia-smi

CUDA_VISIBLE_DEVICES=0 python scripts/train_multitask.py \
  --config configs/multitask_classification.yaml \
  --output-dir outputs/multitask/classification_run_001
```

For a persistent session, use `tmux` when permitted:

```bash
tmux new -s admet-multitask
CUDA_VISIBLE_DEVICES=0 python scripts/train_multitask.py \
  --config configs/multitask_classification.yaml \
  --output-dir outputs/multitask/classification_run_001
```

Hugging Face Accelerate may be used after direct single-GPU execution works:

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 \
  scripts/train_multitask.py \
  --config configs/multitask_classification.yaml \
  --output-dir outputs/multitask/classification_run_001
```

The first full experiment should remain single-GPU. Multi-GPU support is a later optimization and must not be a prerequisite for correctness.

Cluster use must comply with organizational policy. Do not copy AWS credentials, private data, unpublished compounds, or machine-specific secrets to the cluster. Confirm that public TDC datasets, pretrained model downloads, and portfolio use are authorized before running.

### SageMaker integration

SageMaker integration begins only after local smoke validation and at least one successful cluster run. The SageMaker entry point must be a thin adapter that:

1. resolves SageMaker input channels and environment paths;
2. loads the same multi-task YAML configuration;
3. calls the shared trainer under `src/admet_platform/`;
4. writes final model artifacts to `SM_MODEL_DIR`;
5. writes metrics, predictions, history, and manifests to `SM_OUTPUT_DIR`;
6. optionally writes resumable checkpoints to `SM_CHECKPOINT_DIR`.

No training behavior may exist only in the SageMaker wrapper. A selected cluster run should be reproducible in SageMaker using the same source commit, data-manifest hashes, model revision, configuration, and random seed.

## Proposed Repository Additions

```text
configs/
  ames.yaml
  multitask_classification.yaml

docs/
  multitask_chemberta_implementation_plan.md

scripts/
  audit_multitask_splits.py
  train_multitask.py
  evaluate_multitask.py                 # optional after core training works

src/admet_platform/
  data/
    multitask.py
    multitask_audit.py
  models/
    multitask_chemberta.py
  training/
    multitask_losses.py
    task_sampler.py
    multitask_trainer.py
  evaluation/
    multitask_metrics.py

tests/
  test_multitask_data.py
  test_multitask_audit.py
  test_multitask_model.py
  test_multitask_losses.py
  test_multitask_sampler.py
  test_multitask_trainer.py
  test_multitask_smoke.py

# Later, after local and cluster validation
sagemaker/
  train_multitask_chemberta.py

src/admet_platform/sagemaker/
  train_multitask_chemberta.py
  launch_multitask_training.py

scripts/
  launch_multitask_training.py
```

The exact file count may be reduced where a small cohesive module is clearer. Existing modules should be extended instead of copied when they already provide the required behavior.

## Configuration Contract

`configs/multitask_classification.yaml` should contain scientific and training configuration, not AWS account details. An illustrative schema is:

```yaml
run:
  name: chemberta_multitask_classification
  random_seed: 42

model:
  model_name: seyonec/ChemBERTa-zinc-base-v1
  model_revision: null
  max_sequence_length: 128
  pooling: masked_mean
  dropout: 0.15

tasks:
  bbb_martins:
    endpoint_config: configs/bbb_martins.yaml
    task_type: binary_classification
    primary_metric: roc_auc
  herg_karim:
    endpoint_config: configs/herg_karim.yaml
    task_type: binary_classification
    primary_metric: roc_auc
  ames:
    endpoint_config: configs/ames.yaml
    task_type: binary_classification
    primary_metric: roc_auc

data:
  prepared_root: outputs/local/multitask/coordinated
  split_track: coordinated_multitask
  enforce_exact_smiles_exclusion: true
  enforce_scaffold_exclusion: true

training:
  epochs: 3
  train_batch_size: 8
  evaluation_batch_size: 16
  encoder_learning_rate: 2.0e-5
  head_learning_rate: 1.0e-4
  weight_decay: 0.01
  gradient_clip_norm: 1.0
  gradient_accumulation_steps: 1
  mixed_precision: "no"
  task_sampling: round_robin
  early_stopping_patience: 2
  checkpoint_metric: mean_validation_roc_auc

artifacts:
  save_predictions: true
  save_optimizer_state: true
  save_scheduler_state: true
```

Model revisions should be resolved and recorded when the Hugging Face API makes the commit available. Optional null values must not be sent as malformed command-line strings.

## Data Contract and Separate Task DataLoaders

Each endpoint retains independent prepared train, validation, and test files using the existing project schema:

```text
molecule_id,smiles,canonical_smiles,target,split
```

The multi-task data layer must not require every compound to have every label. It should create one dataset and one DataLoader per task:

```python
train_loaders = {
    "bbb_martins": bbb_train_loader,
    "herg_karim": herg_train_loader,
    "ames": ames_train_loader,
}
```

Each batch should include tokenized inputs, labels, stable row identity, and task identity:

```python
{
    "input_ids": ...,
    "attention_mask": ...,
    "labels": ...,
    "molecule_id": ...,
    "canonical_smiles": ...,
    "task_name": "bbb_martins",
}
```

Only `canonical_smiles` is tokenized. Identifiers, raw SMILES, endpoint metadata, split labels, and targets must never be added to model features.

Tokenization settings must be shared across tasks. Dataset objects should avoid materializing unnecessary duplicate token arrays when caching or lazy tokenization provides a safe alternative.

## Shared Encoder and Task Heads

The first model should use `AutoModel`, not a single-task `AutoModelForSequenceClassification`, because the platform must attach multiple heads to one encoder.

Conceptually:

```python
class MultiTaskChemBERTa(nn.Module):
    encoder: AutoModel
    heads: nn.ModuleDict
```

Requirements:

- one shared ChemBERTa encoder;
- one linear, single-logit head per classification endpoint;
- masked mean pooling or another explicitly configured pooling strategy;
- deterministic head ordering in configuration and metadata;
- dropout configured once and recorded;
- clear failure for an unknown task name;
- standard checkpoint load/save behavior;
- no silent fallback to a different pretrained checkpoint.

Model output should remain logits during training. Sigmoid probabilities belong in prediction and metric code, not inside the model head when using `BCEWithLogitsLoss`.

## Task Sampling

The first implementation uses deterministic round-robin sampling:

```text
BBB -> hERG -> AMES -> BBB -> hERG -> AMES -> ...
```

Round-robin sampling gives each endpoint an equal number of optimizer steps and prevents the largest endpoint from overwhelming smaller tasks. The sampler must:

- use a stable task order;
- restart an exhausted task iterator without stopping the epoch;
- record how many batches and examples each task contributed;
- support deterministic behavior under a fixed seed;
- work with a bounded `max_steps` smoke mode;
- avoid accidentally evaluating or training on the wrong split.

Later ablations may compare square-root dataset-size sampling or configurable task weights. They must not be introduced before the round-robin baseline is established.

## Losses and Optimization

### Classification-only milestone

Each task uses its own `BCEWithLogitsLoss`. Class weighting is calculated from that endpoint's training labels only and recorded in metadata.

For binary label `1`, the conventional positive weight is:

```text
negative training count / positive training count
```

If a training set has only one class, training must stop with a clear error. No validation or test labels may influence class weights.

Use separate learning-rate groups initially:

- encoder: conservative pretrained-model learning rate;
- heads: higher learning rate suitable for newly initialized linear layers.

Apply gradient clipping, explicit seeds, and JSON-safe logging. The trainer must fail on non-finite loss or gradients rather than continuing silently.

### Caco-2 extension

The later regression phase will add:

- a single-value regression head;
- target mean and standard deviation fitted on Caco-2 training rows only;
- `SmoothL1Loss` or a documented alternative;
- inverse transformation before output and metric calculation;
- safeguards for zero target variance;
- explicit task-loss normalization so regression does not dominate classification.

## Validation, Checkpoint Selection, and Test Isolation

Every endpoint must be evaluated separately. There is no generic multi-task accuracy.

For the classification-only model, report at least:

- ROC-AUC;
- PR-AUC;
- accuracy;
- balanced accuracy;
- precision;
- recall;
- F1;
- Matthews correlation coefficient;
- confusion matrix.

Unavailable metrics must be `null` with a warning, never `NaN` or infinity.

Checkpoint selection uses validation metrics only. The initial composite score is the mean of available per-task validation ROC-AUC values. The selection record must also show every endpoint metric so an average cannot hide severe negative transfer.

Recommended safeguards:

- require all configured tasks to produce a usable validation metric;
- record the worst per-task change versus the relevant single-task reference;
- do not label a checkpoint preferred if one endpoint degrades beyond a configured tolerance without an explicit scientific review;
- evaluate test splits only after the best validation checkpoint is restored;
- never tune thresholds or loss weights on test results.

## Checkpoint and Resume Behavior

Checkpoints must be sufficient to resume training deterministically. Save:

- shared encoder and task-head weights;
- optimizer state;
- scheduler state when used;
- current epoch and global step;
- task-sampler state and next task;
- random states for Python, NumPy, PyTorch CPU, and CUDA when applicable;
- resolved model and tokenizer revisions;
- effective configuration;
- per-task training counts and class weights;
- best validation score and checkpoint path;
- training history to date.

Resume must validate compatibility before loading:

- identical task set and task types;
- compatible head shapes;
- same base checkpoint and tokenizer contract;
- matching coordinated split hashes;
- matching feature/input schema;
- no attempt to resume a full run from a development-limited checkpoint unless explicitly allowed and recorded.

A smoke test must save a checkpoint, instantiate a new trainer, resume, and demonstrate that the global step advances without resetting task history.

## Coordinated Cross-Task Leakage Audit

Multi-task learning creates leakage paths that do not exist in isolated endpoint training. A compound held out for BBB could appear in hERG training, allowing the shared encoder to see the held-out structure.

Before multi-task training, generate a cross-task audit containing:

1. exact canonical-SMILES overlap within each endpoint split;
2. exact canonical-SMILES overlap across task train, validation, and test splits;
3. Bemis-Murcko scaffold overlap within and across tasks;
4. duplicate molecules within every split;
5. duplicate endpoint labels;
6. conflicting labels for the same canonical molecule and endpoint;
7. invalid or missing molecules;
8. rows removed by coordinated exclusions;
9. final per-task class distributions;
10. cryptographic hashes of final split files.

The coordinated multi-task track should ensure that molecules and, when configured, scaffolds held out for validation or test in any target task are excluded from all task training sets. The exclusion policy must be deterministic and recorded.

If global scaffold grouping across the union is used, assign a scaffold group to only one split and preserve all endpoint labels associated with that group. If this changes official TDC splits, label the output as a multi-task research split rather than a TDC benchmark split.

Audit failures are blocking. Training must not silently continue when prohibited cross-task overlap remains.

## Two Evaluation Tracks

### Track A: official endpoint-specific TDC benchmark

Purpose:

- preserve comparability with existing endpoint baselines;
- evaluate descriptor, Morgan, and single-task ChemBERTa models;
- retain the prescribed endpoint split and metric contracts.

These models are trained separately. Results remain the principal endpoint benchmark.

### Track B: coordinated leakage-safe multi-task research

Purpose:

- measure whether a shared encoder helps across tasks;
- prevent one task's training set from exposing another task's held-out molecules or scaffolds;
- evaluate storage and inference trade-offs.

Because coordinated exclusions or global splitting can change the populations, Track B metrics must not be presented as directly interchangeable with Track A metrics. A fair single-task comparator should be retrained on the same coordinated splits when estimating the effect of multi-task learning.

Every report and model card must state the track, split-manifest ID, and comparability limitation.

## Artifact Contract

Each run should write a stable artifact package:

```text
run_output/
  model/
    model_state.pt or model.safetensors
    multitask_model_config.json
  tokenizer/
  checkpoints/
  metrics.json
  training_history.json
  training_metadata.json
  run_manifest.json
  warnings.json
  split_audit.json
  split_manifest.json
  predictions/
    validation/
      bbb_martins.csv
      herg_karim.csv
      ames.csv
    test/
      bbb_martins.csv
      herg_karim.csv
      ames.csv
```

Classification prediction files should include:

- `molecule_id`;
- `canonical_smiles`;
- observed target;
- predicted probability;
- predicted class;
- task name;
- split;
- run ID.

### Required run metadata

The run manifest and training metadata should include:

- run ID and status;
- Git commit when available;
- creation, start, completion, and runtime timestamps;
- execution environment: local, cluster, or SageMaker;
- development-mode flag and row/step limits;
- endpoint configs and exact TDC dataset names;
- official or coordinated evaluation track;
- data file hashes and split-manifest ID;
- base model name and resolved revision;
- tokenizer revision and maximum sequence length;
- model architecture, pooling, dropout, and head definitions;
- all hyperparameters and random seeds;
- task sampling method and per-task contribution counts;
- class weights and label distributions;
- checkpoint-selection rule and best checkpoint;
- package versions and hardware summary;
- warnings and scientific limitations.

Do not include credentials, tokens, full environment dumps, private local paths, or account-specific identifiers in public artifacts. All JSON must contain standard JSON values and no NumPy scalars, `NaN`, or infinity.

## Evaluation and Registry Integration

The existing evaluation layer should be extended only after the core multi-task artifact contract stabilizes. It should create one comparison row per endpoint head and preserve the parent multi-task run ID.

For every endpoint, compare:

```text
RDKit descriptor baseline
Morgan fingerprint baseline
single-task ChemBERTa
multi-task ChemBERTa head
```

Recommendations remain endpoint-specific. The platform must not recommend the entire multi-task artifact for all endpoints merely because its average validation score is highest.

The registry representation may use:

- one parent shared-model record;
- child endpoint-head records referencing the parent artifact;
- endpoint-specific validation/test metrics, thresholds, and approval states;
- a shared tokenizer and encoder artifact reference.

Initial approval status remains `pending_review`. Multi-task models are ADMET research models, not clinical, regulatory, or experimental safety evidence.

## Phased Implementation Plan

### Phase 0: documentation and baseline inventory

- approve this implementation plan;
- inventory current endpoint configs and prepared datasets;
- add a public-safe AMES endpoint configuration;
- record current descriptor, Morgan, and single-task ChemBERTa reference results;
- identify missing single-task results required for a fair comparison.

Exit condition: endpoint and baseline inventory is documented without application-code changes beyond configuration needed by the next phase.

### Phase 1: multi-task data loading and audit

- implement typed multi-task configuration loading;
- reuse existing endpoint preparation schemas;
- implement separate datasets and DataLoaders;
- implement coordinated exact-SMILES and scaffold audits;
- generate deterministic split and exclusion manifests;
- add unit tests using synthetic public-safe molecules.

Exit condition: three endpoint loaders and a blocking leakage audit work without downloading data during unit tests.

### Phase 2: shared model and losses

- implement shared `AutoModel` encoder;
- implement masked pooling and three linear heads;
- implement per-task class weights and binary losses;
- implement deterministic task routing;
- add model save/load tests using mocks or tiny local models without network access.

Exit condition: every head supports forward and backward passes and an unknown task fails clearly.

### Phase 3: trainer, sampling, metrics, and resume

- implement round-robin sampling;
- implement optimizer groups, clipping, evaluation, and checkpoint selection;
- write per-task metrics and predictions;
- implement full checkpoint/resume state;
- add JSON-safe artifact generation.

Exit condition: synthetic training completes, resumes, and selects a checkpoint using validation metrics only.

### Phase 4: local CPU smoke run

- prepare small deterministic subsets for BBB, hERG, and AMES;
- run a few bounded steps locally;
- verify all heads receive updates;
- reload the best checkpoint and evaluate all tasks;
- clearly label all results as development smoke metrics.

Exit condition: the first-milestone acceptance criteria below are satisfied.

### Phase 5: work-cluster single-GPU validation

- stage only public data and pinned pretrained assets;
- run a short full-data cluster smoke test;
- measure GPU memory, throughput, and checkpoint size;
- verify interruption and resume behavior;
- run the selected full configuration on one GPU;
- repeat with additional seeds only after one run is correct.

Exit condition: a reproducible full run produces per-task metrics, predictions, and a complete manifest.

### Phase 6: controlled scientific comparison

- train single-task comparators on the coordinated multi-task splits;
- compare multi-task heads against same-split single-task and classical baselines;
- quantify per-task gains or negative transfer;
- add bootstrap confidence intervals where practical;
- document storage, latency, and memory trade-offs.

Exit condition: the project can state whether sharing helps each endpoint without relying on test-set selection.

### Phase 7: SageMaker reproduction

- add the thin SageMaker container entry point;
- add local `/opt/ml` contract simulation;
- add dry-run launcher support;
- obtain necessary quotas and cost approval;
- reproduce one selected configuration in SageMaker;
- reuse the existing evaluation, model-card, and registry pipeline.

Exit condition: SageMaker artifacts match the shared contract and S3 provenance is recorded.

### Phase 8: Caco-2 regression extension

- add training-only target standardization;
- add Caco-2 regression head and loss;
- implement mixed task/loss balancing;
- add regression predictions and metrics;
- repeat leakage audit, local smoke, cluster comparison, and SageMaker validation.

Exit condition: classification metrics remain acceptable and Caco-2 is evaluated against same-split Ridge/Morgan and single-task ChemBERTa comparators.

## Unit-Test Requirements

Tests must not contact TDC, Hugging Face, AWS, or external model registries unless explicitly marked as integration tests and excluded from the standard suite.

Required unit coverage includes:

1. multi-task YAML loading and validation;
2. deterministic task order;
3. separate DataLoaders and correct split use;
4. canonical-SMILES-only tokenization;
5. identifier and target exclusion from features;
6. class-weight calculation from training data only;
7. one-class training error behavior;
8. task-head construction and routing;
9. unknown task rejection;
10. output shapes for all classification heads;
11. finite loss and gradients;
12. deterministic round-robin sampling;
13. exhausted-loader restart behavior;
14. per-task batch accounting;
15. exact cross-task overlap detection;
16. scaffold-overlap detection;
17. coordinated exclusion behavior;
18. conflicting-label reporting;
19. split hash and manifest generation;
20. validation-only checkpoint selection;
21. proof that test metrics cannot change selection;
22. unavailable metric handling;
23. checkpoint save/load;
24. optimizer, scheduler, sampler, and RNG resume;
25. incompatible checkpoint rejection;
26. per-task prediction schemas;
27. JSON-safe metadata;
28. deterministic output for fixed inputs and seeds;
29. CLI argument parsing;
30. CPU smoke execution using tiny local fixtures.

The full existing test suite must remain green.

## Smoke-Test Requirements

The local smoke run must:

- use 100-300 or fewer synthetic/public-safe examples per task;
- run on CPU without requiring CUDA;
- use a bounded number of steps;
- process at least two batches per task;
- produce finite per-task losses;
- prove encoder and appropriate head gradients are present;
- save a checkpoint;
- resume from that checkpoint;
- restore the best validation checkpoint;
- write separate validation metrics and predictions for all three tasks;
- write manifests and warnings;
- avoid downloading models by using a locally available checkpoint, mock, or tiny constructed model in automated tests.

Smoke metrics must never be presented as scientific performance.

## First-Milestone Acceptance Criteria

The initial classification-only implementation is complete only when:

- `BBB_Martins`, hERG, and AMES are represented by validated endpoint configs;
- one shared encoder and three binary heads are implemented;
- separate task DataLoaders work with deterministic round-robin sampling;
- per-task class weights use training labels only;
- cross-task exact-SMILES and scaffold audit artifacts are generated;
- prohibited cross-task train/validation/test leakage blocks training;
- a local CPU smoke run completes;
- each head produces separate validation metrics and predictions;
- checkpoint saving, loading, and resume work;
- checkpoint selection uses validation metrics only;
- run metadata records config, seed, split hashes, model revision, package versions, and development limits;
- all JSON artifacts are standards-compliant;
- all existing and new tests pass;
- no AWS job is launched;
- no Docker image is pushed;
- no model, dataset, cache, credential, or local path is committed;
- no Git commit is created without explicit user approval.

## Risks and Mitigations

### Negative transfer

One endpoint may harm another because useful features and label noise differ.

Mitigation:

- retain single-task references;
- report every endpoint independently;
- monitor worst-task degradation, not only the mean score;
- test encoder freezing or adapters later if full sharing is harmful.

### Task domination

Larger endpoints may dominate gradient updates.

Mitigation:

- use round-robin sampling first;
- record task contribution counts;
- compare alternative sampling only as controlled ablations.

### Class imbalance

Endpoint class distributions differ and accuracy may be misleading.

Mitigation:

- calculate per-task weights from training data only;
- emphasize ROC-AUC, PR-AUC, balanced accuracy, and MCC;
- tune any threshold on validation data only.

### Cross-task leakage

A held-out compound or scaffold for one task may appear in another task's training set.

Mitigation:

- make coordinated overlap auditing mandatory;
- use deterministic exclusions or a global scaffold split;
- keep official and coordinated evaluation tracks distinct.

### Small validation sets

Metrics may be unstable, particularly for minority classes.

Mitigation:

- preserve counts and class distributions;
- add uncertainty estimates in scientific comparisons;
- avoid overinterpreting small differences.

### Checkpoint-selection bias

A mean score may hide endpoint degradation or repeated validation tuning may overfit.

Mitigation:

- show per-task metrics and degradation limits;
- limit search scope;
- reserve test data for one final evaluation.

### Reproducibility across environments

Different CUDA, PyTorch, Transformers, tokenizer, or model revisions can change results.

Mitigation:

- pin compatible dependencies;
- record package and hardware versions;
- pin or resolve the pretrained revision;
- use the same shared training entry point everywhere.

### Cluster policy and persistence

Without a scheduler, jobs may collide with other users or stop after disconnects.

Mitigation:

- confirm authorization and GPU-use rules;
- select one free GPU explicitly;
- use `tmux` or approved persistence tooling;
- save resumable checkpoints frequently enough for the environment.

### Cost and quota risk

SageMaker experiments can be delayed by quotas or multiplied by debugging runs.

Mitigation:

- debug locally;
- use the cluster for repeated scientific experiments when authorized;
- run a selected configuration once in SageMaker;
- retain dry-run, maximum-runtime, budget, and manifest controls.

## Recommended Implementation Sequence

Create small, reviewable Codex tasks in this order:

1. add AMES endpoint configuration and multi-task configuration schema;
2. implement multi-task data loading and leakage audit;
3. implement shared encoder and classification heads;
4. implement losses and round-robin sampler;
5. implement trainer, metrics, artifacts, and checkpoint resume;
6. run and verify the local CPU smoke workflow;
7. document and execute the one-GPU cluster smoke workflow;
8. compare full single-task and multi-task results on coordinated splits;
9. add the thin SageMaker training contract and dry-run launcher;
10. add Caco-2 regression only after classification results are reviewed.

Each task should preserve unrelated worktree changes, run tests proportionate to risk, report generated artifacts, and avoid commits or external side effects unless explicitly requested.
