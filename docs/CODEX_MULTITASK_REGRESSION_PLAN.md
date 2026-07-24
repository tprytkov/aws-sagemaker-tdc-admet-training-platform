# Codex Implementation Plan: Separate Multi-Task Regression Pipeline

## Goal

Add a **separate multi-task regression model** to the existing ADMET training platform while preserving the current multi-task classification pipeline.

The project should support two independent model families:

1. **Multi-task classification model**
   - Existing and already validated.
   - Current endpoints: BBB_Martins, hERG_Karim, AMES.
   - Do not redesign or destabilize this pipeline.

2. **Multi-task regression model**
   - New parallel pipeline.
   - Shared ChemBERTa encoder with one regression head per continuous ADMET endpoint.
   - Start with several continuous endpoints rather than Caco-2 alone.
   - Initial target: 3–5 carefully selected regression endpoints.

The long-term deployment target is MolOptima, where outputs from both models will be combined into one unified ADMET report.

---

## Core Architecture

```text
                         SMILES
                           |
              +------------+------------+
              |                         |
              v                         v
    Multi-task Classification   Multi-task Regression
         ChemBERTa encoder        ChemBERTa encoder
              |                         |
       +------+------+          +-------+-------+------+
       |      |      |          |       |       |      |
      BBB    hERG   AMES       Caco2   LogS    PPB    ...
       |      |      |          |       |       |      |
 probabilities / classes        continuous predictions
```

The regression model should use:

```text
Shared ChemBERTa encoder
        |
pooled molecular representation
        |
+-------+--------+--------+--------+
|       |        |        |        |
head1   head2    head3    head4    ...
|       |        |        |
scalar  scalar   scalar   scalar
```

Each regression head outputs one continuous value.

---

## Important Design Constraint

**Do not convert the existing classification trainer into a mixed classification/regression trainer in this milestone.**

Keep classification and regression as separate training paths.

Reuse shared infrastructure where safe, but regression-specific logic should be isolated.

Preferred structure:

```text
shared/common
├── encoder loading
├── tokenizer
├── prepared split utilities
├── leakage-control helpers
├── deterministic training utilities
├── checkpoint/provenance helpers
└── hashing/manifests

classification
├── BCE loss
├── sigmoid probabilities
├── ROC-AUC / PR-AUC / F1 / MCC
└── existing validated workflow

regression
├── Huber or MSE loss
├── target transforms
├── train-only normalization
├── inverse transforms
├── RMSE / MAE / R2 / Spearman / Pearson
└── new workflow
```

---

# Phase 1 — Audit Existing Code Before Editing

Before implementation:

1. Inspect the current model, trainer, data loaders, split logic, checkpoint selection, manifests, final evaluation workflow, tests, and configuration structure.
2. Identify which modules are truly generic and safe to reuse.
3. Do not modify production classification behavior unless required for a clearly shared abstraction.
4. Preserve all existing classification tests and outputs.
5. Do not alter current coordinated classification split files.

Produce a short implementation note before making broad refactors.

---

# Phase 2 — Select Initial Regression Endpoints

Identify 3–5 continuous ADMET endpoints suitable for a first multi-task regression model.

Prefer endpoints that:

- are continuous rather than binary,
- have enough samples for meaningful training,
- have scientifically useful ADMET meaning,
- are available through the existing TDC/PyTDC workflow,
- have clearly documented units and target definitions,
- can be split with the same leakage-safe global scaffold strategy.

Include **Caco2_Wang** if compatible with the current source/version.

Potential endpoint categories:

- Caco-2 permeability
- aqueous solubility / LogS
- plasma protein binding
- clearance
- half-life
- other continuous pharmacokinetic properties

Do not silently assume that similarly named datasets use equivalent target units.

For each selected endpoint, record:

```text
dataset name
source
row count
target column
units
target meaning
recommended transform
missing-value policy
duplicate policy
```

Create a source-audit config or manifest similar in spirit to the classification workflow.

---

# Phase 3 — Leakage-Safe Coordinated Regression Splits

Implement a separate coordinated regression split workflow.

Requirements:

1. Canonicalize SMILES.
2. Use the existing safe Murcko scaffold utility where appropriate.
3. Prevent exact-molecule leakage across train/validation/test globally across all regression endpoints.
4. Prevent scaffold leakage globally across regression endpoints.
5. Handle duplicates explicitly.
6. For duplicate structures within one endpoint:
   - do not blindly keep conflicting numeric labels,
   - define a deterministic aggregation or quarantine rule,
   - record what was done.
7. Preserve test data as untouched until final locked evaluation.
8. Store hashes and manifests.

Suggested new files:

```text
src/admet_platform/data/coordinated_multitask_regression.py
scripts/build_coordinated_multitask_regression_splits.py
configs/multitask_regression_source_audit.yaml
```

Do not overwrite or reuse the classification coordinated split directory.

Suggested output root:

```text
outputs/local/multitask_regression/coordinated/
```

---

# Phase 4 — Train-Only Target Transformations

Regression targets may have different units and distributions.

Implement endpoint-specific target preprocessing.

Pipeline:

```text
raw target
    |
optional scientifically justified transform
    |
fit normalization on TRAIN ONLY
    |
standardized target
    |
model training
```

At inference/evaluation:

```text
standardized prediction
    |
inverse normalization
    |
inverse optional transform
    |
prediction in original units
```

Requirements:

- fit mean/std or other scaler parameters using train only,
- never use validation/test statistics to fit transformations,
- save transformation metadata per endpoint,
- support inverse transform,
- preserve original target units in evaluation outputs,
- record transform decisions in model metadata.

Suggested artifact:

```text
target_transforms.json
```

Do not hard-code transforms without documenting scientific justification.

---

# Phase 5 — Regression Model

Reuse the existing generic ChemBERTa encoder design where possible.

Expected behavior:

```text
AutoModel encoder
    |
masked_mean or configured pooling
    |
dropout
    |
one Linear(hidden_size, 1) head per regression task
```

No sigmoid.

Output should be raw scalar predictions on the normalized target scale during training.

Potential file:

```text
src/admet_platform/models/multitask_regression_chemberta.py
```

If the existing `MultiTaskChemBERTa` is already generic enough to support scalar heads without classification-specific assumptions, prefer minimal reuse rather than duplication.

Do not break existing classification loading/checkpoint formats.

---

# Phase 6 — Regression Trainer

Create a separate trainer.

Suggested files:

```text
src/admet_platform/training/multitask_regression_trainer.py
src/admet_platform/training/multitask_regression_control.py
src/admet_platform/training/regression_metrics.py
scripts/train_multitask_regression.py
configs/multitask_regression.yaml
```

Training requirements:

- shared encoder,
- one regression head per endpoint,
- round-robin task scheduling initially,
- configurable task weights,
- BF16 support,
- deterministic mode,
- gradient clipping,
- separate encoder/head learning rates,
- warmup/decay scheduler,
- checkpointing,
- validation-only checkpoint selection,
- no test access during training.

Initial loss:

```text
Huber loss
```

Reason:
- more robust to outliers than plain MSE,
- often appropriate for noisy experimental ADMET measurements.

Make the loss configurable so MSE can be tested later without redesign.

---

# Phase 7 — Regression Metrics

For each endpoint report at least:

```text
RMSE
MAE
R2
Spearman correlation
Pearson correlation
row count
validation/test loss
```

Save predictions with:

```text
molecule_id
canonical_smiles
target_original
prediction_original
residual_original
target_normalized
prediction_normalized
```

Do not use ROC-AUC, PR-AUC, F1, MCC, sensitivity, or specificity for regression.

---

# Phase 8 — Checkpoint Selection

Do not average raw RMSE values across endpoints with different units.

Primary checkpoint criterion:

```text
lowest mean validation normalized RMSE
```

Possible secondary tie-breaker:

```text
highest mean validation Spearman correlation
```

Requirements:

1. Selection must use validation only.
2. Save the exact selection reason.
3. Save endpoint-level metrics at every evaluation.
4. Preserve a full validation history.
5. Never choose separate endpoint checkpoints for one shared model.

---

# Phase 9 — Single-Task Regression Baselines

Before claiming multi-task benefit, train one single-task regression baseline per endpoint.

Use:

- same backbone,
- same coordinated split,
- same preprocessing,
- same target transform,
- same optimization policy,
- matched endpoint-specific batch exposure where practical.

Compare:

```text
single-task RMSE vs multi-task RMSE
single-task MAE vs multi-task MAE
single-task Spearman vs multi-task Spearman
```

Define and document a negative-transfer criterion for regression before final comparison.

Do not invent a tolerance after seeing results.

---

# Phase 10 — Convergence Experiment

The existing classification 3,000-step run was a matched-exposure experiment, not proof of convergence.

For regression, build convergence analysis from the start.

Use a configurable high ceiling, for example:

```text
max_steps: 20000
evaluation_interval_steps: 200
minimum_training_steps_before_stopping: configurable
early_stopping_patience_evaluations: configurable
```

Record validation curves for:

```text
mean normalized RMSE
mean Spearman
endpoint RMSE
endpoint Spearman
```

Operational definition of convergence:

> No meaningful improvement in the primary validation metric over a predefined patience window, with no clear continuing improvement in major endpoints.

Do not claim convergence solely because `max_steps` was reached.

---

# Phase 11 — Locked Final Evaluation

After:

- endpoint set is fixed,
- target transforms are fixed,
- architecture is fixed,
- hyperparameters are fixed,
- checkpoint-selection rule is fixed,
- multi-seed protocol is fixed,

run a single locked evaluation on untouched regression test sets.

Create a regression-specific final evaluation script parallel to the classification workflow.

Suggested file:

```text
scripts/evaluate_final_regression_test.py
```

Outputs:

```text
test_predictions_<endpoint>.csv
test_metrics.json
endpoint_comparison.csv
run_manifest.json
hashes/provenance
```

---

# Phase 12 — MolOptima Export Contract

The training repository should export inference-ready artifacts only.

Do not implement MolOptima application logic here.

Suggested regression artifact:

```text
moloptima-chemberta-admet-regression-v1/
├── model_state/
├── tokenizer/
├── encoder_config/
├── multitask_model_config.json
├── endpoint_metadata.json
├── target_transforms.json
├── inference_config.json
├── model_card.json
├── validation_metrics.json
├── test_metrics.json
├── dataset_manifest.json
└── SHA256SUMS
```

MolOptima will later combine:

```text
classification model outputs
+
regression model outputs
+
RDKit
+
ADMET-AI / Chemprop
+
other evidence providers
```

Do not commit large model weights to GitHub.

---

# Expected Classification/Regression Separation

## Classification model

Current and future binary endpoints:

```text
BBB
hERG
AMES
CYP inhibition classes
toxicity classes
other binary ADMET outcomes
```

Outputs:

```text
probabilities
binary decisions where applicable
```

Metrics:

```text
ROC-AUC
PR-AUC
Balanced Accuracy
F1
MCC
```

## Regression model

Continuous endpoints:

```text
Caco-2 permeability
solubility
PPB
clearance
half-life
other continuous ADMET values
```

Outputs:

```text
continuous values in original scientific units
```

Metrics:

```text
RMSE
MAE
R2
Spearman
Pearson
```

---

# Testing Requirements

Add focused tests for:

1. regression target normalization fit on train only,
2. inverse transformations,
3. no validation/test leakage,
4. continuous-label data loading,
5. regression head output shape,
6. Huber/MSE loss behavior,
7. regression metrics,
8. checkpoint selection using normalized RMSE,
9. round-robin contributions,
10. checkpoint save/load,
11. deterministic initialization,
12. offline local encoder loading,
13. output prediction schema,
14. classification pipeline regression tests remain green.

Run:

```text
pytest
ruff
git diff --check
```

Do not accept implementation that breaks existing classification tests.

---

# Implementation Strategy for Codex

Work in small milestones.

## Milestone 1

Audit current code and implement:

- endpoint source audit,
- coordinated regression splits,
- target transformation metadata,
- tests.

Do not start model training code yet.

## Milestone 2

Implement:

- regression model wrapper or generic-head reuse,
- regression trainer,
- Huber loss,
- regression metrics,
- config,
- unit tests.

Run a tiny synthetic/offline smoke test.

## Milestone 3

Run one real local regression endpoint smoke test.

Verify:

- data loading,
- target transforms,
- BF16,
- forward/backward,
- checkpointing,
- validation,
- saved predictions.

## Milestone 4

Run 3–5 endpoint multi-task regression smoke test.

Confirm:

- one batch per task,
- round-robin behavior,
- finite losses,
- per-task metrics,
- deterministic initialization.

## Milestone 5

Train single-task baselines and controlled multi-task regression model.

Do not open the test set.

## Milestone 6

Convergence, multi-seed analysis, then locked final test.

---

# Non-Negotiable Scientific Constraints

- No exact-molecule leakage across train/validation/test.
- No scaffold leakage across train/validation/test.
- Target transformations fit on train only.
- Test data untouched until the final locked evaluation.
- Validation only for checkpoint selection.
- No post-hoc tolerance selection.
- No averaging raw RMSE across differently scaled endpoints.
- Record units and target definitions.
- Save hashes and provenance.
- Preserve deterministic reproducibility.
- Keep classification and regression models separate.
- Do not silently change the existing classification pipeline.

---

# Definition of Done for the First Regression Milestone

The first milestone is complete when:

1. 3–5 continuous endpoints are selected and documented.
2. Coordinated leakage-safe regression splits are generated.
3. Train-only target normalization is implemented and tested.
4. A shared ChemBERTa regression model supports multiple scalar heads.
5. Multi-task regression training runs offline on GPU.
6. Validation reports RMSE, MAE, R2, Spearman, and Pearson per endpoint.
7. Composite checkpoint selection uses mean normalized validation RMSE.
8. Single-task regression baselines can be trained using the same splits.
9. Existing classification tests remain unchanged and passing.
10. No test data has been used for model selection.

---

# Codex Working Rule

Before editing, inspect the current repository and adapt this plan to the actual file structure.

Do not make a large refactor in one pass.

Prefer small, reviewable commits:

```text
1. regression data/split infrastructure
2. regression transforms/metrics
3. regression model/trainer
4. configs/scripts/tests
5. smoke-test fixes
6. controlled training workflow
```

Preserve current validated classification behavior throughout.
