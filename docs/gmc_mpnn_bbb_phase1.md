# GMC-MPNN BBB Phase-1 scientific and implementation contract

## Purpose and scope

Phase 1 evaluates GMC-MPNN as a specialized, endpoint-specific predictor of blood-brain barrier
(BBB) permeability. The endpoint is `BBB_Martins`, the task is binary classification, and class
`1` is the BBB-permeable (positive) class.

ChemBERTa and Chemprop did not show sufficiently strong independent generalization for BBB in this
project. GMC-MPNN is therefore being evaluated as a possible replacement or specialized BBB
predictor. Phase 1 establishes a faithful, reproducible GMC-MPNN baseline; it does not establish
that GMC-MPNN is superior to either previous approach.

This document is the scientific and implementation contract for the work. Changes to it that could
affect model selection, leakage control, or comparability must be documented before the affected
experiment is run.

## Phase-1 model contract

Phase 1 must reproduce the published-style GMC-MPNN baseline as closely as practical. The official
implementation and a specific source revision must be inspected before model code is ported or
adapted.

The baseline must:

- preserve the original geometric preprocessing and conformer handling;
- preserve the original molecular graph construction and feature definitions;
- preserve the original model architecture; and
- preserve the original training and inference behavior where practical.

An extra global RDKit descriptor branch is explicitly out of scope for Phase 1. This restriction
does not prohibit RDKit where the original GMC-MPNN workflow requires it for molecular parsing,
preprocessing, conformer generation, geometry, or graph construction.

A compatibility change is permitted only when it is strictly required to run the original approach
in this project. Every such change must be recorded with its reason, affected behavior, and expected
scientific impact. It must not be presented as a faithful reproduction without qualification.

## Data and split contract

Phase 1 must reuse the existing leakage-controlled `BBB_Martins` training and validation membership
already used by this project. It must not create a new random train/validation split unless a later,
explicitly approved contract change authorizes one. Molecule identity and split provenance must be
preserved through preprocessing and prediction export.

The previously evaluated 208-molecule BBB test set is an opened test set. It must not be used for:

- architecture selection;
- hyperparameter tuning;
- model selection or checkpoint selection;
- calibration fitting;
- classification-threshold selection; or
- early-stopping decisions.

No labels, predictions, metrics, distributions, or other statistics derived from that set may
influence model development. It may later be reported only as an explicitly exploratory comparison
because it has already been evaluated. A new independent external BBB evaluation set is required
for the final GMC-MPNN comparison and before any claim that GMC-MPNN outperforms the previous BBB
models.

Data preparation and reporting must detect and report exact-SMILES overlap and scaffold overlap
between the relevant splits where applicable. The overlap method, SMILES normalization or
canonicalization procedure, scaffold definition, and counts must be recorded so the check is
reproducible.

## Development order and seeds

The planned seeds are `13`, `37`, `73`, `101`, and `137`, matching the Chemprop BBB experiments.
Five-seed training must not begin immediately. The first milestone is a successful seed-13 data
preparation, training, and inference run. Its learning behavior, outputs, runtime, and
reproducibility must be reviewed before launching seeds `37`, `73`, `101`, and `137`.

## Model selection and leakage control

All model-development and model-selection decisions must use training and validation data only.
Validation data alone must govern early stopping, checkpoint selection, architecture or
hyperparameter comparisons, calibration fitting if calibration is used, and classification-
threshold selection.

Any calibration method must be named and its validation-only fitting procedure documented. The
classification threshold must be exported with the selection rule and the validation predictions
from which it was selected. The old BBB test set must remain absent from these procedures.

## Evaluation and comparison

The primary validation metric is AUROC. Each run must also report:

- AUPRC;
- balanced accuracy;
- Matthews correlation coefficient (MCC);
- sensitivity;
- specificity;
- Brier score;
- calibration error, including the exact definition and binning or estimator settings;
- confusion matrix; and
- classification threshold.

Metrics that require hard predictions must use the recorded validation-selected threshold.
Confusion-matrix class order and the convention that BBB-permeable is positive must be explicit.

The five-seed result must summarize seed-level values and across-seed variability. Its comparison
with the existing Chemprop BBB validation must use exactly the same validation molecules. The
comparison must verify molecule identity and ordering rather than relying only on equal row counts.
Differences in preprocessing, calibration, or threshold selection must be disclosed. Validation
comparison supports model development; it is not evidence of superior independent generalization.

## Reproducibility and run records

Every run must produce a self-contained record sufficient to reconstruct what was executed. At a
minimum, retain:

- resolved configuration;
- model architecture and hyperparameters;
- seed and determinism settings;
- preprocessing configuration and settings;
- geometry and conformer-generation settings;
- graph-construction and feature settings;
- dataset hashes or equivalent immutable provenance, including split membership;
- software and package versions;
- Python, PyTorch, CUDA runtime, CUDA driver, and device information where applicable;
- training duration;
- checkpoint path and checkpoint-selection rule;
- best validation epoch;
- validation predictions with stable molecule identifiers and labels;
- validation metrics, confusion matrix, calibration details, and threshold;
- overlap-check results; and
- a machine-readable run summary and enough reproducibility metadata to rerun the experiment.

Paths in committed configuration or documentation must remain portable and must not expose private
local machine paths. Runtime records may resolve paths for the execution environment, but those
records belong with ignored run artifacts unless intentionally sanitized and approved for source
control.

## Artifact policy

The existing project artifact policy remains in force. Large checkpoints, generated conformers,
geometry caches, temporary preprocessing outputs, predictions, and other run artifacts must not be
committed to Git unless their inclusion is explicit, intentional, reviewed, and compatible with
repository size and privacy requirements. Ignore rules and output staging must be verified before
full preprocessing or training begins.

## Proposed project layout

The anticipated implementation layout is:

```text
src/admet_platform/gmc_mpnn/
configs/gmc_mpnn/
scripts/run_gmc_mpnn_experiment.py
tests/gmc_mpnn/
outputs/gpu/pilot/gmc_mpnn_bbb_seed13/
outputs/gpu/pilot/gmc_mpnn_bbb_seed37/
outputs/gpu/pilot/gmc_mpnn_bbb_seed73/
outputs/gpu/pilot/gmc_mpnn_bbb_seed101/
outputs/gpu/pilot/gmc_mpnn_bbb_seed137/
```

This is a proposed boundary, not authorization to create the implementation files. Source modules
should isolate GMC-MPNN preprocessing, graph/geometry construction, model definition, training,
inference, and artifact serialization as warranted by the inspected official implementation.
Configs should make all scientifically meaningful choices explicit. Tests should cover deterministic
data conversion, split preservation, feature/shape contracts, leakage checks, and minimal
training/inference behavior without requiring full production training.

## Phase-1 success criteria

Phase 1 succeeds only when all of the following are satisfied:

1. The original GMC-MPNN implementation, scientific behavior, and dependencies are understood and
   recorded.
2. The existing leakage-controlled BBB data can be converted reproducibly to the required
   GMC-MPNN inputs while preserving molecule identity and split provenance.
3. Seed-13 training and inference complete successfully and emit the required run record,
   validation predictions, metrics, and checkpoint metadata.
4. Seed-13 data preparation and validation results are reproducible under the documented settings.
5. The subsequent five-seed run completes with stable validation behavior and all required
   seed-level and aggregate reporting.
6. Five-seed GMC-MPNN validation is compared with existing Chemprop BBB validation on the same
   validation molecules.
7. No claim of GMC-MPNN superiority is made until evaluation on a new independent external BBB set.

"Stable" five-seed validation means that all five runs complete without unexplained preprocessing,
optimization, or inference failures; produce finite required metrics and complete artifacts; and
show variability that is quantified and scientifically reviewed rather than hidden by an aggregate.

## Phase 2: optional global-descriptor experiment

Phase 2 is a separate, optional experiment that may add a compact, explicitly selected set of
global RDKit descriptors to GMC-MPNN. It must not begin until the descriptor-free Phase-1 baseline
is complete and reviewed. Phase 2 will require its own written contract covering descriptor
selection, train-only fitting, missing-value handling, scaling, architecture integration, ablation,
and validation-only model selection. Phase-2 results must remain distinguishable from the faithful
descriptor-free baseline.

## Next implementation step

The next Codex task should inspect the official GMC-MPNN BBB implementation and prepare a detailed
implementation plan before porting or adapting any model code. That inspection should identify the
source repository and exact revision, license, Python and PyTorch requirements, CUDA requirements,
RDKit requirements, geometric preprocessing, conformer generation, graph construction, feature
definitions, model architecture, loss function, optimizer and scheduler, training loop, checkpoint
format, inference workflow, dataset format, and the original BBB datasets and splits.
