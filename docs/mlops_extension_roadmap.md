# MLOps Extension Roadmap

## Purpose and relationship to the multi-task plan

This document extends the existing
[`multitask_chemberta_implementation_plan.md`](multitask_chemberta_implementation_plan.md). It
does not replace or revise that plan. The existing plan remains the source of truth for the
multi-task scientific design, data contracts, cross-task leakage controls, shared encoder and
task heads, task sampling, loss design, checkpointing, artifact contract, and two-track
evaluation strategy.

This roadmap describes how a validated implementation progresses from low-cost continuous
integration through cluster experiments, final model selection, AWS registration and serving,
operational monitoring, and a separate local Kubernetes inference demonstration. The stages are
deliberately ordered so that expensive or externally stateful work begins only after local and
cluster evidence justifies it.

This is a roadmap, not a record that every capability is already deployed. In particular, a local
registry JSON entry is not a SageMaker Model Registry model version, and a dry-run launch plan is
not an AWS resource.

## Architectural fit

The roadmap preserves the repository's current separation of concerns:

```text
configs/                         scientific and AWS execution configuration
src/admet_platform/data/         preparation, schemas, and leakage auditing
src/admet_platform/features/     RDKit descriptors and fingerprints
src/admet_platform/models/       classical, single-task, and multi-task models
src/admet_platform/training/     reusable platform-independent training logic
src/admet_platform/evaluation/   comparison, model cards, and local registry metadata
src/admet_platform/sagemaker/    thin SageMaker adapters and launch contracts
scripts/                         local and cluster command-line entry points
sagemaker/                       managed-job entry points and dependency manifests
docker/                          existing Processing/evaluation image definitions
infra/terraform/                 S3, ECR, IAM, KMS, CloudWatch, and budget foundation
tests/                           offline unit, contract, and smoke tests
model_registry/                  lightweight public-safe local metadata
```

New components should follow the same pattern: reusable behavior belongs under
`src/admet_platform/`; environment-specific adapters remain thin; infrastructure is declared in
Terraform; generated data, model weights, credentials, account-specific values, and large
artifacts stay out of Git.

## Guiding controls

- Use public TDC or synthetic public-safe data only.
- Keep development and automated CI offline, deterministic, CPU-only, and inexpensive.
- Use validation metrics exclusively for checkpoints, hyperparameters, thresholds, loss weights,
  architecture choices, and final candidate selection.
- Keep test splits untouched until the complete selection policy has chosen a final model.
- Preserve both evaluation tracks from the existing plan: official endpoint-specific TDC
  benchmarks and coordinated leakage-safe multi-task research splits.
- Compare models only on compatible split populations. Retrain single-task comparators on the
  coordinated splits when measuring multi-task effects.
- Treat AWS deployment, endpoint creation, and approval as explicit, reviewed actions with cost,
  security, and rollback considerations.
- Treat model outputs as computational screening signals, not experimental or clinical evidence.

## Phase 1: lightweight GitHub Actions CI

Add one lightweight workflow triggered by both `push` and `pull_request`. Its purpose is rapid
regression detection, not training or cloud integration.

The standard CI job should:

1. check out the repository and configure the project-supported Python 3.11 runtime;
2. install only the dependencies required by the fast test suite;
3. run `python -m pytest -q` or a documented fast-test marker subset if the complete suite later
   includes opt-in integration tests;
4. run entirely on CPU with deterministic synthetic fixtures;
5. test tokenizer, model-forward, loss, sampler, checkpoint, selection-isolation, and artifact
   contracts using tiny Transformers fixtures constructed and saved locally during the test;
6. fail if an unmarked standard test attempts network or AWS access.

CI constraints are strict:

- no AWS credentials or AWS calls;
- no GPU runner;
- no full model training;
- no TDC, Hugging Face, or other external downloads;
- no dependency on a pre-populated Hugging Face cache;
- no Docker image build or push;
- no generated model artifact upload.

Where RDKit installation dominates runtime, use a pinned environment strategy compatible with the
repository dependency files and cache only package-manager artifacts, never downloaded datasets or
model weights. Tests that require real services, full datasets, GPUs, or network access must be
explicitly marked as integration tests and excluded from the default workflow.

Exit criterion: a clean checkout can execute the fast offline CPU suite on every push and pull
request without secrets or external services.

### Phase 1 implementation status

The lightweight workflow is implemented in `.github/workflows/ci.yml` for both `push` and
`pull_request` events on a GitHub-hosted Ubuntu runner with Python 3.11. It installs the pinned,
CPU-only test dependencies in `requirements-ci.txt`, checks Python syntax, parses and contract-loads
the multi-task and three classification endpoint YAML files, and runs an explicit fast pytest
selection. The selected tests cover configuration and data contracts, scaffold preparation, the
shared model with locally constructed tiny Transformers checkpoints, losses, round-robin sampling,
checkpoint/resume determinism, validation-only training controls, and an end-to-end synthetic
offline training smoke path.

The workflow deliberately excludes AWS and SageMaker modules, TDC downloads, Docker workflows,
Terraform, GPU/CUDA execution, full scientific training, deployment, registry, monitoring, and
Kubernetes work. Hugging Face Hub, Transformers, and Datasets offline variables are set for the
entire job, and model tests use only temporary tiny local fixtures.

To reproduce the CI checks locally from an activated Python 3.11 environment, install
`requirements-ci.txt` and the project, then run:

```text
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0
python -m pip install -r requirements-ci.txt
python -m pip install --no-deps -e .
python -m compileall -q src scripts tests
python -c "from pathlib import Path; import yaml; from admet_platform.config import load_endpoint_config; from admet_platform.data.multitask import load_multitask_config; paths=(Path('configs/multitask_classification.yaml'), Path('configs/bbb_martins.yaml'), Path('configs/herg_karim.yaml'), Path('configs/ames.yaml')); assert all(isinstance(yaml.safe_load(path.read_text(encoding='utf-8')), dict) for path in paths); load_multitask_config(paths[0]); [load_endpoint_config(path) for path in paths[1:]]"
python -m pytest -q tests/test_config_loader.py tests/test_multitask_data.py tests/test_scaffold_preparation.py tests/test_multitask_chemberta.py tests/test_multitask_losses.py tests/test_multitask_sampler.py tests/test_multitask_control.py tests/test_multitask_trainer.py tests/test_multitask_prepared_training.py
```

Set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `HF_DATASETS_OFFLINE=1` in the local shell to
match the workflow's download safeguards exactly.

## Phase 2: cluster GPU validation

Use the work cluster primarily for training and validation. Do not assume SLURM, Kubernetes access,
cluster administrator rights, or permission to deploy a persistent service. Follow local cluster
policy, select one authorized GPU explicitly, and use the same `scripts/` entry points and
`src/admet_platform/` training code used locally.

### 2.1 One-GPU float32 smoke

Run a bounded, one-GPU float32 smoke test before mixed precision or full experiments. It should
confirm:

- data and tokenizer contracts load correctly;
- every configured task receives batches and every head receives gradients;
- losses and gradients remain finite;
- validation is run on the correct split;
- checkpoint save, reload, and resume work;
- metrics, predictions, manifests, warnings, and provenance are written;
- GPU memory use and throughput are recorded.

This smoke is a correctness test, not a result for model comparison.

### 2.2 One-GPU mixed-precision smoke

After float32 succeeds, repeat the bounded run with the intended mixed-precision mode. Compare its
loss trajectory and validation outputs with float32 within documented tolerances. Fail clearly on
unsupported hardware, non-finite values, scaler failures, or unacceptable numerical divergence.
Mixed precision is an optimization and must not become a prerequisite for correctness.

### 2.3 Full experiments

Run complete, reproducible experiments on one GPU:

- the existing single-task ChemBERTa configurations for each endpoint;
- the classification-only multi-task configuration for `bbb_martins`, `herg_karim`, and `ames`;
- any same-split single-task comparators needed for the coordinated multi-task evaluation track;
- the Caco-2 regression extension only after its later milestone in the existing plan is ready.

Record the source commit, resolved base-model and tokenizer revisions, environment/package versions,
random seeds, effective configuration, split-manifest hashes, leakage-audit identity, device and
precision, and output artifact hashes. Full experiments may obtain the approved pretrained model
and public datasets according to cluster policy; automated CI may not.

All checkpoint selection and early stopping use validation data only. Test labels and metrics must
not influence checkpoint selection, thresholds, hyperparameters, task weighting, or decisions to
repeat an experiment.

Exit criterion: float32 and mixed-precision smokes pass, and complete single-task and multi-task
runs produce contract-compliant, reproducible candidate artifacts.

## Phase 3: final model selection

Selection must compare the following candidate families for each endpoint:

1. classical Morgan-fingerprint baselines;
2. classical RDKit-descriptor baselines;
3. single-task ChemBERTa;
4. the corresponding head of multi-task ChemBERTa.

Use only validation results from compatible data and split tracks. The official TDC track remains
the endpoint benchmark. The coordinated leakage-safe track provides a fair estimate of shared
representation learning and requires same-split single-task comparators.

### Negative-transfer safeguards

An average multi-task score must not hide harm to an individual endpoint. The selection report
should include:

- every endpoint's primary and secondary validation metrics;
- per-endpoint change from its compatible single-task ChemBERTa reference;
- per-endpoint change from its strongest compatible classical baseline;
- the worst per-task degradation;
- a configured degradation tolerance and an explicit review status when it is exceeded;
- missing or unusable metrics represented as `null` with warnings, never silently omitted;
- efficiency considerations such as artifact size, memory, and latency, reported separately from
  predictive quality.

Recommendations remain endpoint-specific. A multi-task artifact is not automatically approved for
all heads because its mean validation score is best. If negative transfer exceeds tolerance, retain
the stronger single-task or classical model for the affected endpoint, or run a predeclared
training-only mitigation before final selection.

Freeze the candidate set, metric policy, tolerance, and tie-breaking rules before test evaluation.
Then restore the selected validation checkpoint and evaluate each selected model once on its
untouched test set. Test results are final descriptive evidence; they do not reopen selection or
trigger tuning. Any later revision starts a new, clearly identified experiment cycle.

Exit criterion: an auditable validation-only decision record selects one deployable artifact and
its endpoint scope, followed by a separately recorded untouched-test evaluation.

## Phase 4: AWS MLOps lifecycle

Begin AWS work only for the selected, test-evaluated artifact. Use existing dry-run launch contracts
and Terraform outputs where applicable. Review region, account, least-privilege role, encryption,
retention, quota, and expected endpoint cost before creating resources.

### 4.1 Upload the selected artifact to S3

Package the model, tokenizer, inference metadata, effective configuration, validation and final test
reports, model card, split and run manifests, warnings, dependency versions, and cryptographic
checksums. Upload to a versioned, immutable-by-convention S3 prefix such as
`models/<model-id>/<artifact-version>/`. Do not upload credentials, caches, raw private paths, or
unapproved data.

### 4.2 SageMaker evaluation

Run the existing SageMaker evaluation Processing contract against the exact S3 artifact. Evaluation
must verify package integrity, inference loading, schema compatibility, metric reproduction, and
provenance. It must not retrain or use test results to select a different candidate. Persist the
evaluation report and approval recommendation to versioned S3 locations and CloudWatch logs.

### 4.3 SageMaker Model Registry

Create a SageMaker Model Package Group and register a real Model Package version referencing the
selected inference image and S3 model artifact. This is additional to the repository's lightweight
`model_registry/*.json` metadata. Capture at least:

- immutable S3 artifact reference and checksums;
- inference container image URI and preferably image digest;
- endpoint/task schema and supported task heads;
- validation and final test metrics with split provenance;
- model card and SageMaker evaluation locations;
- source revision, package versions, and approval status.

### 4.4 Approval workflow

Register new versions as `PendingManualApproval`. A reviewer checks scientific comparisons,
negative-transfer safeguards, untouched-test evidence, security, reproducibility, inference smoke
results, and cost before changing the version to `Approved`. Rejected versions remain traceable.
Only an approved model version may be deployed.

### 4.5 One SageMaker inference endpoint

Deploy exactly one cost-controlled SageMaker real-time endpoint from the approved Model Package
version. The inference contract should accept validated SMILES, canonicalize consistently with
training, reject invalid input clearly, select only registered task heads, and return prediction,
endpoint, and model-version metadata. Configure one initial production variant, conservative
instance count, health checks, logs, and a documented rollback path. Avoid additional endpoints
until traffic or isolation requirements justify them; delete the demonstration endpoint when it is
not needed.

Exit criterion: one approved, traceable SageMaker Model Package version backs one verified endpoint,
with S3 provenance and rollback instructions recorded.

## Phase 5: monitoring and drift

Monitoring begins with operationally simple, interpretable signals. CloudWatch should capture
service metrics and structured inference logs; scheduled custom analysis can calculate chemistry-
specific drift from privacy-safe aggregates or an approved sample store.

Monitor:

| Signal | Initial measure | Example action |
|---|---|---|
| Prediction distributions | Counts, quantiles, positive-rate or score histograms by endpoint | Investigate sustained shift from the approved reference window |
| Invalid SMILES rate | Invalid requests divided by total requests | Alert on a threshold or abrupt increase; inspect upstream validation |
| Inference latency | p50, p95, and p99 request latency plus errors/timeouts | Check load, payload size, instance health, and scaling needs |
| Model version | Model Package version and artifact identifier on every prediction/log | Detect unexpected versions and support rollback/audit |
| Descriptor drift | PSI or KS tests on selected RDKit descriptors such as molecular weight, logP, TPSA, and heavy-atom count | Review representativeness and retraining need |
| Fingerprint drift | Morgan-fingerprint bit-frequency PSI or distance/similarity to the training reference | Investigate chemistry-domain shift and out-of-domain traffic |

Reference distributions must come only from the selected model's training data; thresholds are set
using training/validation behavior, not test-set optimization. PSI and KS values are indicators,
not proof of model failure. For fingerprints, use a documented approximation such as bit-frequency
shift or a sampled Tanimoto similarity distribution because a raw KS test over sparse bit vectors
is not meaningful by itself.

Avoid logging raw SMILES by default. Prefer request IDs, validity flags, aggregate descriptors,
endpoint, latency, model version, and coarse prediction summaries. If raw inputs are ever retained,
define explicit authorization, encryption, access, retention, and deletion controls first.

Exit criterion: dashboards or reports expose the required signals, thresholds and ownership are
documented, and model-version traceability and a response playbook are verified.

## Phase 6: Kubernetes serving track

This is an inference portability demonstration, not the primary training platform. Do not assume
administrator rights on the work cluster and do not require Kubernetes training. Continue using the
work cluster primarily for authorized GPU experiments.

Use a local Kubernetes environment through Docker Desktop Kubernetes, `kind`, or a similar local
distribution. Build a FastAPI inference container that implements the same request validation,
canonicalization, task-head selection, response schema, and model-version reporting as the
SageMaker endpoint. For a public portfolio demonstration, use only a public-safe selected artifact
or a small synthetic/test artifact and do not embed credentials or private machine paths in the
image or manifests.

The initial Kubernetes manifests should include:

- a `Deployment` with a pinned image tag or digest and rolling-update behavior;
- a `Service` for stable in-cluster access;
- a `ConfigMap` for non-secret settings such as model identifier, task list, and logging level;
- readiness and liveness probes backed by distinct FastAPI health routes;
- CPU and memory resource requests and limits;
- a non-root container security context where the runtime permits it;
- environment-independent model location configuration;
- local smoke tests for valid input, invalid SMILES, unknown task, readiness, and version metadata.

Model binaries should not be stored in a ConfigMap. For local use, mount an approved artifact or
use a development-only image containing a small public-safe artifact. Secrets, if later required,
belong in a secret-management mechanism and never in Git.

EKS is optional later, after the local deployment is reproducible and there is a concrete need for
a second AWS serving platform. EKS, cluster GPU scheduling, operators, service meshes, autoscaling,
and Kubernetes-based training are outside the initial track.

Exit criterion: the FastAPI container runs locally and the Kubernetes Deployment becomes ready,
serves a versioned prediction through the Service, rejects invalid input, and recovers from a pod
restart within declared resources.

## Required implementation order

The following order is a gating sequence, not a set of parallel production launches:

```text
Current multi-task implementation
  -> lightweight GitHub Actions CI
  -> one-GPU float32 cluster smoke
  -> one-GPU mixed-precision smoke
  -> single-task baseline runs
  -> full multi-task training
  -> validation-based model selection
  -> untouched test evaluation
  -> S3 upload
  -> SageMaker evaluation
  -> Model Registry
  -> SageMaker endpoint
  -> monitoring/drift
  -> local Kubernetes inference deployment
```

Each gate must preserve the artifact and provenance contracts of the preceding gate. A failure
returns work to the relevant earlier phase without consulting the untouched test set for tuning.

## Technology mapping

| Technology | Role in this roadmap | Stage |
|---|---|---|
| Git/GitHub | Version source, configuration, tests, documentation, and infrastructure definitions; exclude secrets, large artifacts, data, and private paths | All stages |
| GitHub Actions | Fast `push` and `pull_request` CPU-only offline regression checks | Phase 1 |
| Python/RDKit | Data validation, SMILES canonicalization, descriptors, Morgan fingerprints, baselines, and chemistry drift features | Phases 1-6 |
| PyTorch/Transformers | Single-task and shared-encoder multi-task ChemBERTa training and inference | Phases 1-6 |
| Docker | Reproducible SageMaker Processing/inference and local FastAPI runtimes | Phases 4 and 6 |
| ECR | Versioned storage for approved AWS container images | Phase 4 |
| Terraform | Reviewable AWS foundation and later declarations for registry/serving/monitoring resources where appropriate | Phase 4 onward |
| S3 | Versioned selected artifacts, manifests, evaluation outputs, and approved monitoring references | Phases 4-5 |
| SageMaker | Managed evaluation Processing and AWS model lifecycle integration | Phase 4 |
| SageMaker Model Registry | Real Model Package Group/version, lineage metadata, and approval state | Phase 4 |
| SageMaker Endpoint | The single approved real-time AWS inference deployment | Phases 4-5 |
| FastAPI | Shared HTTP inference contract and health endpoints for the portable serving container | Phases 4 and 6 |
| Kubernetes | Local inference orchestration using Deployment, Service, ConfigMap, probes, and resource controls | Phase 6 |
| CloudWatch/custom drift analysis | Endpoint logs and latency/error metrics plus scheduled prediction and molecular drift checks | Phase 5 |
| pytest | Offline unit/contract tests, CPU smoke coverage, selection-isolation tests, and local inference verification | All stages |

## Delivery boundaries

Implementing this roadmap should remain incremental. No stage implicitly authorizes AWS resource
creation, Docker image publication, work-cluster administration, EKS creation, or production
traffic. Those actions require their own reviewed execution step. Documentation and dry-run plans
must use placeholders for account-specific values and public-safe path examples.

The roadmap is complete when the project can demonstrate a traceable chain from offline tests to
cluster evidence, validation-only selection, one untouched-test evaluation, one approved AWS model
version and endpoint, interpretable monitoring, and a separate local Kubernetes inference example—
without weakening the scientific or leakage controls defined by the existing multi-task plan.
