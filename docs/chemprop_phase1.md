# Internal Chemprop: multitask regression and BBB

This subsystem contains exactly two randomly initialized, graph-only Chemprop 2.3.1 models:

1. one shared D-MPNN encoder with five continuous regression outputs; and
2. one independently initialized, single-output `BBB_Martins` binary classifier.

It contains no ten-head Chemprop classifier, no other Chemprop classification endpoints, and no
mixed classification/regression checkpoint. It does not replace any frozen ChemBERTa or existing
BBB provider.

## Frozen regression task contract

The task names and metadata are copied without renaming from `configs/multitask_regression.yaml`
and its coordinated split manifest.

| Endpoint | TDC dataset | Units | Transform | Train | Validation | Locked test |
|---|---|---|---|---:|---:|---:|
| `caco2_wang` | `Caco2_Wang` | log10(Papp [cm/s]) | identity | 713 | 89 | 100 |
| `lipophilicity_astrazeneca` | `Lipophilicity_AstraZeneca` | log-ratio | identity | 3,390 | 399 | 411 |
| `solubility_aqsoldb` | `Solubility_AqSolDB` | log mol/L | identity | 6,885 | 1,787 | 1,308 |
| `ppbr_az` | `PPBR_AZ` | percent bound | identity | 1,296 | 158 | 160 |
| `vdss_lombardo` | `VDss_Lombardo` | L/kg | log10 | 877 | 108 | 109 |

The five endpoint CSVs are outer-joined by canonical SMILES within each split. `nan` targets are
passed to Chemprop as missing labels, and Chemprop's target mask excludes those cells from loss and
metrics. Each scientific transform is applied first; a distinct mean and standard deviation is then
fitted from that endpoint's training labels only. Predictions are inverse-normalized and
inverse-transformed before per-endpoint metrics are calculated in the units above.

Task loss weight for endpoint `i` is proportional to `1 / n_train_i`, normalized so the five weights
have mean one. Therefore `weight_i * n_train_i` is constant, giving each endpoint equal aggregate
loss influence rather than allowing label-rich solubility to dominate. The resolved counts and
weights are saved with every run. Chemprop 2.3.1 reduces the task-weighted sum over all observed
target cells by the total observed-cell count, so inverse-count weights give equal aggregate
endpoint influence. Checkpoint selection and early stopping instead minimize the equal-weight mean
of the five per-endpoint MAEs in standardized target space. Any hyperparameter search must use this
validation-only selection metric.

## BBB contract

BBB uses the expanded frozen ten-head ChemBERTa coordinated membership: 1,561 train, 196 validation,
and 208 locked-test molecules. Class `1` means the dataset's BBB-permeable class. Platt calibration
and the maximum-MCC threshold are fitted on validation predictions only; threshold 0.5 is also
reported. Evidence remains `experimental_low_confidence`. Results must never be called safe or
unsafe, and no current predictor is replaced before approved locked-test evaluation.

## Environment

```powershell
conda env create -f environment-chemprop.yml
conda activate admet-chemprop
```

The environment intentionally excludes PyTDC and Transformers. Training consumes already prepared,
hashed files and performs no implicit download.

## Local synthetic CPU smoke tests

```powershell
python .\scripts\run_chemprop_smoke.py --task multitask_regression `
  --output-dir .\outputs\chemprop\smoke\multitask-regression --seed 13
python .\scripts\run_chemprop_smoke.py --task binary_classification `
  --output-dir .\outputs\chemprop\smoke\bbb --seed 13
```

The regression fixture includes missing labels on every head and treats solubility as the initially
inspected smoke endpoint, while still constructing the final five-output architecture.

## GPU-server run structure (not executed locally)

The immediate transfer and execution procedure is limited to seed 13 and is documented in
`docs/chemprop_gpu_seed13.md`. Seeds 37, 73, 101, and 137 remain deferred pending review.

Use the verified staging package described in the seed-13 document. The pilot commands are:

```bash
python scripts/run_chemprop_experiment.py \
  --config configs/chemprop/multitask_admet_regression.yaml \
  --output-dir outputs/gpu/pilot/multitask_regression_seed13 \
  --seed 13 --accelerator cuda
python scripts/run_chemprop_experiment.py \
  --config configs/chemprop/bbb_martins.yaml \
  --output-dir outputs/gpu/pilot/bbb_martins_seed13 \
  --seed 13 --accelerator cuda
```

A SageMaker structure should use one job per model/seed, the same immutable prepared-data channel,
separate output prefixes, and no test-data channel during training or selection. Locked-test access
belongs to a later authenticated evaluation job after configurations, scalers, calibration, and
threshold rules are frozen.

## Staged artifacts

The regression export contains one checkpoint explicitly marked as five regression outputs and no
classification outputs, a scaler document with five endpoint entries, task weights, per-endpoint
metadata and original-unit predictions, applicability-domain files, ensemble mean/seed-standard-
deviation files, and SHA-256 hashes. Seed standard deviation is variability across initializations,
not a calibrated confidence interval.

BBB exports a separate checkpoint, uncalibrated and calibrated probabilities, Platt calibrator,
both threshold reports, applicability/uncertainty outputs, manifest identity, and hashes. Generated
artifacts remain under ignored staging/output directories and are not copied into MolOptima.
