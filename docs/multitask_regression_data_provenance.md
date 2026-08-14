# Multi-Task Regression Data Provenance

The initial coordinated regression track is frozen to the public PyTDC 0.3.9 datasets
`Caco2_Wang`, `Lipophilicity_AstraZeneca`, `Solubility_AqSolDB`, `PPBR_AZ`, and
`VDss_Lombardo`.

## Frozen target handling

| Dataset | Scientific transform | Normalization |
|---|---|---|
| Caco2_Wang | identity | train-only z-score |
| Lipophilicity_AstraZeneca | identity | train-only z-score |
| Solubility_AqSolDB | identity | train-only z-score |
| PPBR_AZ | identity | train-only z-score |
| VDss_Lombardo | log10 | train-only z-score |

The versioned Caco-2 label contract is `caco2_wang_log10_cm_per_s_v1`. TDC displays the
underlying physical unit as cm/s, while the distributed values (−7.7600002 to −3.51) and
source convention represent `log10(Papp [cm/s])`. The stored label and model output unit
is therefore `log10(Papp [cm/s])`; no additional logarithm is applied during training.
Convert a reported stored/model value `y` to physical permeability with
`Papp [cm/s] = 10**y`, and convert a strictly positive physical value `p` back with
`y = log10(p)`. Existing labels are unchanged by this metadata clarification.

The PPBR source is exactly the 1,614 rows returned by the installed PyTDC 0.3.9
environment. A separate TDC benchmark view reports 1,797 rows. This project does not
merge, supplement, or silently alter the PyTDC 0.3.9 dataset to match that separate
count.

## Split and lock policy

The source rows are coordinated globally with seed 42. Exact canonical structures and
Murcko scaffold groups cannot cross train, validation, and test, including across
endpoints. Conflicting continuous-label duplicate groups are quarantined rather than
averaged. Identical-label duplicates are collapsed deterministically.

Transformation statistics are fitted from train only. Validation may be used for
plumbing checks and future checkpoint selection, but not to fit transforms. Once the
coordinated files are generated, every test file is locked by SHA-256 for future final
evaluation and must not be opened by training or preflight workflows.

Generated local artifacts include:

- `outputs/local/multitask_regression/coordinated/dataset_audit.json`
- `outputs/local/multitask_regression/coordinated/dataset_audit.md`
- `outputs/local/multitask_regression/coordinated/target_transforms.json`
- `outputs/local/multitask_regression/coordinated/LOCKED_TEST_SPLITS.json`

Generated datasets, model checkpoints, and predictions remain local artifacts and are
not committed to the repository.
