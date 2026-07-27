# Verified TDC Download Environment

Public TDC dataset acquisition is an optional CPU-side data workflow, separate from ChemBERTa
training. The verified Python distribution is:

```text
PyTDC==0.3.9
```

`PyTDC` is the installable distribution name. Its Python import namespace remains `tdc`, as in
`from tdc.single_pred import ADME, Tox`. Do not replace the distribution requirement with the
generic package name `tdc`.

This version successfully loaded the exact configured multi-task datasets `BBB_Martins`,
`hERG_Karim`, and `AMES`. The verified `hERG_Karim` dataset contains 13,445 records. TDC's smaller
`hERG` dataset is separate and is not used by this multi-task track.

## Environment boundary

Use a dedicated Python 3.11 Conda environment for public dataset acquisition when a fresh download
is required:

```powershell
conda create -n admet-tdc-download python=3.11
conda activate admet-tdc-download
python -m pip install -r requirements-tdc-download.txt
```

The main GPU training requirements and `sagemaker/requirements.txt` deliberately exclude PyTDC.
Training and coordinated splitting consume existing prepared CSV files and must not download TDC
data. The CPU SageMaker Processing dependency file includes the verified PyTDC pin because its
`tdc_download` mode may perform acquisition.

## Public dataset acquisition

From the repository root, the following commands acquire only the configured raw normalized data:

```powershell
python .\scripts\download_tdc_dataset.py `
  --config .\configs\bbb_martins.yaml `
  --output-csv .\outputs\local\multitask\raw\bbb_martins.csv `
  --summary-json .\outputs\local\multitask\raw\bbb_martins_summary.json

python .\scripts\download_tdc_dataset.py `
  --config .\configs\herg_karim.yaml `
  --output-csv .\outputs\local\multitask\raw\herg_karim.csv `
  --summary-json .\outputs\local\multitask\raw\herg_karim_summary.json

python .\scripts\download_tdc_dataset.py `
  --config .\configs\ames.yaml `
  --output-csv .\outputs\local\multitask\raw\ames.csv `
  --summary-json .\outputs\local\multitask\raw\ames_summary.json
```

Downloaded and generated dataset files remain ignored by Git. Record the endpoint config, exact
dataset name, PyTDC version, row count, and output hash with experiment provenance.

## Expanded classification acquisition

The candidate classification acquisition milestone adds seven unsplit ADME endpoints:
`HIA_Hou`, `Pgp_Broccatelli`, `CYP1A2_Veith`, `CYP2C19_Veith`, `CYP2C9_Veith`,
`CYP2D6_Veith`, and `CYP3A4_Veith`. PyTDC 0.3.9 normalizes these public display names to the
lowercase ADME registry keys. In particular, the verified name is `CYP1A2_Veith` (no underscore
between `CYP` and `1A2`), despite a spelling inconsistency on one TDC overview page. The acquisition
command verifies the installed PyTDC version and registry membership before loading data:

```powershell
python .\scripts\download_candidate_classification_datasets.py `
  --output-root .\outputs\local\classification_expansion\acquisition
```

This command calls the same unsplit TDC loader used by the existing endpoint downloader. It does
not call `get_split` and does not create train, validation, or test files. For each endpoint it
writes the row-preserving `normalized.csv`, `audit.json`, duplicate details, and separate
quarantine copies for invalid structures, missing or invalid labels, and canonical structures
having conflicting binary labels. No source row is silently discarded at acquisition time.
Everything beneath `outputs/` remains outside Git.

PyTDC 0.3.9 returns 1,218 source rows for `Pgp_Broccatelli`, whereas the TDC benchmark page reports
1,212. The acquisition audit identifies six repeated exact SMILES/label records, accounting for
that difference. Acquisition preserves all 1,218 source rows and reports the duplicates; later
split preparation must apply its explicit duplicate policy.

The label directions are frozen as follows:

- `HIA_Hou`: label 1 is favorable/good intestinal absorption (`%FA > 30`); label 0 is poor
  absorption (`%FA <= 30`). The Hou source paper defines those two classes.
- `Pgp_Broccatelli`: label 1 is P-glycoprotein inhibitor and label 0 is non-inhibitor. This is
  not a P-glycoprotein substrate endpoint.
- The five `CYP*_Veith` endpoints: label 1 is inhibitor and label 0 is non-inhibitor for the named
  CYP isozyme. These are distinct from the separately named Carbon-Mangels substrate datasets.

Sources:

- TDC ADME task and dataset definitions:
  https://tdcommons.ai/single_pred_tasks/adme/
- Hou et al., *Prediction of Oral Absorption by Correlation and Classification*:
  https://pubmed.ncbi.nlm.nih.gov/17238266/
- Broccatelli inhibitor/non-inhibitor dataset description:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3904775/
