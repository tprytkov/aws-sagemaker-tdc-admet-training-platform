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
