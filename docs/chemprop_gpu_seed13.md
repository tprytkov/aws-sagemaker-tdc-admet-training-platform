# Chemprop GPU seed-13 pilot

This handoff stages only code, configuration, split manifests, and prepared **train/validation**
CSVs. Locked-test CSVs, outputs, caches, credentials, unrelated documents, and model weights are
excluded.

## Prepare and transfer from Windows

```powershell
conda activate admet-chemprop
python .\scripts\prepare_chemprop_gpu_transfer.py `
  --destination .\outputs\chemprop\gpu-transfer\seed13
Get-FileHash .\outputs\chemprop\gpu-transfer\seed13\transfer_manifest.json -Algorithm SHA256
tar -czf .\outputs\chemprop\gpu-transfer\chemprop-seed13.tar.gz `
  -C .\outputs\chemprop\gpu-transfer\seed13 .
Get-FileHash .\outputs\chemprop\gpu-transfer\chemprop-seed13.tar.gz -Algorithm SHA256
scp .\outputs\chemprop\gpu-transfer\chemprop-seed13.tar.gz `
  GPU_USER@GPU_HOST:CHEMPROP_TRANSFER_ROOT/
```

Replace uppercase placeholders locally. Do not save hostnames, usernames, credentials, or private
machine paths in source-controlled files.

## Verify and create the Linux environment

```bash
set -euo pipefail
export CHEMPROP_WORK_ROOT="${CHEMPROP_WORK_ROOT:?set an authorized working directory}"
export CHEMPROP_TRANSFER_ROOT="${CHEMPROP_TRANSFER_ROOT:?set the upload directory}"
mkdir -p "${CHEMPROP_WORK_ROOT}"
tar -xzf "${CHEMPROP_TRANSFER_ROOT}/chemprop-seed13.tar.gz" -C "${CHEMPROP_WORK_ROOT}"
cd "${CHEMPROP_WORK_ROOT}"
conda env create -f environment-chemprop-gpu.yml
conda activate admet-chemprop
python scripts/prepare_chemprop_gpu_transfer.py \
  --verify-root . --manifest transfer_manifest.json
```

CUDA/PyTorch compatibility check:

```bash
nvidia-smi
python - <<'PY'
import torch
print({
    "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(), "device_count": torch.cuda.device_count(),
    "device_0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
})
assert torch.cuda.is_available(), "CUDA is unavailable to the pinned PyTorch environment"
x = torch.randn(1024, 1024, device="cuda")
assert torch.isfinite((x @ x.T).mean())
print("CUDA_TORCH_COMPATIBILITY_OK")
PY
```

Stop if CUDA is unavailable or incompatible; do not accidentally run a CPU pilot.

## Short GPU compatibility smoke

```bash
export CUDA_VISIBLE_DEVICES=0
python scripts/run_chemprop_smoke.py \
  --task multitask_regression \
  --output-dir outputs/gpu/compatibility/multitask_regression_seed13 \
  --seed 13 --accelerator cuda
python scripts/run_chemprop_smoke.py \
  --task binary_classification \
  --output-dir outputs/gpu/compatibility/bbb_seed13 \
  --seed 13 --accelerator cuda
```

Inspect each `run_summary.json` and Lightning `metrics.csv` for device, runtime, peak CUDA memory,
finite loss, and a valid checkpoint before continuing.

## Seed-13 train/validation pilots

```bash
export CUDA_VISIBLE_DEVICES=0
python scripts/run_chemprop_experiment.py \
  --config configs/chemprop/multitask_admet_regression.yaml \
  --output-dir outputs/gpu/pilot/multitask_regression_seed13 \
  --seed 13 --accelerator cuda
python scripts/run_chemprop_experiment.py \
  --config configs/chemprop/bbb_martins.yaml \
  --output-dir outputs/gpu/pilot/bbb_martins_seed13 \
  --seed 13 --accelerator cuda
```

These jobs load train and validation only. Do not add or mount locked-test CSVs. Seeds 37, 73, 101,
and 137 remain deferred until seed 13 learning curves, endpoint metrics, missing-label behavior,
runtime, peak memory, and checkpoints are reviewed.

## Package and retrieve artifacts

On Linux:

```bash
cd "${CHEMPROP_WORK_ROOT}"
find outputs/gpu/pilot -type f -print0 | sort -z | xargs -0 sha256sum \
  > outputs/gpu/pilot/SHA256SUMS
sha256sum -c outputs/gpu/pilot/SHA256SUMS
tar -czf chemprop-seed13-results.tar.gz outputs/gpu/pilot
sha256sum chemprop-seed13-results.tar.gz > chemprop-seed13-results.tar.gz.sha256
```

On Windows:

```powershell
scp GPU_USER@GPU_HOST:CHEMPROP_WORK_ROOT/chemprop-seed13-results.tar.gz `
  .\outputs\chemprop\retrieved\
scp GPU_USER@GPU_HOST:CHEMPROP_WORK_ROOT/chemprop-seed13-results.tar.gz.sha256 `
  .\outputs\chemprop\retrieved\
Get-FileHash .\outputs\chemprop\retrieved\chemprop-seed13-results.tar.gz -Algorithm SHA256
```

Compare the local SHA-256 with the retrieved `.sha256` file before extracting. Keep all retrieved
checkpoints, predictions, and logs under ignored output paths.
