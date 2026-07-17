# Multi-Task ChemBERTa: Local and One-GPU Execution

The three-classification trainer is step-based and supports CPU or one directly selected CUDA device. It does not use Accelerate, distributed training, SLURM, or SageMaker. Prepared BBB_Martins, hERG_Karim, and AMES split directories must already exist.

## Local CPU diagnostic

```powershell
conda activate admet-platform
python .\scripts\train_multitask.py `
  --config .\configs\multitask_classification.yaml `
  --prepared-root .\outputs\local\multitask\prepared `
  --output-dir .\outputs\local\multitask\cpu-diagnostic `
  --max-steps 12 `
  --evaluation-interval-steps 3 `
  --checkpoint-interval-steps 3 `
  --limit-samples-per-task 24 `
  --seed 42 `
  --device cpu `
  --mixed-precision no `
  --deterministic-algorithms
```

## One-GPU cluster smoke

```bash
conda activate admet-multitask
nvidia-smi
CUDA_VISIBLE_DEVICES=0 python scripts/train_multitask.py \
  --config configs/multitask_classification.yaml \
  --prepared-root outputs/multitask/prepared \
  --output-dir outputs/multitask/gpu-smoke \
  --max-steps 30 \
  --evaluation-interval-steps 10 \
  --checkpoint-interval-steps 10 \
  --limit-samples-per-task 100 \
  --seed 42 \
  --device cuda \
  --mixed-precision fp16
```

## One-GPU full configured run

Omit `--max-steps` to use `training.max_steps` from the YAML configuration.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_multitask.py \
  --config configs/multitask_classification.yaml \
  --prepared-root outputs/multitask/prepared \
  --output-dir outputs/multitask/classification-run-001 \
  --seed 42 \
  --device cuda \
  --mixed-precision fp16 \
  --classical-baseline-json outputs/baselines/classical.json \
  --single-task-baseline-json outputs/baselines/single-task.json
```

Full float32 (`--mixed-precision no`) is the reproducibility reference. FP16 and BF16 are optional CUDA execution modes and should be compared separately.

## Persistent execution

With `tmux`:

```bash
tmux new -s admet-multitask
CUDA_VISIBLE_DEVICES=0 python scripts/train_multitask.py \
  --config configs/multitask_classification.yaml \
  --prepared-root outputs/multitask/prepared \
  --output-dir outputs/multitask/classification-run-001 \
  --device cuda --mixed-precision fp16
```

Detach with `Ctrl-b d` and resume with `tmux attach -t admet-multitask`.

With `nohup`:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python scripts/train_multitask.py \
  --config configs/multitask_classification.yaml \
  --prepared-root outputs/multitask/prepared \
  --output-dir outputs/multitask/classification-run-001 \
  --device cuda --mixed-precision fp16 \
  > outputs/multitask/classification-run-001.log 2>&1 &
```

The environment manifest records the selected device, GPU name, CUDA version, precision mode, package versions, and peak allocated CUDA memory.
