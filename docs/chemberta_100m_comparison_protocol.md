# ChemBERTa-100M frozen comparison protocol

## Scope and status

The current `seyonec/ChemBERTa-zinc-base-v1` experiments remain the reference. The pinned
`DeepChem/ChemBERTa-100M-MLM` checkpoint is a candidate backbone for a controlled comparison; it
is not an automatic replacement. Phase 1 adds isolated experiment configurations and an encoder
compatibility preflight. It does not authorize training or test-set evaluation.

The candidate identity is frozen to:

- model: `DeepChem/ChemBERTa-100M-MLM`
- revision: `f5c45f44d3061f0346888f5c09db17ec1146d29d`

## Controlled variables

Both backbones use the same coordinated BBB_Martins, hERG_Karim, and AMES splits, the same linear
endpoint heads, masked-mean pooling, and identical optimization and training-control settings.
The candidate configurations differ from their reference configurations only in run name, model
name, and pinned model revision.

The frozen budgets are:

- single-task runs: 1,000 steps per endpoint;
- multi-task controlled run: 3,000 global round-robin steps;
- initial comparison seed: 42.

Checkpoint selection is validation-only. The primary multi-task selection metric is mean
validation ROC-AUC across the three endpoints. Mean validation PR-AUC is used only as a tie-breaker.
Single-task checkpoint selection uses endpoint validation ROC-AUC. Test metrics must never affect
checkpoint selection, early stopping, threshold tuning, hyperparameters, or backbone choice.

The coordinated test splits remain untouched until all validation-based selections are complete.
Final test evaluation is descriptive and occurs once for the already-selected checkpoints.

## Comparison outputs

The backbone comparison reports endpoint and aggregate ROC-AUC and PR-AUC, calibration, runtime,
peak GPU memory, serialized artifact size, and inference latency. Calibration reporting should use
prespecified measures such as Brier score and expected calibration error without test-driven
calibration fitting or threshold selection.

The seed-42 experiment is preliminary. Claims that one backbone is superior require both backbones
to be repeated with the same predefined seed set and evaluated under this unchanged protocol. No
test-driven hyperparameter selection or backbone selection is permitted.

## Compatibility preflight

The generic preflight loads the tokenizer and `AutoModel`, records Hugging Face loading information,
and exercises all three project heads. An MLM checkpoint may legitimately expose unused `lm_head.*`
keys when loaded through `AutoModel`. Missing shared-encoder weights, mismatched tensors, loading
errors, or unexpected non-MLM keys fail the preflight. No fallback checkpoint is substituted.
An optional model-created pooler may be reported as expected missing state because this project
uses token-level hidden states and its own masked-mean pooling rather than the encoder pooler.

Online verification:

```bash
python scripts/verify_encoder_compatibility.py \
  --model-name-or-path DeepChem/ChemBERTa-100M-MLM \
  --revision f5c45f44d3061f0346888f5c09db17ec1146d29d \
  --output-json outputs/local/preflight/chemberta_100m_compatibility.json
```

Offline verification after an explicit download:

```bash
python scripts/verify_encoder_compatibility.py \
  --model-name-or-path models/chemberta-100m-mlm \
  --local-files-only \
  --output-json outputs/local/preflight/chemberta_100m_offline_compatibility.json
```

The local model directory is expected to contain the complete Hugging Face snapshot:

```text
models/chemberta-100m-mlm/
├── config.json
├── model.safetensors or pytorch_model.bin
├── tokenizer_config.json
├── tokenizer.json and/or vocab/merges files
└── special_tokens_map.json (when supplied by the checkpoint)
```

The compatibility report records the resolved architecture, tokenizer identity, special-token IDs,
loading diagnostics, forward-pass shapes, finite-logit checks, and parameter counts.
