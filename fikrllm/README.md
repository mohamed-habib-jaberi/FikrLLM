# Package `fikrllm`

This directory contains the main Python package for the project. It turns artifacts prepared by `scripts/` into a trainable GPT model that can generate text or be adapted to chat format.

## Package role

```text
tokenizer.json + train.bin / eval.bin
        ↓
fikrllm/tokenizer.py
        ↓
fikrllm/data/
        ↓
fikrllm/model/
        ↓
fikrllm/training/
        ↓
fikrllm/generation/
```

## Modules

| Module | Role |
|---|---|
| `config.py` | Defines model, training, and generation hyperparameters. |
| `tokenizer.py` | Loads the BPE tokenizer and exposes `encode`, `decode`, and `tokenize`. |
| `data/` | Converts packed tokens or Q/A pairs into PyTorch batches. |
| `model/` | Contains the GPT architecture: embeddings, attention, Transformer blocks, LoRA. |
| `training/` | Contains the training loop, checkpoints, scheduler, and tracking. |
| `generation/` | Generates text from a trained checkpoint. |
| `assets/` | Stores the official tokenizer used by the whole project. |

## Main idea

The package does not download data. It assumes the preparation steps have already produced:

- `fikrllm/assets/tokenizer.json`
- `scripts/ch03-build-tokenizer/output/train.bin`
- `scripts/ch03-build-tokenizer/output/eval.bin`
- `scripts/ch03-build-tokenizer/output/meta.json`

Then `run_pretrain.py` and `run_finetune_lora.py` assemble the modules for training or adaptation.
