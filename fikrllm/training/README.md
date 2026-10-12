# Module `fikrllm.training`

This module contains the training system: loss, optimization loop, scheduler, checkpoints, and metrics.

## Overview

```text
DataLoader
   ↓
Trainer.train()
   ↓
train_step()
   ↓
GPT forward
   ↓
loss + backward
   ↓
AdamW step
   ↓
scheduler step
   ↓
tracking / evaluation / checkpoint
```

## Files

| File | Role |
|---|---|
| `trainer.py` | Main training and evaluation loop. |
| `schedule.py` | Learning-rate scheduler with warmup followed by cosine decay. |
| `checkpoint.py` | Saves and reloads weights, config, and training state. |
| `tracking.py` | Sends metrics and samples to TrackIO when enabled. |
| `freeze.py` | Freezes or unfreezes selected parts of the model. |
| `__init__.py` | Re-exports public training tools. |

## `trainer.py`

`Trainer` is the orchestrator. It receives:

- a `GPT` model;
- a training `DataLoader`;
- a `TrainingConfig`;
- optionally an evaluation `DataLoader`;
- optionally a tokenizer for sample generation during training.

A training step does:

```text
batch -> model -> logits -> loss -> backward -> gradient clipping -> optimizer step
```

Gradient accumulation simulates a larger batch:

```text
4 micro-batches of 8 examples = effective batch size of 32 examples
```

## `document_cross_entropy`

This loss ignores `[PAD]` tokens.

It also prevents long examples from dominating training: it averages loss per document first, then averages across documents in the batch.

## `schedule.py`

The scheduler has two phases:

```text
linear warmup -> cosine decay
```

Warmup avoids starting with a learning rate that is too aggressive. Cosine decay gradually lowers the learning rate during training.

## `checkpoint.py`

A checkpoint contains:

- model weights;
- model configuration;
- current step;
- metrics;
- optionally optimizer state;
- optionally scheduler state.

This makes it possible to resume training or load the model later for generation.

## `tracking.py`

The tracker centralizes logs:

- training loss;
- evaluation loss;
- gradient norm;
- generated samples;
- comparison tables.

If tracking is disabled, training still works normally.
