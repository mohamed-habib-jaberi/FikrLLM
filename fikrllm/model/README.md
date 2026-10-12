# Module `fikrllm.model`

This module contains the GPT architecture. Token IDs become vectors, pass through Transformer blocks, and are projected back into vocabulary scores.

## Overview

```text
input_ids
   ↓
InputEmbedding
   ↓
TransformerBlock x N
   ↓
final LayerNorm
   ↓
lm_head
   ↓
logits
```

`logits` are scores: for every position, the model scores every possible next token in the vocabulary.

## Files

| File | Role |
|---|---|
| `embeddings.py` | Adds token embeddings and position embeddings. |
| `attention.py` | Implements causal multi-head attention with KV-cache support. |
| `feedforward.py` | Applies the Transformer MLP at each position. |
| `block.py` | Combines LayerNorm, attention, residual paths, MLP, and residual paths. |
| `gpt.py` | Assembles the full GPT model and optionally computes loss. |
| `lora.py` | Adds LoRA adapters for parameter-efficient fine-tuning. |
| `__init__.py` | Re-exports public model components. |

## `embeddings.py`

The model does not understand raw IDs directly:

```text
[12, 91, 305]
```

It turns them into vectors:

```text
(batch, sequence_length) -> (batch, sequence_length, d_model)
```

It also adds position information because a Transformer does not naturally know token order.

## `attention.py`

Attention allows each token to look at previous tokens.

The causal mask prevents looking into the future:

```text
token 1 can see: token 1
token 2 can see: token 1, token 2
token 3 can see: token 1, token 2, token 3
```

During generation, the KV cache avoids recomputing keys and values for all previous tokens.

## `block.py`

A Transformer block contains:

```text
LayerNorm
  ↓
Attention
  ↓
Residual connection
  ↓
LayerNorm
  ↓
FeedForward
  ↓
Residual connection
```

Residual connections stabilize training and help gradients flow through the network.

## `gpt.py`

`GPT` assembles:

- embeddings;
- several Transformer blocks;
- a final LayerNorm;
- a prediction head called `lm_head`.

If `targets` are provided, the model also computes the next-token prediction loss.

## `lora.py`

LoRA adds a small trainable detour to selected linear layers.

Instead of training all model parameters:

```text
frozen base model + small trainable adapters
```

This is useful for adapting the model to chat with much less memory and fewer trainable parameters.
