# FikrLLM Technical Notes

This file summarizes the main technical ideas to remember from the FikrLLM project.

## 1. Big picture

FikrLLM is a bilingual Arabic-English GPT built from scratch with PyTorch.

The full pipeline is:

```text
raw dataset
  -> cleaning
  -> tokenizer training
  -> tokenization
  -> binary packing
  -> Dataset + DataLoader
  -> GPT pretraining
  -> checkpoint
  -> generation
  -> LoRA chat fine-tuning
```

The key idea is that an LLM does not train directly on raw text. Text is converted into token IDs, and the model learns to predict the next token.

## 2. Dataset and cleaning

The raw dataset comes from Hugging Face as Parquet files.

Important files:

| File | Role |
|---|---|
| `01_clean_corpus.py` | Downloads dataset shards and prepares local corpus files. |
| `cleaning.py` | Cleans noisy text and normalizes Arabic/English text. |
| `03_pack_corpus.py` | Converts cleaned text into binary token files. |

There are two cleaning levels:

| Type | Purpose |
|---|---|
| Type A cleaning | Offline destructive cleaning: remove noisy sections, references, categories, duplicates. |
| Type B normalization | Lightweight normalization used by the tokenizer and inference. |

Important distinction:

```text
Type A = for dataset preparation only
Type B = safe for tokenizer and inference
```

## 3. Tokenizer

The tokenizer converts text into token IDs.

Important files:

| File | Role |
|---|---|
| `02_build_tokenizer.py` | Trains the BPE tokenizer. |
| `fikrllm/tokenizer.py` | Loads `tokenizer.json` and exposes `encode`, `decode`, `tokenize`. |
| `fikrllm/assets/tokenizer.json` | Official tokenizer used by training and inference. |

Special tokens:

| Token | Meaning |
|---|---|
| `[PAD]` | Padding token ignored by the loss. |
| `[UNK]` | Unknown token. |
| `[BOS]` | Beginning of sequence. |
| `[EOS]` | End of sequence. |
| `[SYS]` | System message role. |
| `[USER]` | User message role. |
| `[ASST]` | Assistant message role. |

If `tokenizer.json` changes, `train.bin` and `eval.bin` must be rebuilt because token IDs may change.

## 4. Binary packing

Packing means encoding the corpus once and saving token IDs in compact binary files.

Files produced:

```text
train.bin
eval.bin
meta.json
```

Why packing matters:

- avoids tokenizing text again at every epoch;
- makes training faster;
- stores tokens compactly with `uint16`;
- keeps metadata in `meta.json`.

Example:

```text
document A: [BOS] a b c [EOS]
document B: [BOS] d e   [EOS]
train.bin : [BOS] a b c [EOS] [BOS] d e [EOS] ...
```

## 5. Dataset and collate

`PackedDataset` reads `train.bin` / `eval.bin` using `numpy.memmap`.

This keeps the large binary file on disk instead of loading everything into RAM.

`collate_fn` creates the next-token prediction task:

```text
window : [BOS, t1, t2, t3, EOS]
input  : [BOS, t1, t2, t3]
target : [t1,  t2, t3, EOS]
```

The model receives `input_ids` and learns to predict `targets`.

Padding is ignored by the loss through `ignore_index=pad_id`.

## 6. GPT model

Main files:

| File | Role |
|---|---|
| `embeddings.py` | Converts token IDs and positions into vectors. |
| `attention.py` | Implements causal multi-head self-attention. |
| `feedforward.py` | Applies the MLP inside each Transformer block. |
| `block.py` | Combines LayerNorm, attention, residuals, and MLP. |
| `gpt.py` | Assembles the full GPT model. |
| `lora.py` | Adds LoRA adapters for fine-tuning. |

Main architecture:

```text
input_ids
  -> token + position embeddings
  -> Transformer blocks
  -> final LayerNorm
  -> lm_head
  -> logits
```

The model outputs `logits`, not probabilities. Softmax is applied later when computing probabilities or sampling.

## 7. Causal self-attention

Causal attention prevents a token from seeing future tokens.

Example:

```text
position 0 can see: 0
position 1 can see: 0, 1
position 2 can see: 0, 1, 2
```

This is essential for next-token prediction. If the model could see future tokens, it would cheat during training.

Multi-head attention means several attention patterns are learned in parallel.

## 8. Training

Important files:

| File | Role |
|---|---|
| `run_pretrain.py` | Entry point for pretraining. |
| `trainer.py` | Training loop. |
| `schedule.py` | Learning-rate schedule. |
| `checkpoint.py` | Save/load model checkpoints. |
| `tracking.py` | Log metrics and generated samples. |

One training step:

```text
batch
  -> model forward
  -> logits
  -> cross-entropy loss
  -> backward
  -> gradient clipping
  -> AdamW step
  -> scheduler step
```

Gradient accumulation simulates a larger batch:

```text
batch_size 8 * accumulation_steps 4 = effective batch size 32
```

## 9. Checkpoint

A checkpoint is a saved state of the training run.

It can contain:

- model weights;
- model config;
- current step;
- optimizer state;
- scheduler state;
- metrics.

It is used to:

- resume training;
- load a trained model for generation;
- start LoRA fine-tuning from a pretrained model.

## 10. Generation

Generation is autoregressive:

```text
prompt
  -> model predicts next token
  -> append token
  -> repeat
```

Sampling strategies:

| Strategy | Meaning |
|---|---|
| `greedy` | Always choose the highest-scoring token. |
| `temperature` | Controls randomness. |
| `top_k` | Keeps only the best `k` candidates. |
| `top_p` | Keeps candidates covering a probability mass. |

KV cache makes generation faster by reusing previous attention keys and values.

## 11. LoRA fine-tuning

LoRA is a parameter-efficient fine-tuning technique.

Instead of updating all weights, it trains small adapter matrices.

Concept:

```text
frozen base model + trainable LoRA adapters
```

Benefits:

- fewer trainable parameters;
- lower memory usage;
- faster fine-tuning;
- easier adaptation to chat format.

In FikrLLM, LoRA is applied to attention projections such as `W_q`, `W_k`, `W_v`, and `W_o`.

## 12. What to remember

- GPT learns next-token prediction.
- The tokenizer is part of the model contract.
- `tokenizer.json`, `train.bin`, and checkpoints must stay compatible.
- `collate_fn` creates `input_ids` and `targets`.
- The causal mask prevents cheating.
- Checkpoints allow resume, generation, and fine-tuning.
- LoRA adapts the model without retraining all weights.
- Clean data and reproducible pipelines matter as much as model code.
