# Pipeline `ch03-build-tokenizer`

This directory contains the minimal pipeline that turns a Hugging Face dataset into files ready for training.

## Overview

```text
Hugging Face dataset
        ↓
01_clean_corpus.py
        ↓
tokenizer_corpus.parquet
        ↓
02_build_tokenizer.py
        ↓
fikrllm/assets/tokenizer.json
        ↓
03_pack_corpus.py
        ↓
train.bin / eval.bin / meta.json
```

## Files

| File | Role |
|---|---|
| `cleaning.py` | Type A cleaning and Type B normalization functions. |
| `01_clean_corpus.py` | Downloads shards, cleans tokenizer corpus, writes fine-tuning splits. |
| `02_build_tokenizer.py` | Trains the BPE tokenizer and saves `tokenizer.json`. |
| `03_pack_corpus.py` | Tokenizes documents and writes `train.bin`, `eval.bin`, `meta.json`. |
| `output/` | Local directory for generated artifacts. |

## Stage 1 - Cleaning

Quick command:

```bash
python scripts/ch03-build-tokenizer/01_clean_corpus.py \
  --max-shards 1 \
  --tokenizer-docs 1000
```

Outputs:

```text
output/tokenizer_corpus.parquet
output/ft_train_filtered.parquet
output/ft_eval_filtered.parquet
```

`tokenizer_corpus.parquet` trains the tokenizer. The `ft_*` files are used for chat fine-tuning.

## Stage 2 - Tokenizer

Command:

```bash
python scripts/ch03-build-tokenizer/02_build_tokenizer.py
```

The script reads:

```text
output/tokenizer_corpus.parquet
```

and writes:

```text
fikrllm/assets/tokenizer.json
```

This file becomes the source of truth for encoding and decoding.

## Stage 3 - Packing

Quick command:

```bash
python scripts/ch03-build-tokenizer/03_pack_corpus.py --max-shards 1
```

Outputs:

```text
output/train.bin
output/eval.bin
output/meta.json
```

Why pack?

Encoding text at every epoch would be slow. Packing encodes once, writes token IDs in binary form, and lets training read that stream efficiently.

## `cleaning.py`

There are two kinds of cleaning:

| Type | When? | Goal |
|---|---|---|
| Type A | Offline preparation | Remove noise, useless sections, duplicates, and very short texts. |
| Type B | Tokenizer and inference | Normalize text without strongly changing content. |

Type A is destructive and should not be applied directly to user prompts. Type B is lightweight and can be embedded in the tokenizer.
