# Module `fikrllm.data`

This module prepares data immediately before it enters the model. It does not download raw data and does not clean the raw corpus. Its role starts after the packing step.

## Place in the pipeline

```text
train.bin / eval.bin
        ↓
PackedDataset
        ↓
DataLoader
        ↓
collate_fn
        ↓
batch {"input_ids", "targets"}
        ↓
GPT
```

## Files

| File | Role |
|---|---|
| `dataset.py` | Reads `train.bin` / `eval.bin` with `numpy.memmap` and creates token windows. |
| `collate.py` | Groups examples into batches and creates the next-token prediction task. |
| `chat.py` | Converts question/answer pairs into chat fine-tuning examples. |
| `__init__.py` | Re-exports useful classes and functions from the module. |

## `dataset.py`

`PackedDataset` reads a binary token file without loading the full file into memory.

Conceptual example:

```text
train.bin = [BOS, t1, t2, t3, EOS, BOS, t4, t5, EOS, ...]
```

The dataset cuts this stream into windows of size `max_seq_len + 1`.

Why `+ 1`?

Because `collate.py` then creates:

```text
input_ids = all tokens except the last one
targets   = all tokens except the first one
```

## `collate.py`

`collate_fn` receives several examples and groups them into one batch.

Example:

```text
example A : [BOS, Paris, is, beautiful, EOS]
example B : [BOS, GPT, learns, EOS]
```

After padding:

```text
A : [BOS, Paris, is, beautiful, EOS]
B : [BOS, GPT, learns, EOS, PAD]
```

After shifting:

```text
input_ids:
A : [BOS, Paris, is, beautiful]
B : [BOS, GPT, learns, EOS]

targets:
A : [Paris, is, beautiful, EOS]
B : [GPT, learns, EOS, PAD]
```

The model therefore learns to predict the next token.

## `chat.py`

`chat.py` is used for instruction/chat fine-tuning. It transforms:

```text
question + answer
```

into:

```text
[BOS] [SYS] system prompt [USER] question [ASST] answer [EOS]
```

The important part: the loss ignores the question and the system prompt. The model is graded only on the answer.

```text
prompt tokens  -> ignored
answer tokens  -> used for training
```

This avoids training the model to copy user questions.
