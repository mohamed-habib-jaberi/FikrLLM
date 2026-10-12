# Directory `scripts`

This directory contains data-preparation scripts. They are separated from the `fikrllm/` package because they build the artifacts required by the model; they are not imported during inference.

## Main subdirectories

| Directory / file | Role |
|---|---|
| `ch01-pytorch-intro/` | Introductory PyTorch notebook. |
| `ch03-build-tokenizer/` | Minimal pipeline: clean data, train tokenizer, pack corpus. |
| `wiki_data_collecting/` | Wikipedia data collection and augmentation. |
| `clean_and_build_dataset.py` | Merges, cleans, and normalizes collected datasets. |
| `assets/` | Templates or resources used by some scripts. |

## Relationship with `fikrllm/`

```text
scripts/
   ↓ produces
tokenizer.json + train.bin + eval.bin + fine-tuning files
   ↓ used by
fikrllm/
```

## When to use these scripts

- At the beginning of the project to create local datasets.
- When the source dataset changes.
- When the tokenizer must be rebuilt.
- When the corpus must be repacked with a new tokenizer version.

## Notes

Some scripts may download several GB of data or call external services. For quick tests, use options such as `--max-shards`, `--limit`, or small input files when available.
