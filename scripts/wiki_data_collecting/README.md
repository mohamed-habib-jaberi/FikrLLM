# Pipeline `wiki_data_collecting`

This directory contains Wikipedia collection and augmentation scripts. It is used to build or enrich datasets before final cleaning.

## Goal

Create useful data for:

- pretraining;
- question/answer fine-tuning;
- Arabic-English bilingual enrichment;
- experiments on Egypt-related corpora.

## Files

| File | Role |
|---|---|
| `step05-collect-english-base-model-dataset.py` | Collects English articles from Wikipedia using existing inputs. |
| `step06-translate-finetune-dataset-to-english.py` | Translates fine-tuning Q/A pairs into English. |
| `step07-augment-english-dataset.py` | Augments the English dataset with additional Wikipedia sources. |

## Logical flow

```text
Wikipedia sources / initial datasets
        ↓
step05
        ↓
clean English articles
        ↓
step06
        ↓
translated Q/A pairs
        ↓
step07
        ↓
augmented English dataset
        ↓
clean_and_build_dataset.py
```

## Why is this directory separate?

These scripts perform collection and augmentation. They may:

- call external APIs;
- read and write JSONL files;
- take time;
- depend on an execution order;
- be rerun independently from model code.

They are therefore not part of `fikrllm/`, which contains model, data loading, training, and generation code.

## Good practices

- Keep intermediate outputs versioned or traceable.
- Avoid rerunning a full collection if an output file already exists.
- Document parameters used for each run.
- Add DVC later to track datasets and heavy artifacts.
