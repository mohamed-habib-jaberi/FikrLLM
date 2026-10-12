# Directory `fikrllm.assets`

This directory contains artifacts used by the `fikrllm` package.

## Files

| File / directory | Role |
|---|---|
| `tokenizer.json` | Official BPE tokenizer used by training, packing, notebooks, and generation. |
| `notebooks/` | Educational notebooks for understanding model components. |

## `tokenizer.json`

This file is critical: it defines the mapping between text and token IDs.

All modules must use the same tokenizer to avoid drift:

```text
text -> IDs during packing
text -> IDs during inference
IDs  -> text during generation
```

If this file changes, the corpus must be packed again because the IDs in `train.bin` and `eval.bin` depend on the vocabulary.

## Notebooks

| Notebook | Role |
|---|---|
| `tokenizer_embeddings.ipynb` | Explore the tokenizer, encoding, decoding, and embeddings. |
| `attention-step-by-step.ipynb` | Understand attention step by step. |
| `model-step-by-step.ipynb` | Understand the progressive construction of the model. |
| `chinchilla_llm_prompt.md` | Educational notes or prompts around scaling laws / LLMs. |

These notebooks are not required to run training. They are mainly used for learning and concept checks.
