# Directory `tests`

This directory contains automated checks for the core behavior of FikrLLM.

## Goal

The tests protect the most important contracts in the project:

- model configuration validity;
- tokenizer special tokens;
- batch creation and next-token shifting;
- GPT forward pass shapes and loss computation;
- LoRA merge behavior.

## File

| File | Role |
|---|---|
| `test_core.py` | Core unit tests for tokenizer, data batching, model forward pass, and LoRA. |

## How to run

```bash
python -m pytest
```

These tests are intentionally small. They are designed to catch broken interfaces quickly, not to validate model quality.
