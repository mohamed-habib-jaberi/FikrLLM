# Chapter `ch01-pytorch-intro`

This directory contains the PyTorch introduction for the project.

## Goal

Before building a GPT, it helps to understand the basic building blocks:

- tensors;
- vectorized operations;
- gradients;
- PyTorch modules;
- loss functions;
- optimizers;
- a simple training loop.

## File

| File | Role |
|---|---|
| `pytorch_101.ipynb` | Educational notebook for practicing PyTorch basics. |

## Place in the project

```text
PyTorch basics
      ↓
embeddings
      ↓
attention
      ↓
Transformer block
      ↓
complete GPT
```

This chapter is not required to execute the pipeline, but it helps explain why modules in `fikrllm/model/` are written the way they are.
