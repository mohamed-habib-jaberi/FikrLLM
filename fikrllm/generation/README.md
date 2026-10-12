# Module `fikrllm.generation`

This module turns a trained model into a text generator.

## Overview

```text
text prompt
   ↓
Tokenizer.encode
   ↓
GPT
   ↓
logits for the last token
   ↓
sampling
   ↓
new token
   ↓
repeat until EOS or max_new_tokens
```

## Files

| File | Role |
|---|---|
| `generate.py` | Autoregressive generation loop. |
| `sampling.py` | Next-token selection strategies: greedy, temperature, top-k, top-p. |
| `__init__.py` | Re-exports public generation functions. |

## `generate.py`

`generate_ids()` works directly with token IDs.

At each step:

1. the model receives the context;
2. it produces logits;
3. the last-position logits are selected;
4. `sampling.py` chooses the next token;
5. that token is appended to the sequence;
6. the loop repeats.

`generate()` is the simpler interface:

```python
text = generate(model, tokenizer, "The Nile flows", config)
```

It encodes the prompt, calls `generate_ids()`, and decodes only the newly generated tokens.

## Chat mode

With `chat=True`, the prompt is formatted like fine-tuning data:

```text
[BOS] [SYS] system [USER] question [ASST]
```

This helps a fine-tuned model answer in the same format it saw during training.

## `sampling.py`

Main strategies:

| Strategy | Effect |
|---|---|
| `greedy` | Always chooses the most likely token. Stable, but sometimes repetitive. |
| `temperature` | Makes the distribution stricter or more creative. |
| `top_k` | Keeps only the `k` best candidates. |
| `top_p` | Keeps candidates that cover a given probability mass. |

Applied order:

```text
logits -> temperature -> top-k -> top-p -> softmax -> multinomial
```

This order matters: the distribution is adjusted first, filtered second, and sampled last.
