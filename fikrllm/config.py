"""Source of truth for FikrLLM configuration values.

This file separates three different concerns:

``ModelConfig``
    What is the shape of the network? These values must match a checkpoint
    exactly: vocabulary, width, number of blocks, heads, and context length.
``GenerationConfig``
    How should the next tokens be chosen from model logits?
``TrainingConfig``
    How should weights be optimized, evaluated, tracked, and saved?

This separation makes it possible to change the learning rate or sampling
strategy without changing the architecture stored in a checkpoint.
"""

import math
import random
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ModelConfig:
    """Describe every dimension that determines the GPT shape.

    Changing one of these fields usually changes the size or structure of the
    model tensors. A checkpoint can therefore only be loaded into a ``GPT``
    built with the same configuration.
    """

    vocab_size: int = 32_000  # Number of possible tokenizer outputs.
    d_model: int = 768  # Width of one token vector inside the network.
    d_ff: int | None = None  # Internal MLP width; defaults to 4 * d_model, like GPT-2.
    max_seq_len: int = 1024  # Maximum number of tokens in the context.
    dropout: float = 0.1  # Regularization applied during training.
    n_layers: int = 12  # Number of stacked Transformer blocks.

    n_heads: int = 12  # Number of parallel attention views.
    qkv_bias: bool = False  # Whether to add bias to Q/K/V projections.
    pad_id: int = 0  # Target ignored by the loss.

    tie_weights: bool = True  # Share input embeddings and output projection.

    # Purpose: post init.
    def __post_init__(self) -> None:
        """Fill derived values and reject impossible tensor shapes."""
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model

        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} is not divisible by n_heads={self.n_heads}"
            )

    # Purpose: d k.
    @property
    def d_k(self) -> int:
        """Width of one head's Query, Key, and Value vectors."""
        return self.d_model // self.n_heads

    # Purpose: fikrllm.
    @classmethod
    def fikrllm(cls) -> "ModelConfig":
        """Main project configuration: 8 blocks with width 512."""
        return cls(d_model=512, n_heads=8, n_layers=8, dropout=0.05)

    # Purpose: tiny.
    @classmethod
    def tiny(cls) -> "ModelConfig":
        """Small model for quick tests and machines without a GPU."""
        return cls(d_model=128, n_heads=4, n_layers=2, max_seq_len=256)


@dataclass
class GenerationConfig:
    """Control how logits are transformed into text.

    Sampling is the step where the model can become non-deterministic. The
    settings below control the tradeoff between coherence and diversity.
    """

    max_new_tokens: int = 100  # Limit the continuation length.
    temperature: float = 0.8  # 0 = greedy; < 1 safer; > 1 more varied.

    top_k: int | None = 50  # Keep only the k best candidates.
    top_p: float | None = 0.95  # Nucleus sampling: maximum cumulative mass.

    use_cache: bool = True  # Reuse Key/Value tensors from previous tokens.
    seed: int | None = None  # Make sampling reproducible.

    # Choose the highest-scoring token from model logits.
    @classmethod
    def greedy(cls) -> "GenerationConfig":
        """Always select the most likely token."""
        return cls(temperature=0.0, top_k=None, top_p=None)


@dataclass
class TrainingConfig:
    """Control optimization without changing the model shape.

    Unlike ``ModelConfig``, these values can change when the same weights are
    resumed on a new corpus or adapted during fine-tuning.
    """

    # AdamW optimizer and gradient stability.
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0

    # Run length and learning-rate schedule.
    max_steps: int = 10_000
    warmup_steps: int = 500

    min_lr_ratio: float = 0.1

    # Effective batch = batch_size * accumulation_steps.
    batch_size: int = 24
    accumulation_steps: int = 1

    # Tracking, evaluation, and checkpoint frequency.
    log_every: int = 10
    print_every: int = 0
    eval_every: int = 500
    eval_batches: int = 50

    save_every: int = 1_000

    project: str = "fikrllm"
    run_name: str = "train-phase"

    checkpoint_dir: str = "checkpoints"
    seed: int = 0

    sample_every: int = 500
    sample_chat: bool = False  # Use the chat template during fine-tuning.
    sample_max_new_tokens: int = 60
    sample_prompts: tuple[str, ...] = (
        "وُلد في مدينة بغداد عام",  # biography
        "The early life of",  # biography
        "تقع هذه المدينة في شمال",  # geography
        "This city is located on the eastern coast of",  # geography
        "شهدت الحرب العالمية الثانية",  # history
        "During the nineteenth century,",  # history
        "يُعرف هذا الفنان بأعماله في مجال",  # arts
        "The film was directed by",  # arts
        "فاز الفريق بالبطولة بعد",  # sports
        "The scientific study found that",  # science
    )


# Build a readable run name with a timestamp suffix.
def unique_run_name(base: str) -> str:
    """Create a readable run name that is unique enough for TrackIO."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"run-{random.randint(0, 9999):04d}-{base}-{stamp}"


# Convert a requested number of epochs into optimizer steps.
def steps_for_epochs(
    num_examples: int,
    batch_size: int,
    accumulation_steps: int,
    epochs: float,
) -> int:
    """Convert an epoch count into actual optimizer updates."""
    effective_batch = batch_size * accumulation_steps
    steps_per_epoch = math.ceil(num_examples / effective_batch)
    return max(1, math.ceil(epochs * steps_per_epoch))
