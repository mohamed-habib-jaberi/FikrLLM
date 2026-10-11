"""
feedforward.py -- the position-wise feed-forward network
"""

import torch.nn as nn
from torch import Tensor

from fikrllm.config import ModelConfig


class FeedForward(nn.Module):
    """Position-wise MLP used inside every Transformer block.

    Each token is processed independently: its vector is expanded from
    ``d_model`` to ``d_ff``, transformed with GELU, then projected back to
    ``d_model`` so it can be added to the residual stream.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        d_ff = config.d_ff
        if d_ff is None:
            raise ValueError("ModelConfig must initialize d_ff before building the model")

        self.fc1 = nn.Linear(config.d_model, d_ff)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(d_ff, config.d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Transform a tensor shaped ``(batch, sequence, d_model)``."""
        x = self.fc1(x)
        x = self.act(x)
        return self.fc2(x)
