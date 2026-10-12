"""
embeddings.py -- token IDs to the vectors the model actually reads
"""

import torch
import torch.nn as nn
from torch import Tensor

from fikrllm.config import ModelConfig


class InputEmbedding(nn.Module):

    # Initialize this object with the state it needs later.
    def __init__(self, config: ModelConfig) -> None:

        super().__init__()
        
        self.max_seq_len = config.max_seq_len
        self.dropout = nn.Dropout(config.dropout)

        self.token_embedding = nn.Embedding(
            config.vocab_size, config.d_model
        )

        self.position_embedding = nn.Embedding(
            config.max_seq_len, config.d_model
        )



    # Run the module forward pass.
    def forward(self, input_ids: Tensor, position_offset: int = 0) -> Tensor:

        _, seq_len = input_ids.shape
        if position_offset + seq_len > self.max_seq_len:
            raise ValueError(
                f"sequence length {position_offset + seq_len} exceeds "
                f"max_seq_len={self.max_seq_len}"
            )

        positions = torch.arange(
            position_offset, position_offset + seq_len, device=input_ids.device
        )

        tokens = self.token_embedding(input_ids)
        pos = self.position_embedding(positions)

        return self.dropout(tokens + pos)  # (B, seq_length, d_model)
