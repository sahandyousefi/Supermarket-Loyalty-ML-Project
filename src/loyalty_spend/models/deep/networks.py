"""Network architectures for tabular regression.

``EmbeddingMLP``
    Categorical entity embeddings (Guo and Berkhahn, 2016) concatenated with
    the numerics and passed through residual MLP blocks.
``EmbeddingMLP(numeric_embedding="plr")``
    The same MLP with periodic embeddings of the numerics followed by a
    linear layer and ReLU (PLR; Gorishniy et al., 2022), which lets the
    network represent sharp and oscillating effects of a numeric feature.
``FTTransformer``
    Feature Tokenizer Transformer (Gorishniy et al., 2021): every feature
    becomes a token, a [CLS] token attends over them, and its final state
    gives the prediction.

Both return ``(batch, n_outputs)``. With two outputs the second is the log
variance of a Gaussian, used by deep ensembles.
"""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)
