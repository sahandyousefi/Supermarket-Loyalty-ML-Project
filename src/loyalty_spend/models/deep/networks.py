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

from loyalty_spend.models.deep.tabular import embedding_dim


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


class EmbeddingMLP(nn.Module):
    """MLP over standardized numerics and learned category embeddings."""

    def __init__(
        self,
        n_numeric: int,
        cardinalities: list[int],
        width: int = 128,
        depth: int = 3,
        dropout: float = 0.1,
        n_outputs: int = 1,
        numeric_embedding: str | None = None,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(c, embedding_dim(c)) for c in cardinalities
        )
        self.numeric = None
        n_num = n_numeric
        if numeric_embedding == "plr":
            self.numeric = PeriodicEmbedding(n_numeric)
            n_num = n_numeric * self.numeric.d_out
        elif numeric_embedding is not None:
            raise ValueError("numeric_embedding must be None or 'plr'")
        n_in = n_num + sum(e.embedding_dim for e in self.embeddings)
        self.input = nn.Linear(n_in, width)
        self.blocks = nn.Sequential(
            *[ResidualBlock(width, dropout) for _ in range(depth)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(width), nn.GELU(), nn.Linear(width, n_outputs)
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        emb = [e(x_cat[:, i]) for i, e in enumerate(self.embeddings)]
        num = x_num if self.numeric is None else self.numeric(x_num)
        h = self.input(torch.cat([num, *emb], dim=1))
        return self.head(self.blocks(h))
