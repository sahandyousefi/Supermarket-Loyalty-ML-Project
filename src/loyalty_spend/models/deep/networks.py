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


class PeriodicEmbedding(nn.Module):
    """PLR embedding of each numeric feature.

    ``x -> ReLU(W [sin(2 pi c x), cos(2 pi c x)] + b)`` with trainable
    frequencies ``c`` initialized from ``N(0, sigma^2)``, one set per feature.
    """

    def __init__(
        self,
        n_features: int,
        n_frequencies: int = 16,
        sigma: float = 0.5,
        d_out: int = 16,
    ) -> None:
        super().__init__()
        self.frequencies = nn.Parameter(torch.randn(n_features, n_frequencies) * sigma)
        self.weight = nn.Parameter(torch.empty(n_features, 2 * n_frequencies, d_out))
        self.bias = nn.Parameter(torch.zeros(n_features, d_out))
        nn.init.xavier_uniform_(self.weight)
        self.d_out = d_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = 2 * torch.pi * self.frequencies * x.unsqueeze(-1)
        periodic = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        out = torch.einsum("bfk,fkd->bfd", periodic, self.weight) + self.bias
        return torch.relu(out).flatten(1)


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
        n_in = n_num + sum(embedding_dim(c) for c in cardinalities)
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


class FeatureTokenizer(nn.Module):
    """One d-dimensional token per feature.

    A numeric feature ``x_j`` becomes ``x_j * w_j + b_j``; a categorical
    feature is looked up in its own embedding table and shifted by a bias.
    """

    def __init__(self, n_numeric: int, cardinalities: list[int], d_token: int) -> None:
        super().__init__()
        self.num_weight = nn.Parameter(torch.empty(n_numeric, d_token))
        self.num_bias = nn.Parameter(torch.zeros(n_numeric, d_token))
        self.cat_embeddings = nn.ModuleList(
            nn.Embedding(c, d_token) for c in cardinalities
        )
        self.cat_bias = nn.Parameter(torch.zeros(len(cardinalities), d_token))
        nn.init.kaiming_uniform_(self.num_weight, a=5**0.5)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        num = x_num.unsqueeze(-1) * self.num_weight + self.num_bias
        cat = torch.stack(
            [e(x_cat[:, i]) for i, e in enumerate(self.cat_embeddings)], dim=1
        )
        return torch.cat([num, cat + self.cat_bias], dim=1)


class FTTransformer(nn.Module):
    """Transformer encoder over feature tokens with a [CLS] readout."""

    def __init__(
        self,
        n_numeric: int,
        cardinalities: list[int],
        d_token: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        n_outputs: int = 1,
    ) -> None:
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_numeric, cardinalities, d_token)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=2 * d_token,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, n_layers, enable_nested_tensor=False
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_token), nn.GELU(), nn.Linear(d_token, n_outputs)
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x_num, x_cat)
        cls = self.cls.expand(tokens.shape[0], -1, -1)
        h = self.encoder(torch.cat([cls, tokens], dim=1))
        return self.head(h[:, 0])


ARCHITECTURES = {"mlp": EmbeddingMLP, "ft_transformer": FTTransformer}
