import numpy as np
import pytest

torch = pytest.importorskip("torch")

from loyalty_spend.models.deep.networks import (  # noqa: E402
    EmbeddingMLP,
    FTTransformer,
)
from loyalty_spend.models.deep.tabular import (  # noqa: E402
    TabularEncoder,
    embedding_dim,
)

FAST = dict(epochs=15, patience=5)


def test_encoder_shapes_and_unknown_levels(xy):
    X, _ = xy
    enc = TabularEncoder().fit(X)
    x_num, x_cat = enc.transform(X)
    assert x_num.shape == (len(X), enc.n_numeric)
    assert x_cat.shape == (len(X), 4)
    assert int(x_cat.min()) >= 1
    unseen = X.head(1).copy()
    unseen["region"] = np.nan
    assert int(enc.transform(unseen)[1][0, 0]) == 0


def test_embedding_dim_grows_slowly():
    assert embedding_dim(3) < embedding_dim(13) <= 16


@pytest.mark.parametrize("cls", [EmbeddingMLP, FTTransformer])
def test_network_output_shape(cls):
    net = cls(n_numeric=4, cardinalities=[5, 6, 14, 3], n_outputs=2)
    out = net(torch.randn(8, 4), torch.randint(0, 3, (8, 4)))
    assert out.shape == (8, 2)
