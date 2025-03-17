import numpy as np
import pytest

torch = pytest.importorskip("torch")

from loyalty_spend.models.deep.estimator import (  # noqa: E402
    DeepEnsembleRegressor,
    TorchTabularRegressor,
)
from loyalty_spend.models.deep.networks import (  # noqa: E402
    EmbeddingMLP,
    FTTransformer,
    PeriodicEmbedding,
)
from loyalty_spend.models.deep.tabular import (  # noqa: E402
    TabularEncoder,
    embedding_dim,
)
from loyalty_spend.models.deep.training import gaussian_nll  # noqa: E402

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


def test_periodic_embedding_shape():
    emb = PeriodicEmbedding(n_features=3, n_frequencies=4, d_out=5)
    assert emb(torch.randn(7, 3)).shape == (7, 15)


def test_gaussian_nll_prefers_correct_variance():
    target = torch.zeros(1000)
    noise = torch.randn(1000)
    good = torch.stack([noise, torch.zeros(1000)], dim=1)
    bad = torch.stack([noise, torch.full((1000,), 3.0)], dim=1)
    assert gaussian_nll(good, target) < gaussian_nll(bad, target)


@pytest.mark.slow
@pytest.mark.parametrize("arch", ["mlp", "ft_transformer"])
def test_regressor_learns_signal(xy, arch):
    X, y = xy
    model = TorchTabularRegressor(architecture=arch, epochs=60, patience=15).fit(X, y)
    pred = model.predict(X)
    assert np.sqrt(np.mean((pred - y) ** 2)) < 0.6 * y.std()
    assert model.history_.best_epoch < len(model.history_.val_loss)


def test_regressor_is_reproducible(xy):
    X, y = xy
    a = TorchTabularRegressor(**FAST, seed=3).fit(X, y).predict(X.head(5))
    b = TorchTabularRegressor(**FAST, seed=3).fit(X, y).predict(X.head(5))
    assert np.allclose(a, b)


def test_probabilistic_head_and_ensemble(xy):
    X, y = xy
    model = TorchTabularRegressor(**FAST, probabilistic=True).fit(X, y)
    mean, var = model.predict_dist(X.head(4))
    assert mean.shape == var.shape == (4,) and (var > 0).all()
    with pytest.raises(AttributeError):
        TorchTabularRegressor(**FAST).fit(X, y).predict_dist(X)
    ens = DeepEnsembleRegressor(n_members=2, epochs=10, patience=5).fit(X, y)
    aleatoric, epistemic = ens.predict_components(X.head(4))
    _, total = ens.predict_dist(X.head(4))
    assert np.allclose(total, aleatoric + epistemic)


def test_unknown_architecture(xy):
    X, y = xy
    with pytest.raises(ValueError):
        TorchTabularRegressor(architecture="cnn").fit(X, y)
