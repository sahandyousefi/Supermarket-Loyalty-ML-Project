import numpy as np

from loyalty_spend.monitoring.residual_stream import band_stream, chart_stream


def _residuals(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    bands = rng.choice(["a", "b", "c", "d", "e"], n)
    return rng.normal(0, np.where(bands == "b", 0.6, 0.3)), bands


def test_stream_is_standard_normal_in_control():
    e, b = _residuals()
    stream = band_stream(e, b, batch_size=200, n_batches=400)
    assert list(stream.columns) == ["a", "b", "c", "d", "e"]
    assert abs(stream.to_numpy().mean()) < 0.1
    assert 0.85 < stream.to_numpy().std() < 1.15


def test_injected_drop_in_one_band_is_detected_after_it_starts():
    e, b = _residuals()
    stream = band_stream(e, b, 100, 200, shift=0.15, band="c", start=100, seed=1)
    result = chart_stream(stream).set_index("chart")
    first = result["first_signal"]
    assert first.notna().all() and (first > 100).all()
    assert first["MHWMA"] <= first["EWMA"]
