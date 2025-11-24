"""Results must not depend on hash seeds, thread counts or the process.

Each run happens in a fresh interpreter with a different hash seed and
number of BLAS and OpenMP threads, and prints the quantities that feed the
report: out-of-fold forecasts of a linear and a boosted model, a conformal
quantile, Monte Carlo reward cost and a synthetic sample.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

SCRIPT = r"""
import json, sys
import numpy as np
from loyalty_spend.business.rewards import RewardSchedule, forecast_liability
from loyalty_spend.data.io import load_clean
from loyalty_spend.evaluation.cv import cross_validate, make_splits
from loyalty_spend.features.preprocess import split_xy
from loyalty_spend.models.registry import build_model
from loyalty_spend.synthetic.conditional import ConditionalCopulaSynthesizer
from loyalty_spend.uncertainty.conformal import conformal_quantile

df, _ = load_clean(sys.argv[1], impute=False)
X, y = split_xy(df)
splits = make_splits(len(y), 3, 1, 0)
ols = cross_validate("ols", build_model("ols"), X, y, splits).oof
gbm = cross_validate("hist_gbm", build_model("hist_gbm"), X, y, splits).oof
q = conformal_quantile(np.abs(y.to_numpy() - ols), 0.1)
cost = forecast_liability(
    np.arange(50), ols[:50], y.to_numpy() - ols,
    RewardSchedule(((0.0, 0.0), (120.0, 0.01), (135.0, 0.02))), 400, 0,
)
synth = ConditionalCopulaSynthesizer("loyalty_years", seed=0).fit(
    df.drop(columns="customer_id")
).sample(50)
print(json.dumps({
    "ols": ols.tolist(), "gbm": gbm.tolist(), "q": q,
    "cost": cost.total_draws.tolist(), "synth": synth["spend"].tolist(),
}))
"""


def _run(raw, hash_seed, threads):
    env = {
        **os.environ,
        "PYTHONHASHSEED": str(hash_seed),
        "OMP_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
    }
    out = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(raw / "train.csv")],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.slow
def test_results_do_not_depend_on_process_settings(raw_dir):
    a = _run(raw_dir, 0, 1)
    b = _run(raw_dir, 12345, 2)
    for key in ("gbm", "q", "cost", "synth"):
        assert np.array_equal(np.asarray(a[key]), np.asarray(b[key])), key
    # Least squares may sum in a different order with more BLAS threads.
    assert np.allclose(a["ols"], b["ols"], rtol=0, atol=1e-9)
