import pandas as pd
import pytest

from loyalty_spend.cli import main
from loyalty_spend.data.io import as_text, load_clean, read_raw

pytest.importorskip("pyarrow")


def test_parquet_and_csv_give_the_same_clean_frame(raw_dir, tmp_path):
    csv = raw_dir / "loyalty.csv"
    typed = pd.read_csv(csv)
    path = tmp_path / "loyalty.parquet"
    typed.to_parquet(path, index=False)
    from_csv, report_csv = load_clean(csv, impute=False)
    from_parquet, report_pq = load_clean(path, impute=False)
    pd.testing.assert_frame_equal(from_csv, from_parquet)
    assert report_csv.to_frame().equals(report_pq.to_frame())


def test_as_text_keeps_missing_values():
    frame = pd.DataFrame({"a": [1.5, None], "b": ["x", None]})
    text = as_text(frame)
    assert text.loc[0, "a"] == "1.5" and pd.isna(text.loc[1, "a"])
    assert text.loc[0, "b"] == "x" and pd.isna(text.loc[1, "b"])


@pytest.fixture
def cfg(raw_dir, tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        f"paths:\n  raw_dir: {raw_dir}\n  output_dir: {tmp_path / 'art'}\n"
        "cv:\n  n_splits: 3\n  n_repeats: 1\n"
    )
    return path


def test_predict_reads_and_writes_parquet(cfg, raw_dir, tmp_path):
    model = tmp_path / "model.joblib"
    assert (
        main(["train", "--config", str(cfg), "--model", "ols", "--output", str(model)])
        == 0
    )
    source = tmp_path / "test.parquet"
    pd.read_csv(raw_dir / "test.csv").to_parquet(source, index=False)
    out = tmp_path / "scores.parquet"
    assert (
        main(
            ["predict", str(source), str(out), "--model", str(model), "--alpha", "0.1"]
        )
        == 0
    )
    scores = pd.read_parquet(out)
    assert {"customer_id", "spend", "lower", "upper"} <= set(scores.columns)
    assert len(scores) == len(read_raw(raw_dir / "test.csv"))
