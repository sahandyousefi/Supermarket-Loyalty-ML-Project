import pytest

from loyalty_spend.ops.tracking import log_run, read_runs


def test_local_fallback_appends_runs(tmp_path):
    path = tmp_path / "runs.jsonl"
    where = log_run(
        "a", {"model": "ols"}, {"rmse": 1.2}, local_path=path, use_mlflow=False
    )
    log_run("b", {"model": "ridge"}, {"rmse": 1.1}, local_path=path, use_mlflow=False)
    assert where == str(path)
    runs = read_runs(path)
    assert [r["run"] for r in runs] == ["a", "b"]
    assert runs[1]["metrics"]["rmse"] == pytest.approx(1.1)
    assert read_runs(tmp_path / "missing.jsonl") == []


def test_mlflow_backend(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())
    assert log_run("c", {"model": "ols"}, {"rmse": 1.0}, use_mlflow=True) == "mlflow"
    runs = mlflow.search_runs(experiment_names=["loyalty-spend"])
    assert len(runs) == 1
