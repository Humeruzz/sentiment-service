import json

import mlflow
import pytest


def test_write_run_sidecar(tmp_path):
    from src.mlflow_utils import write_run_sidecar

    out = tmp_path / "mlflow_meta.json"
    write_run_sidecar("abc123", "2", str(out))
    data = json.loads(out.read_text())
    assert data["run_id"] == "abc123"
    assert data["model_version"] == "2"
    assert "registered_at" in data


def test_log_model_artifacts(tmp_path):
    mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
    mlflow.set_experiment("test")

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "roberta"}')

    from src.mlflow_utils import log_model_artifacts

    with mlflow.start_run():
        log_model_artifacts(str(model_dir))
        run_id = mlflow.active_run().info.run_id

    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run_id, "model")
    assert any(a.path == "model/config.json" for a in artifacts)
