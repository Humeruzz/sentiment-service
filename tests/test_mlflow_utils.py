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
    from unittest.mock import MagicMock, patch

    from src.mlflow_utils import log_model_artifacts

    mock_logged = MagicMock()
    mock_logged.model_uri = "models:/m-test123"
    with patch("mlflow.transformers.log_model", return_value=mock_logged) as mock_log:
        uri = log_model_artifacts(MagicMock(), MagicMock())

    mock_log.assert_called_once()
    _, kwargs = mock_log.call_args
    assert kwargs["name"] == "model"
    assert kwargs["task"] == "text-classification"
    assert uri == "models:/m-test123"
