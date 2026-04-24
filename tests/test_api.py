import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    with patch("src.inference.get_classifier"):
        from src.api import app
        yield TestClient(app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_metadata_no_file(client, tmp_path, monkeypatch):
    monkeypatch.setattr("src.api.MLFLOW_META_PATH", str(tmp_path / "mlflow_meta.json"))
    response = client.get("/metadata")
    assert response.status_code == 200
    assert response.json()["run_id"] is None


def test_metadata_with_file(client, tmp_path, monkeypatch):
    meta = {"run_id": "abc123", "model_version": "1", "registered_at": "2026-01-01T00:00:00+00:00"}
    meta_file = tmp_path / "mlflow_meta.json"
    meta_file.write_text(json.dumps(meta))
    monkeypatch.setattr("src.api.MLFLOW_META_PATH", str(meta_file))
    response = client.get("/metadata")
    assert response.status_code == 200
    assert response.json()["run_id"] == "abc123"


def test_predict_success(client):
    with patch("src.api.predict") as mock_predict:
        mock_predict.return_value = {"text": "I love this", "label": "positive", "confidence": 0.95}
        response = client.post("/predict", json={"text": "I love this", "lang": "en"})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "positive"
    assert data["lang"] == "en"


def test_predict_invalid_input(client):
    with patch("src.api.predict", side_effect=ValueError("Input text must not be empty.")):
        response = client.post("/predict", json={"text": "", "lang": "en"})
    assert response.status_code == 422
