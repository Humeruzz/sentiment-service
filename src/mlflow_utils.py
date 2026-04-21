import json
from datetime import datetime, timezone

import mlflow
from mlflow.tracking import MlflowClient

REGISTERED_MODEL_NAME = "sentiment-classifier"


def log_model_artifacts(output_dir: str) -> None:
    mlflow.log_artifacts(output_dir, artifact_path="model")


def register_model(run_id: str, model_name: str = REGISTERED_MODEL_NAME) -> str:
    model_uri = f"runs:/{run_id}/model"
    version = mlflow.register_model(model_uri, model_name)
    MlflowClient().transition_model_version_stage(
        name=model_name,
        version=version.version,
        stage="Staging",
    )
    return str(version.version)


def write_run_sidecar(run_id: str, model_version: str, output_path: str) -> None:
    data = {
        "run_id": run_id,
        "model_version": model_version,
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
