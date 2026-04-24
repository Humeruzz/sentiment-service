import json
from datetime import datetime, timezone

import mlflow
import mlflow.transformers
from mlflow.tracking import MlflowClient

REGISTERED_MODEL_NAME = "sentiment-classifier"


def log_model_artifacts(model, tokenizer) -> str:
    logged = mlflow.transformers.log_model(
        transformers_model={"model": model, "tokenizer": tokenizer},
        name="model",
        task="text-classification",
        pip_requirements=["transformers", "torch", "sentencepiece"],
    )
    return logged.model_uri


def register_model(model_uri: str, model_name: str = REGISTERED_MODEL_NAME) -> str:
    version = mlflow.register_model(model_uri, model_name)
    MlflowClient().set_registered_model_alias(
        name=model_name,
        alias="staging",
        version=version.version,
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
