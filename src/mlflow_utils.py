import json
import os
from datetime import datetime, timezone

import mlflow
import mlflow.transformers
import requests
from mlflow.tracking import MlflowClient
from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn, TransferSpeedColumn

REGISTERED_MODEL_NAME = "sentiment-classifier"


def log_model_artifacts(model, tokenizer) -> str:
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=Console(force_terminal=True),
        transient=False,
    )
    original_send = requests.Session.send
    upload_started = False

    with progress:
        overall = progress.add_task("[yellow]Serializing model...", total=None)

        def send_with_progress(self, request, **kwargs):
            nonlocal upload_started
            if request.method == "PUT" and request.body is not None and hasattr(request.body, "read"):
                if not upload_started:
                    upload_started = True
                    progress.update(overall, description="[yellow]Uploading artifacts...")
                try:
                    size = os.fstat(request.body.fileno()).st_size
                    name = os.path.basename(request.body.name)
                    task = progress.add_task(f"  [cyan]{name}", total=size)
                    original_read = request.body.read

                    def read_with_progress(n=-1):
                        chunk = original_read(n)
                        if chunk:
                            progress.advance(task, len(chunk))
                        return chunk

                    request.body.read = read_with_progress
                except (AttributeError, OSError):
                    pass
            return original_send(self, request, **kwargs)

        requests.Session.send = send_with_progress
        try:
            logged = mlflow.transformers.log_model(
                transformers_model={"model": model, "tokenizer": tokenizer},
                name="model",
                task="text-classification",
                pip_requirements=["transformers", "torch", "sentencepiece"],
            )
        finally:
            requests.Session.send = original_send

        progress.update(overall, description="[green]Artifacts uploaded.", total=1, completed=1)

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
