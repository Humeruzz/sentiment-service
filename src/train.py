import json
import logging
import os
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import torch
import typer
import yaml
from datasets import load_dataset
from rich.logging import RichHandler
from sklearn.metrics import accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.mlflow_utils import log_model_artifacts, register_model, write_run_sidecar

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(show_path=False)],
)
log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

DEFAULT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
DEFAULT_OUTPUT_DIR = "/app/models/sentiment"
NUM_LABELS = 3  # negative, neutral, positive

_PARAMS_PATH = Path(__file__).resolve().parent.parent / "params.yaml"


def _load_params() -> dict:
    if not _PARAMS_PATH.exists():
        return {}
    with open(_PARAMS_PATH) as f:
        data = yaml.safe_load(f) or {}
    return data.get("train", {})


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}


app = typer.Typer(add_completion=False)


def train(
    model_name: str = DEFAULT_MODEL,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    epochs: int = 2,
    train_batch_size: int = 16,
    eval_batch_size: int = 32,
    train_samples: int = 2000,
    val_samples: int = 400,
    max_length: int = 128,
    learning_rate: float = 5e-5,
    warmup_ratio: float = 0.0,
    weight_decay: float = 0.0,
    run_name: str = None,
):
    use_gpu = torch.cuda.is_available()
    log.info(f"Device: {'GPU (' + torch.cuda.get_device_name(0) + ')' if use_gpu else 'CPU'}")

    log.info("Loading dataset...")
    try:
        dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    except Exception as e:
        log.error(f"Failed to load dataset: {e}")
        raise SystemExit(1)

    train_data = dataset["train"].shuffle(seed=42)
    eval_data = dataset["validation"].shuffle(seed=42)
    if train_samples > 0:
        train_data = train_data.select(range(min(train_samples, len(train_data))))
    if val_samples > 0:
        eval_data = eval_data.select(range(min(val_samples, len(eval_data))))
    log.info(f"Train: {len(train_data)} samples | Val: {len(eval_data)} samples")

    log.info(f"Loading tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        log.error(f"Failed to load tokenizer: {e}")
        raise SystemExit(1)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)

    train_data = train_data.map(tokenize, batched=True, num_proc=1)
    eval_data = eval_data.map(tokenize, batched=True, num_proc=1)

    log.info("Loading model...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=NUM_LABELS)
    except Exception as e:
        log.error(f"Failed to load model: {e}")
        raise SystemExit(1)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("sentiment-analysis")

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=50,
        fp16=use_gpu,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        compute_metrics=compute_metrics,
    )

    log.info("Training...")
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "model_name": model_name,
            "train_samples": train_samples,
            "val_samples": val_samples,
            "max_length": max_length,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "warmup_ratio": warmup_ratio,
            "weight_decay": weight_decay,
        })
        trainer.train()
        eval_results = trainer.evaluate()
        mlflow.log_metrics({k: v for k, v in eval_results.items() if k in ("eval_loss", "eval_accuracy")})
        repo_root = Path(__file__).resolve().parent.parent
        with open(repo_root / "metrics.json", "w") as f:
            json.dump(
                {
                    "eval_loss": eval_results.get("eval_loss"),
                    "eval_accuracy": eval_results.get("eval_accuracy"),
                },
                f,
                indent=2,
            )
        log.info(f"Saving model to {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        model_uri = log_model_artifacts(trainer.model, tokenizer)
        version = register_model(model_uri)
        write_run_sidecar(run.info.run_id, version, f"{output_dir}/mlflow_meta.json")
        log.info(f"MLflow run: {run.info.run_id} | model version: {version}")

    log.info("Done.")


@app.command()
def main(
    model_name: Optional[str] = typer.Option(None, "--model-name", help="Pre-trained model to fine-tune"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", help="Where to save the trained model"),
    epochs: Optional[int] = typer.Option(None, "--epochs", help="Number of training epochs"),
    train_batch_size: Optional[int] = typer.Option(None, "--train-batch-size"),
    eval_batch_size: Optional[int] = typer.Option(None, "--eval-batch-size"),
    train_samples: Optional[int] = typer.Option(None, "--train-samples", help="Training subset size (0 = full dataset)"),
    val_samples: Optional[int] = typer.Option(None, "--val-samples", help="Validation subset size (0 = full dataset)"),
    max_length: Optional[int] = typer.Option(None, "--max-length", help="Max tokenizer sequence length"),
    learning_rate: Optional[float] = typer.Option(None, "--learning-rate", help="AdamW learning rate"),
    warmup_ratio: Optional[float] = typer.Option(None, "--warmup-ratio", help="Fraction of steps used for LR warmup"),
    weight_decay: Optional[float] = typer.Option(None, "--weight-decay", help="L2 regularization coefficient"),
    run_name: Optional[str] = typer.Option(None, "--run-name", help="MLflow run name"),
):
    p = _load_params()
    train(
        model_name=model_name if model_name is not None else p.get("model_name", DEFAULT_MODEL),
        output_dir=output_dir if output_dir is not None else p.get("output_dir", DEFAULT_OUTPUT_DIR),
        epochs=epochs if epochs is not None else p.get("epochs", 2),
        train_batch_size=train_batch_size if train_batch_size is not None else p.get("train_batch_size", 16),
        eval_batch_size=eval_batch_size if eval_batch_size is not None else p.get("eval_batch_size", 32),
        train_samples=train_samples if train_samples is not None else p.get("train_samples", 2000),
        val_samples=val_samples if val_samples is not None else p.get("val_samples", 400),
        max_length=max_length if max_length is not None else p.get("max_length", 128),
        learning_rate=learning_rate if learning_rate is not None else p.get("learning_rate", 5e-5),
        warmup_ratio=warmup_ratio if warmup_ratio is not None else p.get("warmup_ratio", 0.0),
        weight_decay=weight_decay if weight_decay is not None else p.get("weight_decay", 0.0),
        run_name=run_name,
    )


if __name__ == "__main__":
    app()
