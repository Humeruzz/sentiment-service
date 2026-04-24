import logging
import os

import mlflow
import numpy as np
import torch
import typer
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


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}


app = typer.Typer(add_completion=False)


@app.command()
def main(
    model_name: str = typer.Option(DEFAULT_MODEL, "--model-name", help="Pre-trained model to fine-tune"),
    output_dir: str = typer.Option(DEFAULT_OUTPUT_DIR, "--output-dir", help="Where to save the trained model"),
    epochs: int = typer.Option(2, "--epochs", help="Number of training epochs"),
    train_batch_size: int = typer.Option(16, "--train-batch-size"),
    eval_batch_size: int = typer.Option(32, "--eval-batch-size"),
    train_samples: int = typer.Option(2000, "--train-samples", help="Training subset size (0 = full dataset)"),
    val_samples: int = typer.Option(400, "--val-samples", help="Validation subset size (0 = full dataset)"),
    max_length: int = typer.Option(128, "--max-length", help="Max tokenizer sequence length"),
):
    use_gpu = torch.cuda.is_available()
    log.info(f"Device: {'GPU (' + torch.cuda.get_device_name(0) + ')' if use_gpu else 'CPU'}")

    log.info("Loading dataset...")
    try:
        dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    except Exception as e:
        log.error(f"Failed to load dataset: {e}")
        raise typer.Exit(1)

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
        raise typer.Exit(1)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)

    train_data = train_data.map(tokenize, batched=True, num_proc=1)
    eval_data = eval_data.map(tokenize, batched=True, num_proc=1)

    log.info("Loading model...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=NUM_LABELS)
    except Exception as e:
        log.error(f"Failed to load model: {e}")
        raise typer.Exit(1)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("sentiment-analysis")

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
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
    with mlflow.start_run() as run:
        mlflow.log_params({
            "model_name": model_name,
            "train_samples": train_samples,
            "val_samples": val_samples,
            "max_length": max_length,
            "epochs": epochs,
        })
        trainer.train()
        eval_results = trainer.evaluate()
        mlflow.log_metrics({k: v for k, v in eval_results.items() if k in ("eval_loss", "eval_accuracy")})
        log.info(f"Saving model to {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        model_uri = log_model_artifacts(trainer.model, tokenizer)
        version = register_model(model_uri)
        write_run_sidecar(run.info.run_id, version, f"{output_dir}/mlflow_meta.json")
        log.info(f"MLflow run: {run.info.run_id} | model version: {version}")

    log.info("Done.")


if __name__ == "__main__":
    app()
