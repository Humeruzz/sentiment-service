import logging
from pathlib import Path

import torch
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from transformers import pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(show_path=False)],
)
log = logging.getLogger(__name__)
console = Console()

MODEL_DIR = "/app/models/sentiment"

_classifier = None


def get_classifier(model_dir: str):
    global _classifier
    if _classifier is None:
        device = 0 if torch.cuda.is_available() else -1
        log.info(f"Loading model from {model_dir} (device={'GPU' if device == 0 else 'CPU'})")
        _classifier = pipeline(
            "text-classification",
            model=model_dir,
            tokenizer=model_dir,
            device=device,
        )
        log.info("Model loaded.")
    return _classifier


def predict(text: str, model_dir: str = MODEL_DIR) -> dict:
    if not text.strip():
        raise ValueError("Input text must not be empty.")

    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}. Run train.py first."
        )

    classifier = get_classifier(model_dir)
    try:
        result = classifier(text)[0]
    except Exception as e:
        raise RuntimeError(f"Inference failed: {e}") from e

    return {
        "text": text,
        "label": result["label"],
        "confidence": round(result["score"], 4),
    }


app = typer.Typer(add_completion=False)


@app.command()
def main(
    text: str = typer.Argument(..., help="Text to classify"),
    model_dir: str = typer.Option(MODEL_DIR, "--model-dir", help="Path to saved model"),
):
    try:
        output = predict(text, model_dir=model_dir)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        log.error(str(e))
        raise typer.Exit(1)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("[bold]Text[/bold]", output["text"])
    table.add_row("[bold]Sentiment[/bold]", output["label"])
    table.add_row("[bold]Confidence[/bold]", str(output["confidence"]))
    console.print(table)


if __name__ == "__main__":
    app()
