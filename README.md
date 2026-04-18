# Multilingual Sentiment Analysis Service

A learning project working through the core ML engineering stack — one tool at a time, all built on the same service.

Fine-tunes [`cardiffnlp/twitter-xlm-roberta-base-sentiment`](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment) on multilingual tweet data and serves predictions via CLI and REST API.

**Labels:** `negative` · `neutral` · `positive`  
**Languages:** English, German, Arabic, and more (XLM-RoBERTa multilingual backbone)

---

## Roadmap

| Step | Skill | Status |
|---|---|---|
| 1 | Docker — containerized training + CLI inference | ✅ Done |
| 2 | FastAPI — REST API serving predictions | 🔨 In progress |
| 3 | MLflow — experiment tracking + model registry | Planned |
| 4 | GitHub Actions — CI/CD, auto-test + auto-build | Planned |
| 5 | GCP Cloud Run — public URL, auto-deployed | Planned |
| 6 | DVC — fully reproducible ML pipeline | Planned |

Each step builds on the previous one. No rewrites — just additions.

---

## Step 1 — Docker

### What was built

- `src/train.py` — fine-tuning pipeline using HuggingFace `Trainer`
- `src/inference.py` — CLI inference using `transformers.pipeline`
- `Dockerfile` — slim Python image, runs as non-root user
- `docker-compose.yml` — `train` and `predict` services with volume mounts

### Quickstart

**Train:**
```bash
docker compose up train
```
Model is saved to `./models/sentiment/` via volume mount.

**Predict:**
```bash
# Default text
docker compose run predict

# Custom text
PREDICT_TEXT="Das ist fantastisch!" docker compose run predict
```

**Training options:**
```
--model-name        Pre-trained model to fine-tune        [default: cardiffnlp/twitter-xlm-roberta-base-sentiment]
--output-dir        Where to save the trained model       [default: /app/models/sentiment]
--epochs            Training epochs                       [default: 2]
--train-samples     Training subset size (0 = full)       [default: 2000]
--val-samples       Validation subset size (0 = full)     [default: 400]
--max-length        Max tokenizer sequence length         [default: 128]
```

### Project structure

```
sentiment-service/
├── src/
│   ├── train.py        # fine-tuning pipeline
│   └── inference.py    # CLI inference
├── data/               # dataset cache (git-ignored)
├── models/             # saved model output (git-ignored)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Local development

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
python src/train.py
python src/inference.py "Jag älskar det här!"
```
