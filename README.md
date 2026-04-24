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
| 2 | FastAPI — REST API serving predictions | ✅ Done |
| 3 | MLflow — experiment tracking + model registry | ✅ Done |
| 4 | GitHub Actions — CI/CD, auto-test + auto-build | Planned |
| 5 | GCP Cloud Run — public URL, auto-deployed | Planned |
| 6 | DVC — fully reproducible ML pipeline | Planned |

Each step builds on the previous one. No rewrites — just additions.

---

## Step 3 — MLflow

### What was built

- `src/mlflow_utils.py` — `log_model_artifacts`, `register_model`, `write_run_sidecar`
- Experiment tracking: params, `eval_loss`, `eval_accuracy` logged per run
- Model registry: trained model registered as `sentiment-classifier` with `staging` alias
- `GET /metadata` endpoint — returns `run_id`, `model_version`, `registered_at` from sidecar file
- MLflow server runs as a separate Docker service, artifacts written directly to `./mlruns/`

### Quickstart

```bash
docker compose up mlflow   # start MLflow server (http://localhost:5001)
docker compose up train    # train + track experiment + register model
docker compose up sweep    # run 3 configs (baseline / conservative / aggressive), compare in MLflow UI
docker compose up api      # serve API (model must exist)
```

**MLflow UI:** http://localhost:5001

**Metadata endpoint:**
```bash
curl http://localhost:8000/metadata
```

---

## Step 2 — FastAPI

### What was built

- `src/api.py` — FastAPI app, reuses `predict()` from `inference.py`
- `GET /health` — returns `{"status": "ok", "model_loaded": true}`
- `POST /predict` — accepts JSON, returns sentiment label + confidence
- Model loads once at startup via lifespan (not per request)
- Swagger UI auto-generated at `/docs`

### Quickstart

```bash
docker compose up train   # train first (model must exist)
docker compose up api     # start API on port 8000
```

**Predict:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Ich liebe das!", "lang": "de"}'
```

**Health check:**
```bash
curl http://localhost:8000/health
```

**Interactive docs:** http://localhost:8000/docs

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
│   ├── train.py          # fine-tuning pipeline
│   ├── inference.py      # CLI + library inference
│   ├── api.py            # FastAPI REST service
│   └── mlflow_utils.py   # MLflow logging helpers
├── tests/
├── data/                 # dataset cache (git-ignored)
├── models/               # saved model output (git-ignored)
├── mlruns/               # MLflow runs + artifacts (git-ignored)
├── interactive.ipynb
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Local development

Requires Python 3.11+. Installs CPU-only PyTorch — fine for running the API and tests locally.

```bash
pip install -r requirements.txt
python src/train.py
python src/inference.py "Jag älskar det här!"
```

> **GPU note:** The Dockerfile is hardcoded for AMD GPU training via ROCm 7.2 (`https://download.pytorch.org/whl/rocm7.2`). If you have an NVIDIA GPU or want CPU-only Docker training, change the `pip install torch torchvision` line in the Dockerfile to use the appropriate wheel index.
