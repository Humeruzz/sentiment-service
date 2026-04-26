# Multilingual Sentiment Analysis Service

![CI](https://github.com/Humeruzz/sentiment-service/actions/workflows/ci.yml/badge.svg)

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
| 4 | GitHub Actions — CI/CD, auto-test + auto-build | ✅ Done |
| 5 | DVC — fully reproducible ML pipeline | ✅ Done |

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
├── .github/
│   └── workflows/
│       ├── ci.yml            # run tests on every push/PR
│       └── docker.yml        # build + push image on merge to main
├── src/
│   ├── train.py              # fine-tuning pipeline
│   ├── inference.py          # CLI + library inference
│   ├── api.py                # FastAPI REST service
│   ├── mlflow_utils.py       # MLflow logging helpers
│   └── sweep.py              # hyperparameter sweep runner
├── tests/
├── data/                     # dataset cache (git-ignored)
├── models/                   # saved model output (git-ignored)
├── mlruns/                   # MLflow runs + artifacts (git-ignored)
├── interactive.ipynb
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── requirements-dev.txt      # test dependencies (pytest, httpx)
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

## Step 4 — GitHub Actions

### What was built

- `.github/workflows/ci.yml` — runs `pytest` on every push and every PR to `main`; blocks merges on failure
- `.github/workflows/docker.yml` — builds and pushes Docker image to `ghcr.io` on every merge to `main`
- `requirements-dev.txt` — separates test dependencies (`pytest`, `httpx`) from production requirements
- `Dockerfile` gains `ARG TORCH_INDEX_URL` so CI builds a fast CPU image while local/production keeps ROCm
- Docker image tagged with both `latest` and the exact git SHA for full traceability

### Published image

```bash
docker pull ghcr.io/humeruzz/sentiment-service:latest
```

### CI badge

The badge at the top of this README reflects the current test status of `main`.

---

## Step 5 — DVC

### What was built

- `params.yaml` — single source of truth for all training hyperparameters
- `dvc.yaml` — pipeline definition: deps, params, outs, and metrics for the `train` stage
- `dvc.lock` — committed pipeline lock; proves exact input/output hashes after each run
- `metrics.json` — eval loss + accuracy written after training; committed to git for history
- `.dvcignore` — excludes HuggingFace caches from DVC tracking
- `src/train.py` updated to read defaults from `params.yaml`; CLI overrides still work

**DVC vs MLflow — complementary roles:**

| Concern | Tool |
|---|---|
| Per-run experiment tracking | MLflow |
| Comparing hyperparameter sweeps | MLflow UI |
| Pipeline reproducibility across commits | DVC |
| Pulling a model artifact without re-training | DVC remote |
| What changed between two commits? | `dvc params diff` / `dvc metrics diff` |

### Quickstart

```bash
# Initialize DVC (first time only)
dvc init
dvc remote add -d local ./dvc-remote

# Reproduce the pipeline (skips if nothing changed)
dvc repro

# Check if pipeline is stale
dvc status

# Visualize the pipeline DAG
dvc dag

# Show current metrics
dvc metrics show

# Compare metrics vs previous commit
dvc metrics diff HEAD~1

# Show all current params (vs committed state)
dvc params diff

# Show what params changed vs previous commit
dvc params diff HEAD~1

# Push model artifacts to remote (so others can pull without re-training)
dvc push

# Restore model artifacts without re-training
rm -rf models/sentiment
dvc pull
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
