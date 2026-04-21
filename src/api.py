import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.inference import get_classifier, predict

MODEL_DIR = "/app/models/sentiment"
MLFLOW_META_PATH = f"{MODEL_DIR}/mlflow_meta.json"


class PredictRequest(BaseModel):
    text: str
    lang: str = "en"


class PredictResponse(BaseModel):
    text: str
    label: str
    confidence: float
    lang: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_classifier(MODEL_DIR)
    yield


app = FastAPI(title="Sentiment Analysis API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    try:
        result = predict(request.text, model_dir=MODEL_DIR)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return PredictResponse(**result, lang=request.lang)


@app.get("/metadata")
def metadata():
    meta_file = Path(MLFLOW_META_PATH)
    if not meta_file.exists():
        return {"mlflow_run_id": None, "model_version": None}
    return json.loads(meta_file.read_text())
