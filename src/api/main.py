import io
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.api.schemas import PredictionResponse
from src.data.preprocess import EVAL_TRANSFORMS
from src.model.architecture import SimpleCNN
from src.report_writer import write_report

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(REPO_ROOT, "models", "cats_dogs_cnn.pt")

PREDICTION_COUNTER = Counter(
    "cats_dogs_predictions_total",
    "Total predictions made by the inference service",
    ["prediction_label"],
)

_model: SimpleCNN | None = None


def _get_model() -> SimpleCNN:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _model


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    device = torch.device("cpu")
    model = SimpleCNN().to(device)
    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    _model = model
    logger.info(f"[api] Model loaded from {MODEL_PATH}")
    yield
    _model = None


app = FastAPI(
    title="Cats vs Dogs Classifier",
    description="Binary image classification API (SimpleCNN, PyTorch)",
    version="1.0",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app)


@app.get("/health")
def health():
    return {"status": "ok", "model": "SimpleCNN", "version": "1.0"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    t0 = time.perf_counter()

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot decode image")

    tensor = EVAL_TRANSFORMS(img).unsqueeze(0)

    model = _get_model()
    with torch.no_grad():
        prob = model(tensor).item()

    label = "dog" if prob >= 0.5 else "cat"
    latency_ms = (time.perf_counter() - t0) * 1000

    PREDICTION_COUNTER.labels(prediction_label=label).inc()

    log_entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "path": "/predict",
        "label": label,
        "probability": round(prob, 4),
        "latency_ms": round(latency_ms, 2),
    }
    logger.info(json.dumps(log_entry))

    return PredictionResponse(
        label=label,
        probability=round(prob, 4),
        inference_time_ms=round(latency_ms, 2),
    )


def _write_api_report():
    lines = [
        "FastAPI Inference Service",
        f"Model path : {MODEL_PATH}",
        "Endpoints  :",
        "  GET  /health  → {\"status\": \"ok\", \"model\": \"SimpleCNN\", \"version\": \"1.0\"}",
        "  POST /predict → PredictionResponse(label, probability, inference_time_ms)",
        "  GET  /metrics → Prometheus text format (prometheus-fastapi-instrumentator)",
        "",
        "Custom metric: cats_dogs_predictions_total{prediction_label=\"cat|dog\"}",
        "Structured JSON log per /predict call: {ts, path, label, probability, latency_ms}",
        "Model loaded once at startup via lifespan context manager (no reload on request)",
    ]
    write_report("step_7_api", lines)


if __name__ == "__main__":
    _write_api_report()
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False)
