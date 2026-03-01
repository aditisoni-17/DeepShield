"""
FastAPI backend for DeepShield deepfake detection.

Run from project root:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health            — health check, confirms model is loaded
    POST /predict/image     — analyze uploaded image
    POST /predict/video     — analyze uploaded video (samples N frames)
    WS   /ws/webcam         — real-time frame-by-frame WebSocket inference

Interactive docs: http://localhost:8000/docs
"""

import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MODEL_PATH = os.path.join(ROOT, "saved_models", "best_model.pth")

# Module-level model cache — loaded once at startup
_model = None
_device = None


def get_model():
    """Return the cached (model, device) tuple. Raises if not loaded yet."""
    if _model is None:
        raise RuntimeError("Model not loaded. Server may still be starting up.")
    return _model, _device


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once when the server starts; release on shutdown."""
    global _model, _device
    from inference.predict import load_model
    print(f"[DeepShield] Loading model from {MODEL_PATH} …")
    _model, _device = load_model(MODEL_PATH)
    print(f"[DeepShield] Model ready on device: {_device}")
    yield
    _model = None
    _device = None
    print("[DeepShield] Model unloaded.")


app = FastAPI(
    title="DeepShield API",
    description="Real-time deepfake detection — offline, privacy-preserving.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow Streamlit (localhost:8501) and any local frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from api.routes.predict import router as predict_router
from api.routes.stream import router as stream_router

app.include_router(predict_router, tags=["Prediction"])
app.include_router(stream_router, tags=["Streaming"])


@app.get("/health", tags=["Health"])
def health():
    from api.schemas import HealthResponse
    loaded = _model is not None
    return HealthResponse(
        status="ok" if loaded else "model_not_loaded",
        model_loaded=loaded,
        device=str(_device) if _device else "unknown",
    )