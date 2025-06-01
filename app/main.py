# File: gesture-maze-backend/app/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
import time
from datetime import datetime
import logging
from contextlib import asynccontextmanager
import psutil
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

from app.utils import preprocess_landmarks

# -------------- Logging Setup --------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------- PROMETHEUS METRICS --------------
# Model-related metrics
prediction_counter = Counter('gesture_predictions_total', 'Total number of predictions made', ['gesture'])
prediction_confidence = Histogram('gesture_prediction_confidence', 'Confidence scores of predictions')
model_loading_time = Gauge('model_loading_time_seconds', 'Time taken to load the model')

# Data-related metrics
invalid_requests_counter = Counter('invalid_requests_total', 'Total number of invalid requests', ['error_type'])
processing_time = Histogram('request_processing_seconds', 'Time spent processing requests')

# Server-related metrics
active_connections = Gauge('active_connections', 'Number of active connections')
memory_usage = Gauge('memory_usage_bytes', 'Memory usage in bytes')
cpu_usage = Gauge('cpu_usage_percent', 'CPU usage percentage')

# -------------- SCHEMAS --------------

class Landmark(BaseModel):
    x: float
    y: float
    z: float

class HandLandmarksRequest(BaseModel):
    landmarks: List[Landmark]  # Exactly 21 landmarks expected

class PredictionResponse(BaseModel):
    label: str       # "up" | "down" | "left" | "right"
    confidence: float
    probabilities: dict
    processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str
    memory_usage_mb: float
    cpu_usage_percent: float

class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None

# -------------- LABEL MAP --------------
LABEL_MAP = {
    0: "down",
    1: "left", 
    2: "right",
    3: "up",
}

# -------------- PREPROCESSING HELPERS --------------

def validate_landmarks(landmarks: List[Landmark]) -> bool:
    """
    Only checks that exactly 21 landmarks are provided.
    """
    if len(landmarks) != 21:
        logger.warning(f"Validation failed: Expected 21 landmarks, got {len(landmarks)}")
        invalid_requests_counter.labels(error_type="invalid_landmark_count").inc()
        return False
    return True

def update_system_metrics():
    """Update system-related metrics"""
    memory_usage.set(psutil.virtual_memory().used)
    cpu_usage.set(psutil.cpu_percent())

# -------------- MODEL PREDICTOR --------------

class GesturePredictor:
    def __init__(self):
        self.model = None
        self.is_loaded = False

    def load_model(self):
        try:
            start_time = time.time()
            self.model = joblib.load("Model/best_xgboost.joblib")
            loading_time = time.time() - start_time
            
            model_loading_time.set(loading_time)
            self.is_loaded = True
            logger.info(f"✅ Model loaded successfully in {loading_time:.2f} seconds")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            self.is_loaded = False

    def predict(self, raw_landmarks: List[Landmark]):
        """
        Predict gesture from landmarks with metrics tracking
        """
        # Convert landmarks and preprocess
        lm_array = np.array([[lm.x, lm.y, lm.z] for lm in raw_landmarks], dtype=np.float32)
        features = preprocess_landmarks(lm_array)

        # Get predictions
        proba = self.model.predict_proba(features)[0]
        idx = int(np.argmax(proba))
        
        label = LABEL_MAP[idx]
        confidence = float(proba[idx])
        prob_dict = {LABEL_MAP[i]: float(p) for i, p in enumerate(proba.tolist())}

        # Update metrics
        prediction_counter.labels(gesture=label).inc()
        prediction_confidence.observe(confidence)

        return label, confidence, prob_dict

# -------------- FASTAPI SETUP --------------

predictor: GesturePredictor | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    predictor = GesturePredictor()
    predictor.load_model()
    yield
    logger.info("🛑 Shutting down...")

app = FastAPI(
    title="Hand Gesture Recognition API",
    description="Recognize hand gestures from 3D landmarks with monitoring",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------- ROUTES --------------

@app.get("/")
async def root():
    return {
        "message": "Hand Gesture Recognition API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    update_system_metrics()
    loaded = predictor.is_loaded if predictor else False
    
    return HealthResponse(
        status="healthy" if loaded else "unhealthy",
        model_loaded=loaded,
        timestamp=datetime.utcnow().isoformat(),
        memory_usage_mb=psutil.virtual_memory().used / 1024 / 1024,
        cpu_usage_percent=psutil.cpu_percent()
    )

@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics"""
    update_system_metrics()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: HandLandmarksRequest):
    with processing_time.time():
        # Update active connections
        active_connections.inc()
        
        try:
            # Validate model is loaded
            if not predictor or not predictor.is_loaded:
                invalid_requests_counter.labels(error_type="model_not_loaded").inc()
                raise HTTPException(status_code=503, detail="Model not loaded")

            # Validate landmarks
            if not validate_landmarks(request.landmarks):
                raise HTTPException(status_code=400, detail="Invalid number of landmarks (must be 21)")

            # Make prediction
            start_time = time.time()
            label, confidence, probs = predictor.predict(request.landmarks)
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            
            return PredictionResponse(
                label=label,
                confidence=confidence,
                probabilities=probs,
                processing_time_ms=elapsed_ms
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            invalid_requests_counter.labels(error_type="prediction_error").inc()
            raise HTTPException(status_code=500, detail="Prediction failed")
        finally:
            active_connections.dec()

@app.get("/gestures")
async def gestures():
    if not predictor or not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "gestures": list(LABEL_MAP.values()),
        "count": len(LABEL_MAP)
    }

# -------------- EXCEPTION HANDLERS --------------

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)