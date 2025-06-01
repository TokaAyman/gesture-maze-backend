# File: gesture-maze-backend/app/metrics.py

import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Request, Response, APIRouter

# --------------------------------------------------------------------------------
# 1. MODEL-RELATED METRICS
# --------------------------------------------------------------------------------

# Count how many times /predict (and other endpoints) are called:
REQUEST_COUNT = Counter(
    'api_request_count', 
    'Total number of API requests received',
    ['method', 'endpoint']
)

# Measure inference time (in seconds). We will `.observe()` for only /predict calls.
INFERENCE_LATENCY = Histogram(
    'inference_latency_seconds', 
    'Time spent in seconds to process each /predict inference'
)

# --------------------------------------------------------------------------------
# 2. DATA-RELATED METRIC (example: payload size in bytes)
# --------------------------------------------------------------------------------

# Track size (in bytes) of request bodies to /predict. Uses a histogram
REQUEST_PAYLOAD_SIZE = Histogram(
    'request_payload_size_bytes',
    'Size of request payloads in bytes (for /predict)',
    buckets=[100, 500, 1000, 5000, 10000, 50000, 100000]
)

# --------------------------------------------------------------------------------
# 3. Expose router for metrics endpoints
# --------------------------------------------------------------------------------
metrics_router = APIRouter()

@metrics_router.get("/metrics")
def metrics():
    """
    Expose the Prometheus metrics. 
    Prometheus will scrape this endpoint.
    """
    data = generate_latest()  # Collects all registered metrics
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

@metrics_router.get("/health")
def health():
    """
    A simple health check for readiness.
    """
    return {"status": "ok"}
