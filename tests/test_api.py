# File: gesture-maze-backend/tests/test_api.py

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict_valid():
    # Adjust `vector_length` so it matches your model’s expected input size.
    vector_length = 57
    dummy_features = [0.0] * vector_length
    response = client.post("/predict", json={"features": dummy_features})
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert data["label"] in ["up", "down", "left", "right", "none"]
    assert "confidence" in data
    assert isinstance(data["confidence"], float)

def test_predict_invalid_shape():
    # Send a feature vector of incorrect length to trigger an error
    bad_features = [0.0] * 5
    response = client.post("/predict", json={"features": bad_features})
    # Expect a 422 (validation error) or 500 if model inference fails
    assert response.status_code in (422, 500)
