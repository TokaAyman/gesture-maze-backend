from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Gesture Maze API is running"}

def test_predict():
    payload = {"features": [0.5, -1.2, 1.3, 0.7]}  # example input
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "direction" in response.json()
