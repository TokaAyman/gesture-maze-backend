from fastapi import FastAPI
from app.model import predict_gesture
from app.schemas import GestureRequest, GestureResponse

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Gesture Maze API is running"}

@app.post("/predict", response_model=GestureResponse)
def predict(gesture: GestureRequest):
    prediction = predict_gesture(gesture.features)
    return {"direction": prediction}
