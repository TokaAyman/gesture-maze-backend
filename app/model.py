# File: gesture-maze-backend/app/model.py

import joblib
import os

# If you prefer to call load_model() rather than loading directly in main.py, you can use this.
# But since main.py already does joblib.load(MODEL_PATH) directly, this helper is optional.
MODEL_PATH = os.getenv("MODEL_PATH", "model/best_xgboost.joblib")

def load_model():
    return joblib.load(MODEL_PATH)
