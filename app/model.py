import joblib
import numpy as np

model = joblib.load("best_xgboost.joblib")

def predict_gesture(features):
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)[0]
    return prediction
