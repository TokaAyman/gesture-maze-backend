# File: call_api_row3.py

import json
import numpy as np
import requests

# 1) Define the same feature vector you tested above (row 3):
row3_features = [
    0.000000,  0.000000,   -0.009741,  -0.006678,  -0.009764,  -0.020566,
    0.007800,  -0.030191,   0.022742,  -0.038564,  -0.012111,  -0.066596,
    -0.021992, -0.095398,  -0.025892, -0.112634,  -0.029628,  -0.127852,
    0.003159,  -0.069367,   0.003156,  -0.102689,   0.005376,  -0.122302,
    0.005758,  -0.138741,   0.016012,  -0.063150,   0.028335,  -0.086269,
    0.038306,  -0.101853,   0.046145,  -0.116191,   0.026321,  -0.049806,
    0.029488,  -0.059667,   0.025885,  -0.049010,   0.022867,  -0.036388,
    0.103513,   0.101607,   0.081078,   0.002180,   0.037023,   0.076665,
    0.105458,   0.046257,   0.103774,   0.083129,   0.045518,   0.063711,
    0.069423,   0.061003,   0.013856
]

# 2) Set up the request payload
payload = {"features": row3_features}

# 3) URL of your running FastAPI server
URL = "http://127.0.0.1:8000/predict"  # adjust if different

# 4) Send the POST request
try:
    response = requests.post(URL, json=payload)
    response.raise_for_status()
except Exception as e:
    print("❌ Failed to call /predict endpoint:", e)
    print("  Is your FastAPI server running on port 8000?")
    exit(1)

# 5) Print out exactly what the API returned
print("API status code:", response.status_code)
print("API response JSON:", json.dumps(response.json(), indent=2))
