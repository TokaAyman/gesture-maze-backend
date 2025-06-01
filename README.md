# Gesture‐Maze Backend

This repository contains the FastAPI backend for serving a pre‐trained hand‐gesture model and exposing metrics for Prometheus/Grafana.

## Repo Structure

gesture-maze-backend/
├── app/
│ ├── init.py
│ ├── main.py
│ ├── schemas.py
│ ├── model.py (optional)
│ └── utils.py (optional)
├── model/
│ └── gesture_model.joblib
├── tests/
│ └── test_api.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
├── prometheus/
│ └── prometheus.yml
└── README.md

