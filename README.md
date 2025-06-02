# Hand Gesture Maze Backend 🎮✋

A machine learning-powered backend service that enables gesture-controlled maze navigation. This FastAPI-based application uses computer vision and ML models to interpret hand gestures and provide real-time game control.

## 🚀 Features

- **Real-time Hand Gesture Recognition**: ML model processes hand landmarks to predict movement directions
- **RESTful API**: Fast and efficient endpoints for gesture prediction
- **Monitoring & Observability**: Comprehensive metrics using Prometheus and Grafana dashboards
- **CI/CD Pipeline**: Automated deployment using GitHub Actions
- **Containerized Deployment**: Docker-based deployment on Railway platform
- **Model Performance Tracking**: Built-in metrics for prediction accuracy and API usage

## 🏗️ Architecture

```
gesture-maze-backend/
├── app/                          # FastAPI application
│   ├── __pycache__/
│   ├── __init__.py
│   ├── main.py                   # FastAPI app entry point
│   ├── metrics.py                # Prometheus metrics configuration
│   ├── model.py                  # ML model loading and prediction logic
│   ├── schemas.py                # Pydantic schemas for API validation
│   └── utils.py                  # Utility functions
├── monitoring/                   # Monitoring stack
│   └── grafana/
│       ├── dashboards/           # Grafana dashboard configurations
│       └── provisioning/         # Grafana provisioning configs
│       └── prometheus.yml        # Prometheus configuration
├── model/                        # ML model artifacts
├── tests/                        # Test suite
│   ├── __pycache__/
│   ├── test_api.py              # API endpoint tests
│   └── test_main.py             # Main application tests
├── .github/workflows/           # CI/CD pipeline
│   └── deploy.yml               # GitHub Actions workflow
├── docker-compose.yml           # Multi-container orchestration
├── Dockerfile                   # Container configuration
├── requirements.txt             # Python dependencies
└── railway.json                 # Railway deployment config
```

## 🔧 Tech Stack

- **Backend Framework**: FastAPI
- **Machine Learning**: TensorFlow/PyTorch (for gesture recognition)
- **Monitoring**: Prometheus + Grafana
- **Containerization**: Docker
- **Deployment**: Railway
- **CI/CD**: GitHub Actions
- **Testing**: Pytest

## 📊 Monitoring & Metrics

The application includes comprehensive monitoring with the following key metrics:

### Core Prediction Metrics

1. **`predict_input_feature_count_count`**
   - Monitors frequency of prediction requests
   - Tracks API usage patterns
   - Detects service health issues

2. **`predict_input_feature_count_bucket`**
   - Histogram tracking distribution of input features
   - Validates correct feature size (expected: 57 features)
   - Identifies malformed input requests

3. **`predict_input_feature_count_sum`**
   - Total sum of feature counts across requests
   - Used to calculate average features per request
   - Helps validate input consistency

### Key Insights from Metrics

- **Service Health**: Steadily increasing trends confirm active service
- **Input Validation**: Most requests should hit the highest bucket (57 features)
- **Performance Monitoring**: Sudden spikes may indicate batch predictions or testing
- **Error Detection**: Flat lines or drops signal potential endpoint issues

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Docker and Docker Compose
- Git

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/TokaAyman/hand_gesture_maze_backend.git
   cd hand_gesture_maze_backend
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Access the API**
   - API: `http://localhost:8000`
   - Documentation: `http://localhost:8000/docs`
   - Metrics: `http://localhost:8000/metrics`

### Docker Deployment

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up -d
   ```

2. **Access services**
   - API: `http://localhost:8000`
   - Grafana: `http://localhost:3000`
   - Prometheus: `http://localhost:9090`

## 🔌 API Endpoints

### Health Check
- **GET** `/health` - Service health status

### Prediction
- **POST** `/predict` - Hand gesture prediction
  ```json
  {
    "features": [/* 57 landmark coordinates */]
  }
  ```

### Metrics
- **GET** `/metrics` - Prometheus metrics endpoint

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/

# Run specific test file
pytest tests/test_api.py
```

## 📈 Grafana Dashboards

The monitoring stack includes pre-configured dashboards for:

- **API Performance**: Request rates, response times, error rates
- **Model Metrics**: Prediction accuracy, feature validation
- **System Health**: Resource usage, service availability
- **Usage Analytics**: Request patterns, user behavior

## 🚢 Deployment

### Railway Deployment

The application is configured for automatic deployment on Railway:

1. Connect your GitHub repository to Railway
2. Railway will automatically detect the `railway.json` configuration
3. The CI/CD pipeline will trigger on pushes to main branch

### CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/deploy.yml`) handles:

- Automated testing
- Docker image building
- Deployment to Railway
- Health checks post-deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📋 Environment Variables

```bash
# Application
PORT=8000
HOST=0.0.0.0

# Model Configuration
MODEL_PATH=./model/
FEATURE_COUNT=57

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
```

## 🔍 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure model files are present in the `model/` directory
   - Check model compatibility with current TensorFlow/PyTorch version

2. **Feature Count Mismatch**
   - Verify input contains exactly 57 features
   - Check hand landmark extraction pipeline

3. **Metrics Not Updating**
   - Confirm Prometheus is scraping the `/metrics` endpoint
   - Verify Grafana data source configuration

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Toka Ayman** 
- GitHub: [@TokaAyman](https://github.com/TokaAyman)
- Repository: [hand_gesture_maze_backend](https://github.com/TokaAyman/hand_gesture_maze_backend)

## 🙏 Acknowledgments

- FastAPI for the excellent web framework
- Prometheus and Grafana for monitoring capabilities
- Railway for seamless deployment platform
- MediaPipe for hand landmark detection capabilities

---

⭐ **Star this repository if you found it helpful!**
