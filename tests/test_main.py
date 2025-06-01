# File: tests/test_main.py

import pytest
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import sys
import os

# Add the parent directory to the path to import the app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app, GesturePredictor, validate_landmarks, LABEL_MAP
from app.utils import preprocess_landmarks
from app.main import Landmark

client = TestClient(app)

class TestAPI:
    """Test the FastAPI endpoints"""
    
    def test_root_endpoint(self):
        """Test the root endpoint returns correct information"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["version"] == "1.0.0"
    
    def test_health_endpoint(self):
        """Test the health endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "timestamp" in data
    
    def test_gestures_endpoint(self):
        """Test the gestures endpoint returns valid gestures"""
        response = client.get("/gestures")
        assert response.status_code == 200
        data = response.json()
        assert "gestures" in data
        assert "count" in data
        assert data["count"] == 4
        assert set(data["gestures"]) == {"up", "down", "left", "right"}
    
    @patch('app.main.predictor')
    def test_predict_endpoint_success(self, mock_predictor):
        """Test successful prediction"""
        # Mock the predictor
        mock_predictor.is_loaded = True
        mock_predictor.predict.return_value = ("up", 0.95, {"up": 0.95, "down": 0.02, "left": 0.02, "right": 0.01})
        
        # Create test landmarks
        landmarks = [{"x": 0.1 + i*0.01, "y": 0.2 + i*0.01, "z": 0.3 + i*0.01} for i in range(21)]
        
        response = client.post("/predict", json={"landmarks": landmarks})
        assert response.status_code == 200
        data = response.json()
        assert data["label"] == "up"
        assert data["confidence"] == 0.95
        assert "probabilities" in data
        assert "processing_time_ms" in data
    
    def test_predict_endpoint_invalid_landmarks(self):
        """Test prediction with invalid number of landmarks"""
        # Too few landmarks
        landmarks = [{"x": 0.1, "y": 0.2, "z": 0.3} for _ in range(20)]
        
        response = client.post("/predict", json={"landmarks": landmarks})
        assert response.status_code == 400
        assert "Invalid number of landmarks" in response.json()["error"]
    
    @patch('app.main.predictor')
    def test_predict_endpoint_model_not_loaded(self, mock_predictor):
        """Test prediction when model is not loaded"""
        mock_predictor.is_loaded = False
        
        landmarks = [{"x": 0.1, "y": 0.2, "z": 0.3} for _ in range(21)]
        
        response = client.post("/predict", json={"landmarks": landmarks})
        assert response.status_code == 503
        assert "Model not loaded" in response.json()["error"]


class TestGesturePredictor:
    """Test the GesturePredictor class"""
    
    def test_predictor_initialization(self):
        """Test that predictor initializes correctly"""
        predictor = GesturePredictor()
        assert predictor.model is None
        assert predictor.is_loaded is False
    
    @patch('joblib.load')
    def test_load_model_success(self, mock_joblib_load):
        """Test successful model loading"""
        mock_model = Mock()
        mock_joblib_load.return_value = mock_model
        
        predictor = GesturePredictor()
        predictor.load_model()
        
        assert predictor.model == mock_model
        assert predictor.is_loaded is True
        mock_joblib_load.assert_called_once()
    
    @patch('joblib.load')
    def test_load_model_failure(self, mock_joblib_load):
        """Test model loading failure"""
        mock_joblib_load.side_effect = Exception("File not found")
        
        predictor = GesturePredictor()
        predictor.load_model()
        
        assert predictor.model is None
        assert predictor.is_loaded is False
    
    @patch('app.main.preprocess_landmarks')
    def test_predict_method(self, mock_preprocess):
        """Test the predict method"""
        # Setup mocks
        mock_model = Mock()
        mock_model.predict_proba.return_value = np.array([[0.1, 0.7, 0.1, 0.1]])
        
        mock_preprocess.return_value = np.random.rand(1, 63)
        
        predictor = GesturePredictor()
        predictor.model = mock_model
        predictor.is_loaded = True
        
        # Create test landmarks
        landmarks = [Landmark(x=0.1 + i*0.01, y=0.2 + i*0.01, z=0.3 + i*0.01) for i in range(21)]
        
        label, confidence, probs = predictor.predict(landmarks)
        
        assert label == "left"  # Index 1 maps to "left"
        assert confidence == 0.7
        assert len(probs) == 4
        assert probs["left"] == 0.7


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_validate_landmarks_correct_count(self):
        """Test validation with correct number of landmarks"""
        landmarks = [Landmark(x=0.1, y=0.2, z=0.3) for _ in range(21)]
        assert validate_landmarks(landmarks) is True
    
    def test_validate_landmarks_incorrect_count(self):
        """Test validation with incorrect number of landmarks"""
        landmarks = [Landmark(x=0.1, y=0.2, z=0.3) for _ in range(20)]
        assert validate_landmarks(landmarks) is False
    
    def test_preprocess_landmarks_shape(self):
        """Test preprocessing returns correct shape"""
        landmarks = np.random.rand(21, 3)
        result = preprocess_landmarks(landmarks)
        assert result.shape == (1, 63)
    
    def test_preprocess_landmarks_invalid_shape(self):
        """Test preprocessing with invalid input shape"""
        landmarks = np.random.rand(20, 3)  # Wrong number of landmarks
        with pytest.raises(ValueError):
            preprocess_landmarks(landmarks)
    
    def test_preprocess_landmarks_zero_distance(self):
        """Test preprocessing handles zero distance correctly"""
        landmarks = np.zeros((21, 3))  # All landmarks at origin
        result = preprocess_landmarks(landmarks)
        assert result.shape == (1, 63)
        assert not np.any(np.isnan(result))  # Should not contain NaN values


class TestLabelMapping:
    """Test label mapping consistency"""
    
    def test_label_map_completeness(self):
        """Test that LABEL_MAP contains all expected labels"""
        expected_labels = {"up", "down", "left", "right"}
        actual_labels = set(LABEL_MAP.values())
        assert actual_labels == expected_labels
    
    def test_label_map_indices(self):
        """Test that LABEL_MAP has correct indices"""
        expected_indices = {0, 1, 2, 3}
        actual_indices = set(LABEL_MAP.keys())
        assert actual_indices == expected_indices


if __name__ == "__main__":
    pytest.main([__file__])