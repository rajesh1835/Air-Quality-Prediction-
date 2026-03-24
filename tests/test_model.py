"""
Unit tests for the AQI Prediction model.

Run with:  python -m pytest tests/ -v
"""

import os
import pickle

import numpy as np
import pytest

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, "..")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")

# Try best_model.pkl first, fallback to improved_rf_model.pkl
BEST_MODEL = os.path.join(MODEL_DIR, "best_model.pkl")
LEGACY_MODEL = os.path.join(MODEL_DIR, "improved_rf_model.pkl")

MODEL_PATH = BEST_MODEL if os.path.exists(BEST_MODEL) else LEGACY_MODEL


# ===================================================================
# Fixtures
# ===================================================================
@pytest.fixture
def model():
    """Load the trained model."""
    assert os.path.exists(MODEL_PATH), (
        f"Model file not found at {MODEL_PATH}. "
        "Run `python src/train.py` first."
    )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@pytest.fixture
def sample_input_14():
    """Sample input with 14 features (12 pollutants + Month + DayOfWeek)."""
    return np.array([[
        50.0,   # PM2.5
        80.0,   # PM10
        10.0,   # NO
        25.0,   # NO2
        30.0,   # NOx
        15.0,   # NH3
        2.0,    # CO
        20.0,   # SO2
        40.0,   # O3
        3.0,    # Benzene
        5.0,    # Toluene
        1.0,    # Xylene
        6,      # Month (June)
        2,      # DayOfWeek (Wednesday)
    ]])


@pytest.fixture
def sample_input_15():
    """Sample input with 15 features (14 + City_Encoded)."""
    return np.array([[
        50.0, 80.0, 10.0, 25.0, 30.0,
        15.0, 2.0, 20.0, 40.0, 3.0, 5.0, 1.0,
        6, 2, 0,  # + City_Encoded
    ]])


# ===================================================================
# Tests
# ===================================================================
class TestModelLoading:
    """Tests for model file and loading."""

    def test_model_file_exists(self):
        """Model file should exist on disk."""
        assert os.path.exists(MODEL_PATH), f"Model not found: {MODEL_PATH}"

    def test_model_loads_successfully(self, model):
        """Model should load without errors."""
        assert model is not None

    def test_model_has_predict(self, model):
        """Model should have a predict method."""
        assert hasattr(model, "predict"), "Model missing predict() method"


class TestPrediction:
    """Tests for prediction output."""

    def test_prediction_returns_float(self, model):
        """Prediction should return a numeric value."""
        # Determine expected feature count from model
        n_features = _get_feature_count(model)
        sample = np.zeros((1, n_features))
        pred = model.predict(sample)[0]
        assert isinstance(pred, (int, float, np.integer, np.floating)), (
            f"Expected numeric, got {type(pred)}"
        )

    def test_prediction_in_reasonable_range(self, model):
        """Prediction should be in a plausible AQI range."""
        n_features = _get_feature_count(model)
        sample = np.ones((1, n_features)) * 50  # moderate values
        pred = model.predict(sample)[0]
        assert -100 <= pred <= 2000, f"AQI prediction {pred} outside [-100, 2000]"

    def test_prediction_varies_with_input(self, model):
        """Different inputs should produce different predictions."""
        n_features = _get_feature_count(model)
        low = np.zeros((1, n_features))
        high = np.ones((1, n_features)) * 200
        pred_low = model.predict(low)[0]
        pred_high = model.predict(high)[0]
        assert pred_low != pred_high, "Model predicts the same value for different inputs"

    def test_batch_prediction(self, model):
        """Model should handle batch predictions."""
        n_features = _get_feature_count(model)
        batch = np.random.rand(10, n_features) * 100
        preds = model.predict(batch)
        assert len(preds) == 10, f"Expected 10 predictions, got {len(preds)}"


class TestFeatureImportance:
    """Tests for feature importance artifacts."""

    def test_feature_importance_file_exists(self):
        """Feature importance JSON should exist if best_model exists."""
        imp_path = os.path.join(MODEL_DIR, "feature_importance.json")
        if os.path.exists(BEST_MODEL):
            assert os.path.exists(imp_path), (
                "feature_importance.json missing alongside best_model.pkl"
            )

    def test_metadata_file_exists(self):
        """Model metadata JSON should exist if best_model exists."""
        meta_path = os.path.join(MODEL_DIR, "model_metadata.json")
        if os.path.exists(BEST_MODEL):
            assert os.path.exists(meta_path), (
                "model_metadata.json missing alongside best_model.pkl"
            )


# ===================================================================
# Helpers
# ===================================================================
def _get_feature_count(model):
    """Infer the expected number of features from the model."""
    if hasattr(model, "n_features_in_"):
        return model.n_features_in_
    # Fallback: try 14 (12 pollutants + Month + DayOfWeek)
    return 14
