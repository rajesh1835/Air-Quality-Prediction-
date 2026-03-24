"""
AQI Prediction — Model Training Pipeline

Trains and compares multiple models (Linear Regression, Random Forest,
Gradient Boosting, XGBoost) with time-based cross-validation and
hyperparameter tuning. Exports the best model and feature importance.
"""

import os
import json
import logging
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "air quality data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
FEATURE_IMP_PATH = os.path.join(MODEL_DIR, "feature_importance.json")
MODEL_META_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

# Feature configuration
POLLUTANT_COLS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene",
]
TARGET_COL = "AQI"


# ===================================================================
# Data Loading & Preprocessing
# ===================================================================
def load_data(filepath: str) -> pd.DataFrame:
    """Load the air quality CSV."""
    logger.info("Loading data from %s", filepath)
    df = pd.read_csv(filepath)
    logger.info("Raw dataset: %d rows, %d columns", *df.shape)
    return df


def preprocess(df: pd.DataFrame):
    """Clean data, engineer features, and return X, y."""
    logger.info("Preprocessing data …")

    # Drop rows where AQI is missing
    df = df.dropna(subset=[TARGET_COL]).copy()
    logger.info("After dropping missing AQI: %d rows", len(df))

    # Parse dates
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)

    # Feature engineering — temporal
    df["Month"] = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek

    # Feature engineering — city encoding
    if "City" in df.columns:
        le = LabelEncoder()
        df["City_Encoded"] = le.fit_transform(df["City"])
        feature_cols = POLLUTANT_COLS + ["Month", "DayOfWeek", "City_Encoded"]
    else:
        feature_cols = POLLUTANT_COLS + ["Month", "DayOfWeek"]

    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()

    return X, y, feature_cols


# ===================================================================
# Train / Test Split (time-based)
# ===================================================================
def time_split(X, y, ratio=0.8):
    """80/20 chronological split."""
    n = int(len(X) * ratio)
    X_train, X_test = X.iloc[:n], X.iloc[n:]
    y_train, y_test = y.iloc[:n], y.iloc[n:]
    return X_train, X_test, y_train, y_test


def impute_missing(X_train, X_test):
    """Median imputation using training-set statistics only."""
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)
    return X_train, X_test, medians


# ===================================================================
# Model Definitions
# ===================================================================
def get_models():
    """Return a dict of model_name → estimator."""
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, max_depth=10, min_samples_split=10,
            random_state=42, n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=42,
        ),
    }

    # XGBoost — optional dependency
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=42, n_jobs=-1, verbosity=0,
        )
    except ImportError:
        logger.warning("xgboost not installed — skipping XGBoost model")

    return models


# ===================================================================
# Evaluation
# ===================================================================
def evaluate(model, X_train, X_test, y_train, y_test):
    """Return train/test RMSE and R²."""
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    return {
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, train_pred))),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, test_pred))),
        "train_r2": float(r2_score(y_train, train_pred)),
        "test_r2": float(r2_score(y_test, test_pred)),
    }


# ===================================================================
# Multi-Model Comparison
# ===================================================================
def compare_models(models, X_train, X_test, y_train, y_test):
    """Train each model, print results table, return results dict."""
    results = {}
    logger.info("")
    logger.info("=" * 70)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 70)
    header = f"{'Model':<22} {'Train RMSE':>12} {'Test RMSE':>12} {'Train R²':>10} {'Test R²':>10}"
    logger.info(header)
    logger.info("-" * 70)

    for name, model in models.items():
        logger.info("Training %s …", name)
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_train, X_test, y_train, y_test)
        results[name] = {"model": model, "metrics": metrics}

        logger.info(
            "%-22s %12.4f %12.4f %10.4f %10.4f",
            name,
            metrics["train_rmse"], metrics["test_rmse"],
            metrics["train_r2"], metrics["test_r2"],
        )

    logger.info("=" * 70)
    return results


# ===================================================================
# Hyperparameter Tuning
# ===================================================================
def tune_best_model(best_name, X_train, y_train):
    """Run RandomizedSearchCV on the best model family."""
    logger.info("")
    logger.info("Hyperparameter tuning for %s …", best_name)

    tscv = TimeSeriesSplit(n_splits=5)

    param_grids = {
        "RandomForest": {
            "estimator": RandomForestRegressor(random_state=42, n_jobs=-1),
            "params": {
                "n_estimators": [50, 100, 200, 300],
                "max_depth": [5, 10, 15, 20, None],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 2, 4, 8],
            },
        },
        "GradientBoosting": {
            "estimator": GradientBoostingRegressor(random_state=42),
            "params": {
                "n_estimators": [50, 100, 200, 300],
                "max_depth": [3, 5, 7, 10],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "min_samples_split": [2, 5, 10],
            },
        },
    }

    # XGBoost tuning
    try:
        from xgboost import XGBRegressor
        param_grids["XGBoost"] = {
            "estimator": XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
            "params": {
                "n_estimators": [50, 100, 200, 300],
                "max_depth": [3, 5, 7, 10],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "subsample": [0.7, 0.8, 0.9, 1.0],
            },
        }
    except ImportError:
        pass

    if best_name not in param_grids:
        logger.info("No tuning grid for %s — skipping", best_name)
        return None

    config = param_grids[best_name]
    search = RandomizedSearchCV(
        config["estimator"],
        config["params"],
        n_iter=20,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)

    logger.info("Best params: %s", search.best_params_)
    logger.info("Best CV RMSE: %.4f", -search.best_score_)
    return search.best_estimator_


# ===================================================================
# Feature Importance
# ===================================================================
def get_feature_importance(model, feature_cols):
    """Extract feature importance from tree-based models."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        return None

    imp_dict = {
        name: round(float(val), 6)
        for name, val in sorted(
            zip(feature_cols, importances), key=lambda x: x[1], reverse=True
        )
    }
    return imp_dict


# ===================================================================
# Save Artifacts
# ===================================================================
def save_artifacts(model, feature_importance, best_name, metrics):
    """Save model pickle, feature importance JSON, and metadata."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Model
    with open(BEST_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved to %s", BEST_MODEL_PATH)

    # Feature importance
    if feature_importance:
        with open(FEATURE_IMP_PATH, "w") as f:
            json.dump(feature_importance, f, indent=2)
        logger.info("Feature importance saved to %s", FEATURE_IMP_PATH)

    # Metadata
    metadata = {
        "model_name": best_name,
        "metrics": metrics,
        "features": list(feature_importance.keys()) if feature_importance else [],
    }
    with open(MODEL_META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved to %s", MODEL_META_PATH)


# ===================================================================
# Main Pipeline
# ===================================================================
def main():
    # 1. Load & preprocess
    if not os.path.exists(DATA_PATH):
        logger.error("Data file not found at %s", DATA_PATH)
        return

    df = load_data(DATA_PATH)
    X, y, feature_cols = preprocess(df)

    # 2. Split & impute
    X_train, X_test, y_train, y_test = time_split(X, y)
    X_train, X_test, medians = impute_missing(X_train, X_test)
    logger.info("Train: %d samples | Test: %d samples", len(X_train), len(X_test))

    # 3. Compare models
    models = get_models()
    results = compare_models(models, X_train, X_test, y_train, y_test)

    # 4. Pick best model (lowest test RMSE)
    best_name = min(results, key=lambda k: results[k]["metrics"]["test_rmse"])
    logger.info("")
    logger.info("🏆 Best model: %s (Test RMSE: %.4f)",
                best_name, results[best_name]["metrics"]["test_rmse"])

    # 5. Hyperparameter tuning on the best model
    tuned_model = tune_best_model(best_name, X_train, y_train)

    if tuned_model is not None:
        tuned_metrics = evaluate(tuned_model, X_train, X_test, y_train, y_test)
        logger.info("Tuned Test RMSE: %.4f  |  Tuned Test R²: %.4f",
                     tuned_metrics["test_rmse"], tuned_metrics["test_r2"])

        # Use tuned model if it's better
        if tuned_metrics["test_rmse"] < results[best_name]["metrics"]["test_rmse"]:
            logger.info("✅ Tuned model is better — using tuned version")
            final_model = tuned_model
            final_metrics = tuned_metrics
        else:
            logger.info("⚠️  Tuned model is not better — keeping original")
            final_model = results[best_name]["model"]
            final_metrics = results[best_name]["metrics"]
    else:
        final_model = results[best_name]["model"]
        final_metrics = results[best_name]["metrics"]

    # 6. Feature importance
    feat_imp = get_feature_importance(final_model, feature_cols)
    if feat_imp:
        logger.info("")
        logger.info("Feature Importance (top 5):")
        for i, (name, val) in enumerate(feat_imp.items()):
            if i >= 5:
                break
            logger.info("  %s: %.4f", name, val)

    # 7. Save everything
    save_artifacts(final_model, feat_imp, best_name, final_metrics)
    logger.info("")
    logger.info("✅ Training pipeline complete!")


if __name__ == "__main__":
    main()
