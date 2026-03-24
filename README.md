# Air Quality Index (AQI) Prediction

Predict Air Quality Index based on pollutant concentrations using Machine Learning. The project compares multiple models (Linear Regression, Random Forest, Gradient Boosting, XGBoost) and selects the best performer.

## Project Structure

```
├── data/                  # Dataset (air quality data.csv)
├── models/                # Trained models & metadata
│   ├── best_model.pkl
│   ├── feature_importance.json
│   └── model_metadata.json
├── notebooks/             # Jupyter notebook for EDA
├── src/
│   ├── aqimodel.py        # Streamlit web application
│   └── train.py           # Model training pipeline
├── tests/
│   └── test_model.py      # Unit tests
├── docs/                  # Presentation files
├── Dockerfile             # Docker containerization
├── requirements.txt       # Python dependencies
└── README.md
```

## Features

- **Multi-model comparison**: Trains & evaluates Linear Regression, Random Forest, Gradient Boosting, and XGBoost
- **Hyperparameter tuning**: Uses RandomizedSearchCV with TimeSeriesSplit cross-validation
- **Feature importance analysis**: Identifies which pollutants most affect AQI
- **Interactive web app**: Streamlit dashboard with sidebar inputs, AQI prediction, historical trends, and feature importance visualization
- **City-level encoding**: Leverages city data for improved predictions

## How to Run

### 1. Setup

```bash
# Create virtual environment
python -m venv venv

# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python src/train.py
```

This will:
- Load and preprocess the dataset
- Train 4 different models and compare their performance
- Tune the best model's hyperparameters
- Save `best_model.pkl`, `feature_importance.json`, and `model_metadata.json` to `models/`

### 3. Run the Streamlit App

```bash
streamlit run src/aqimodel.py
```

### 4. Run Tests

```bash
python -m pytest tests/ -v
```

### Using Docker

```bash
docker build -t aqi-prediction .
docker run -p 8501:8501 aqi-prediction
```

## Model Information

The training pipeline compares multiple algorithms and automatically selects the best one based on test RMSE. The app uses the winning model from `models/best_model.pkl`.

**Input Features**: PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene, Month, DayOfWeek, City
