# Air-Quality-Prediction-Model

📘 Overview

The Air Quality Index (AQI) Prediction project is a Machine Learning-based application that predicts the air quality level of a region based on various environmental parameters such as concentrations of pollutants (PM2.5, PM10, CO, NO₂, SO₂, O₃).

The goal of this project is to analyze air pollution data, build predictive models, and forecast AQI values to help monitor environmental health and support pollution control measures.

🎯 Objectives

To collect and preprocess air quality data from reliable sources.

To explore pollutant correlations through data visualization.

To train multiple machine learning models for AQI prediction.

To evaluate model performance and identify the most accurate algorithm.

To deploy a prediction interface for real-time AQI forecasting.

🧩 Machine Learning Workflow

Data Collection: Gathered historical air quality datasets from open data sources (e.g., Kaggle, CPCB, OpenAQ).

Data Preprocessing:

Handled missing values

Normalized numerical features

Encoded categorical values (if any)

Feature Selection: Selected pollutants that most affect AQI levels.

Model Selection: Compared multiple models:

Linear Regression

Random Forest Regressor

XGBoost Regressor

Support Vector Regressor (SVR)

Model Evaluation: Evaluated models using metrics like MAE, RMSE, and R² Score.

Prediction: Used the best-performing model to predict AQI based on new pollutant readings.


📊 Evaluation Metrics

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

R² Score (Coefficient of Determination)

✅ The model with the lowest RMSE and highest R² was selected for deployment.

⚙️ Tech Stack

Language: Python

Libraries: NumPy, Pandas, Matplotlib, Scikit-learn, XGBoost, Seaborn, Joblib

Tools: Jupyter Notebook / VS Code

Visualization: Matplotlib, Seaborn

🚀 Output

The model predicts the Air Quality Index (AQI) category for given pollutant concentrations, classifying the air as:

Good (0–50)

Moderate (51–100)

Unhealthy for Sensitive Groups (101–150)

Unhealthy (151–200)

Very Unhealthy (201–300)

Hazardous (301–500)

📈 Results

After testing multiple models:

Random Forest Regressor achieved the best accuracy and lowest RMSE.

Model performance improved significantly after feature scaling and hyperparameter tuning.

🧰 Future Enhancements

Integrate live air quality APIs for real-time prediction.

Build a web dashboard using Streamlit or React + Flask.

Extend model to multiple cities and climatic zones.

Implement alert systems for high AQI levels.

👨‍💻 Author
Rajesh T
