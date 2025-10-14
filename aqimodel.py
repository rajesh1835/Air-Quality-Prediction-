import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px

# Load model with caching

@st.cache_resource
def load_model():
    with open("random_forest_aqi_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    return model

rf_model = load_model()


# Define features with defaults

features_with_defaults = {
    "PM2.5": 60.5,
    "PM10": 90.3,
    "NO": 25.4,
    "NO2": 40.1,
    "NOx": 60.7,
    "NH3": 10.2,
    "CO": 0.8,
    "SO2": 15.6,
    "O3": 30.5,
    "Benzene": 2.3,
    "Toulene": 5.4,
    "Xylene": 0.9
}

feature_names = list(features_with_defaults.keys())


st.set_page_config(page_title="AQI Prediction", page_icon="🌍", layout="wide")

st.title("🌍 Air Quality Index (AQI) Prediction Dashboard")
st.markdown("### Enter environmental factors to predict AQI")
st.divider()

st.subheader("🔢 Input Air Quality Parameters")

cols = st.columns(2)
user_input = []

for i, (feature, default_value) in enumerate(features_with_defaults.items()):
    col = cols[i % 2]
    with col:
        value = st.slider(f"{feature}", 0.0, 500.0, float(default_value))
        user_input.append(value)


if st.button("🔮 Predict AQI", use_container_width=True):
    user_input_array = np.array(user_input).reshape(1, -1)
    prediction = rf_model.predict(user_input_array)[0]

    # Display Result
    st.subheader("🌫️ Predicted AQI")
    st.success(f"**Predicted Air Quality Index (AQI): {prediction:.2f}**")

    # AQI Category & Color
    if prediction <= 50:
        category, color, msg = "Good", "green", "Air quality is satisfactory."
    elif prediction <= 100:
        category, color, msg = "Moderate", "yellow", "Acceptable air quality."
    elif prediction <= 150:
        category, color, msg = "Unhealthy for Sensitive Groups", "orange", "Sensitive groups may experience effects."
    elif prediction <= 200:
        category, color, msg = "Unhealthy", "red", "Everyone may start to experience health effects."
    elif prediction <= 300:
        category, color, msg = "Very Unhealthy", "purple", "Health warnings of emergency conditions."
    else:
        category, color, msg = "Hazardous", "maroon", "Serious health effects. Stay indoors!"

    st.markdown(f"### 🟩 **{category}**")
    st.markdown(f"**{msg}**")


    st.subheader("📊 AQI Severity Level")

    gauge_df = pd.DataFrame({
        "AQI": [prediction],
        "Category": [category]
    })

    fig = px.bar(
        gauge_df,
        x=["Good", "Moderate", "Unhealthy for Sensitive", "Unhealthy", "Very Unhealthy", "Hazardous"],
        y=[prediction, 0, 0, 0, 0, 0],
        title="AQI Gauge Visualization",
        labels={"x": "Category", "y": "AQI Value"},
        color_discrete_sequence=[color]
    )
    fig.update_yaxes(range=[0, 500])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 Feature Importance")

    importance = rf_model.feature_importances_
    imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importance})
    imp_df = imp_df.sort_values(by="Importance", ascending=False)

    fig2 = px.bar(
        imp_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance of Air Quality Factors",
        color="Importance",
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.markdown(
    "<center>Developed with ❤️ using Streamlit | Random Forest Model for AQI Prediction</center>",
    unsafe_allow_html=True,
)

