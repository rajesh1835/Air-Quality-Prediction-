import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open("random_forest_aqi_model.pkl", "rb") as model_file:
    rf_model = pickle.load(model_file)

# Define feature names
feature_names = ["PM2.5", "PM10", "NO", "NO2","NOx","NH3","CO","SO2","O3","Benzene","Toulene","Xylene"]

# Streamlit UI
st.title("🌍 Air Quality Index (AQI) Prediction")
st.write("Enter the environmental factors to predict AQI.")

# Create input fields for each feature
user_input = []
for feature in feature_names:
    value = st.number_input(f"{feature}",format="%.2f")
    user_input.append(value)

# Predict AQI on button click
if st.button("Predict AQI"):
    user_input_array = np.array(user_input).reshape(1, -1)
    prediction = rf_model.predict(user_input_array)[0]
    
    # Display AQI Prediction
    st.success(f"🟢 Predicted Air Quality Index (AQI): {prediction:.2f}")
    
    if prediction <= 50:
        st.info("✅ **Good** (Air quality is satisfactory)")
    elif prediction <= 100:
        st.warning("🟡 **Moderate** (Acceptable air quality)")
    elif prediction <= 150:
        st.warning("🟠 **Unhealthy for Sensitive Groups**")
    elif prediction <= 200:
        st.error("🔴 **Unhealthy** (Everyone may start to experience health effects)")
    elif prediction <= 300:
        st.error("🟣 **Very Unhealthy** (Health warnings of emergency conditions)")
    else:
        st.error("☠️ **Hazardous** (Serious health effects)")

# Run using: streamlit run app.py
