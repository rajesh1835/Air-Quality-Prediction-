import datetime
import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AQI Prediction",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "air quality data.csv")

# Try new model first, fallback to old one
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
LEGACY_MODEL_PATH = os.path.join(MODEL_DIR, "improved_rf_model.pkl")
FEAT_IMP_PATH = os.path.join(MODEL_DIR, "feature_importance.json")
META_PATH = os.path.join(MODEL_DIR, "model_metadata.json")


# ---------------------------------------------------------------------------
# Load model & metadata
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load the trained model (best_model.pkl or fallback)."""
    import pickle

    for path in [BEST_MODEL_PATH, LEGACY_MODEL_PATH]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f), os.path.basename(path)
    st.error("No model file found. Run `python src/train.py` first.")
    st.stop()


@st.cache_data
def load_feature_importance():
    """Load feature importance JSON if available."""
    if os.path.exists(FEAT_IMP_PATH):
        with open(FEAT_IMP_PATH) as f:
            return json.load(f)
    return None


@st.cache_data
def load_metadata():
    """Load model metadata JSON if available."""
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            return json.load(f)
    return None


@st.cache_data
def load_historical_data():
    """Load the original CSV for historical trends."""
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        return df
    return None


model, model_file = load_model()
feat_imp = load_feature_importance()
metadata = load_metadata()

# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------
POLLUTANT_FEATURES = [
    "PM2.5", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene",
]

SLIDER_RANGES = {
    "PM2.5": (0.0, 500.0, "μg/m³"),
    "PM10": (0.0, 600.0, "μg/m³"),
    "NO": (0.0, 150.0, "μg/m³"),
    "NO2": (0.0, 200.0, "μg/m³"),
    "NOx": (0.0, 250.0, "ppb"),
    "NH3": (0.0, 200.0, "μg/m³"),
    "CO": (0.0, 50.0, "mg/m³"),
    "SO2": (0.0, 100.0, "μg/m³"),
    "O3": (0.0, 200.0, "μg/m³"),
    "Benzene": (0.0, 50.0, "μg/m³"),
    "Toluene": (0.0, 200.0, "μg/m³"),
    "Xylene": (0.0, 50.0, "μg/m³"),
}

AQI_CATEGORIES = [
    (50,  "Good", "Air quality is satisfactory", "#28a745"),
    (100, "Moderate", "Acceptable air quality", "#ffc107"),
    (150, "Unhealthy for Sensitive Groups", "Vulnerable people may be affected", "#fd7e14"),
    (200, "Unhealthy", "Health effects for everyone", "#dc3545"),
    (300, "Very Unhealthy", "Emergency conditions", "#6f42c1"),
    (500, "Hazardous", "Serious health effects for all", "#800000"),
]


def classify_aqi(value):
    """Return (label, description, color) for an AQI value."""
    for threshold, label, desc, color in AQI_CATEGORIES:
        if value <= threshold:
            return label, desc, color
    return AQI_CATEGORIES[-1][1], AQI_CATEGORIES[-1][2], AQI_CATEGORIES[-1][3]


# ===================================================================
# SIDEBAR — Inputs
# ===================================================================
st.sidebar.title("🎛️ Input Parameters")
st.sidebar.markdown("Adjust pollutant concentrations below.")

user_input = []
for feature in POLLUTANT_FEATURES:
    min_val, max_val, unit = SLIDER_RANGES[feature]
    val = st.sidebar.slider(
        f"{feature} ({unit})",
        min_value=min_val,
        max_value=max_val,
        value=round((min_val + max_val) / 4, 1),
        step=0.1,
    )
    user_input.append(val)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📅 Prediction Date")
prediction_date = st.sidebar.date_input("Date", datetime.date.today())

# Check if model expects City_Encoded feature
model_expects_city = False
if metadata and "features" in metadata:
    if "City_Encoded" in metadata["features"]:
        model_expects_city = True

if model_expects_city:
    historical = load_historical_data()
    if historical is not None and "City" in historical.columns:
        cities = sorted(historical["City"].dropna().unique().tolist())
        selected_city = st.sidebar.selectbox("🏙️ City", cities)
    else:
        selected_city = None
        model_expects_city = False

# ===================================================================
# MAIN AREA
# ===================================================================
st.title("🌬️ Air Quality Index (AQI) Prediction")
st.markdown("Predict the Air Quality Index based on pollutant concentrations using machine learning.")

# ---- Tabs ----
tab_predict, tab_importance, tab_trends, tab_about = st.tabs(
    ["🔮 Prediction", "📊 Feature Importance", "📈 Historical Trends", "ℹ️ About"]
)

# ==========================  PREDICTION TAB  ==========================
with tab_predict:
    # Input validation
    all_zero = all(v == 0 for v in user_input)
    if all_zero:
        st.warning("⚠️ All pollutant values are zero. Please adjust the sliders in the sidebar for a meaningful prediction.")

    if st.button("🔮 Predict AQI", use_container_width=True, type="primary"):
        month = prediction_date.month
        day_of_week = prediction_date.weekday()
        final_input = user_input + [month, day_of_week]

        # Add city encoding if model expects it
        if model_expects_city:
            from sklearn.preprocessing import LabelEncoder
            historical = load_historical_data()
            if historical is not None:
                le = LabelEncoder()
                le.fit(historical["City"].dropna())
                city_code = int(le.transform([selected_city])[0])
                final_input.append(city_code)

        input_array = np.array(final_input).reshape(1, -1)

        try:
            prediction = model.predict(input_array)[0]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info("The model may expect a different number of features. Try re-running `python src/train.py`.")
            st.stop()

        # Display result
        label, desc, color = classify_aqi(prediction)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Predicted AQI", f"{prediction:.1f}")
            st.markdown(
                f'<div style="background-color:{color}; padding:10px; border-radius:8px; '
                f'color:white; text-align:center; font-weight:bold;">'
                f'{label}</div>',
                unsafe_allow_html=True,
            )
            st.caption(desc)

        with col2:
            # AQI gauge chart
            levels = [cat[1] for cat in AQI_CATEGORIES]
            thresholds = [cat[0] for cat in AQI_CATEGORIES]
            colors = [cat[3] for cat in AQI_CATEGORIES]

            fig = go.Figure(go.Bar(
                x=thresholds,
                y=levels,
                orientation="h",
                marker_color=colors,
                text=[f"≤ {t}" for t in thresholds],
                textposition="inside",
            ))
            # Add prediction line
            fig.add_vline(
                x=prediction, line_dash="dash", line_color="white", line_width=3,
                annotation_text=f"Your AQI: {prediction:.1f}",
                annotation_font_color="white",
            )
            fig.update_layout(
                title="AQI Category Scale",
                xaxis_title="AQI Value",
                yaxis_title="",
                height=350,
                template="plotly_dark",
            )
            st.plotly_chart(fig, use_container_width=True)


# ==========================  FEATURE IMPORTANCE TAB  ==========================
with tab_importance:
    if feat_imp:
        st.subheader("🔑 What Drives AQI the Most?")
        st.markdown("Feature importance shows which pollutants most influence the model's predictions.")

        features = list(feat_imp.keys())
        values = list(feat_imp.values())

        fig = px.bar(
            x=values,
            y=features,
            orientation="h",
            color=values,
            color_continuous_scale="RdYlGn_r",
            labels={"x": "Importance", "y": "Feature"},
        )
        fig.update_layout(
            height=500,
            yaxis=dict(autorange="reversed"),
            template="plotly_dark",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance data not available. Run `python src/train.py` to generate it.")


# ==========================  HISTORICAL TRENDS TAB  ==========================
with tab_trends:
    st.subheader("📈 Historical AQI Trends")
    historical = load_historical_data()

    if historical is not None and "AQI" in historical.columns:
        cities = sorted(historical["City"].dropna().unique().tolist())
        selected = st.multiselect("Select cities to compare", cities, default=cities[:3])

        if selected:
            filtered = historical[
                (historical["City"].isin(selected)) & (historical["AQI"].notna())
            ].copy()

            # Monthly average for smoother trends
            filtered["YearMonth"] = filtered["Date"].dt.to_period("M").astype(str)
            monthly = filtered.groupby(["YearMonth", "City"])["AQI"].mean().reset_index()

            fig = px.line(
                monthly,
                x="YearMonth",
                y="AQI",
                color="City",
                title="Monthly Average AQI by City",
                labels={"YearMonth": "Month", "AQI": "Average AQI"},
            )
            fig.update_layout(
                height=500,
                template="plotly_dark",
                xaxis_tickangle=-45,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Summary stats
            st.markdown("### Summary Statistics")
            summary = (
                filtered.groupby("City")["AQI"]
                .agg(["mean", "median", "min", "max", "count"])
                .round(1)
                .rename(columns={
                    "mean": "Mean AQI", "median": "Median AQI",
                    "min": "Min AQI", "max": "Max AQI", "count": "Data Points",
                })
            )
            st.dataframe(summary, use_container_width=True)
        else:
            st.info("Select at least one city to view trends.")
    else:
        st.warning("Historical data not available.")


# ==========================  ABOUT TAB  ==========================
with tab_about:
    st.subheader("ℹ️ About This Application")

    st.markdown("""
    ### What is AQI?
    The **Air Quality Index (AQI)** is a standardized indicator of air quality in a given area.
    It tells you how clean or polluted the air is, and what associated health effects might be
    of concern.

    ### AQI Categories
    | AQI Range | Category | Health Impact |
    |---|---|---|
    | 0–50 | 🟢 Good | Minimal impact |
    | 51–100 | 🟡 Moderate | Acceptable for most |
    | 101–150 | 🟠 Unhealthy for Sensitive Groups | Vulnerable people affected |
    | 151–200 | 🔴 Unhealthy | Health effects for everyone |
    | 201–300 | 🟣 Very Unhealthy | Emergency conditions |
    | 301–500 | 🟤 Hazardous | Serious health effects |

    ### Model Details
    """)

    if metadata:
        st.markdown(f"- **Algorithm**: {metadata.get('model_name', 'Unknown')}")
        metrics = metadata.get("metrics", {})
        if metrics:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Train RMSE", f"{metrics.get('train_rmse', 0):.2f}")
            col2.metric("Test RMSE", f"{metrics.get('test_rmse', 0):.2f}")
            col3.metric("Train R²", f"{metrics.get('train_r2', 0):.4f}")
            col4.metric("Test R²", f"{metrics.get('test_r2', 0):.4f}")
    else:
        st.info("Run `python src/train.py` to generate model metadata.")

    st.markdown(f"""
    ### Input Features
    The model uses **{len(POLLUTANT_FEATURES)} pollutant measurements** plus temporal
    features (month, day of week) to predict AQI.

    **Pollutants**: {', '.join(POLLUTANT_FEATURES)}

    ### How to Use
    1. Adjust the pollutant sliders in the **sidebar**
    2. Select a prediction date
    3. Click **Predict AQI** to get the result

    ---
    *Model file: `{model_file}`*
    """)
