import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import numpy as np
import requests

# ✅ Replace this with your actual Render API URL
API_URL = "https://ecosentinel-wildfireai.onrender.com/predict"

st.set_page_config(page_title="EcoSentinel Dashboard", layout="wide")

st.title("🌍 EcoSentinel Wildfire Intelligence Dashboard")

tab1, tab2, tab3 = st.tabs(["🔥 Live API Predictor", "🗺️ Heatmap Viewer", "📁 Batch Upload"])

# ---------- TAB 1: Predict with Live API ----------
with tab1:
    st.header("🔥 Wildfire Confidence Prediction (via FastAPI)")

    scan = st.slider("Scan", 0.0, 5.0, 1.5, 0.1)
    track = st.slider("Track", 0.0, 5.0, 1.2, 0.1)
    latitude = st.slider("Latitude", -90.0, 90.0, 34.0, 0.1)
    longitude = st.slider("Longitude", -180.0, 180.0, -118.0, 0.1)

    if st.button("Predict"):
        payload = {
            "scan": scan,
            "track": track,
            "latitude": latitude,
            "longitude": longitude
        }

        try:
            res = requests.post(API_URL, json=payload)
            res.raise_for_status()
            confidence = res.json().get("predicted_confidence", "Unknown")
            st.success(f"🔥 API Response: **{confidence}**")
        except Exception as e:
            st.error(f"API Error: {e}")

# ---------- TAB 2: Wildfire Heatmap ----------
with tab2:
    st.header("🗺️ Wildfire Hotspot Heatmap")

    try:
        df = pd.read_csv("data/processed_wildfire_data.csv")
        if 'confidence' not in df.columns and 'confidence_class' in df.columns:
            df['confidence'] = df['confidence_class'].map({0: 'Low', 1: 'Nominal', 2: 'High'})

        map_center = [df['latitude'].mean(), df['longitude'].mean()]
        m = folium.Map(location=map_center, zoom_start=5)

        for _, row in df.iterrows():
            conf = str(row['confidence']).lower()
            color = 'green' if conf == 'low' else 'orange' if conf == 'nominal' else 'red'

            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=3,
                popup=f"Confidence: {row['confidence']}",
                color=color,
                fill=True,
                fill_opacity=0.7
            ).add_to(m)

        st_folium(m, width=1200, height=600)

    except FileNotFoundError:
        st.error("Processed data file not found. Run the pipeline first.")

# ---------- TAB 3: Batch Upload ----------
with tab3:
    st.header("📁 Batch Upload for Wildfire Prediction")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        required_cols = ['scan', 'track', 'latitude', 'longitude']
        try:
            df_input = pd.read_csv(uploaded_file)

            if not all(col in df_input.columns for col in required_cols):
                st.error(f"CSV must contain: {required_cols}")
            else:
                responses = []
                for _, row in df_input.iterrows():
                    payload = {
                        "scan": row["scan"],
                        "track": row["track"],
                        "latitude": row["latitude"],
                        "longitude": row["longitude"]
                    }
                    try:
                        r = requests.post(API_URL, json=payload)
                        r.raise_for_status()
                        prediction = r.json().get("predicted_confidence", "Unknown")
                        responses.append(prediction)
                    except:
                        responses.append("Error")

                df_input['predicted_confidence'] = responses
                st.success("✅ Predictions complete.")
                st.dataframe(df_input)

                csv = df_input.to_csv(index=False).encode("utf-8")
                st.download_button("Download Results CSV", csv, "wildfire_predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"Processing error: {e}")
