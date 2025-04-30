import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import joblib
import numpy as np

st.set_page_config(page_title="EcoSentinel AI Dashboard", layout="wide")

st.title("🌍 EcoSentinel Wildfire Intelligence Dashboard")

# Load the model
model = joblib.load("models/wildfire_rf_model.pkl")

# Create tab layout
tab1, tab2, tab3 = st.tabs(["🔥 Predictor", "🗺️ Heatmap Viewer", "📁 Batch Upload"])

# ---------- TAB 1: Predictor ----------
with tab1:
    st.header("🔥 Single Wildfire Confidence Prediction")
    
    scan = st.slider("Scan", 0.0, 5.0, 1.5, 0.1)
    track = st.slider("Track", 0.0, 5.0, 1.2, 0.1)
    latitude = st.slider("Latitude", -90.0, 90.0, 34.0, 0.1)
    longitude = st.slider("Longitude", -180.0, 180.0, -118.0, 0.1)

    if st.button("Predict"):
        features = np.array([[scan, track, latitude, longitude]])
        prediction = model.predict(features)[0]
        label_map = {0: "Low", 1: "Nominal", 2: "High"}
        st.success(f"🔥 Predicted Confidence: **{label_map[prediction]}**")

# ---------- TAB 2: Heatmap Viewer ----------
with tab2:
    st.header("🗺️ Wildfire Hotspot Map")

    try:
        df = pd.read_csv("data/processed_wildfire_data.csv")
        if 'confidence' not in df.columns and 'confidence_class' in df.columns:
            df['confidence'] = df['confidence_class'].map({0: 'Low', 1: 'Nominal', 2: 'High'})

        map_center = [df['latitude'].mean(), df['longitude'].mean()]
        m = folium.Map(location=map_center, zoom_start=5)

        for _, row in df.iterrows():
            confidence = str(row['confidence']).strip().lower()
            color = 'green' if confidence == 'low' else 'orange' if confidence == 'nominal' else 'red'

            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=3,
                popup=f"Confidence: {row['confidence']}",
                color=color,
                fill=True,
                fill_opacity=0.7
            ).add_to(m)

        st_data = st_folium(m, width=1200, height=600)

    except FileNotFoundError:
        st.error("🔥 Processed data not found. Please run the data pipeline first.")

# ---------- TAB 3: Batch Upload ----------
with tab3:
    st.header("📁 Batch Wildfire Confidence Prediction")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        required_cols = ['scan', 'track', 'latitude', 'longitude']
        try:
            df_input = pd.read_csv(uploaded_file)

            if not all(col in df_input.columns for col in required_cols):
                st.error(f"CSV must contain columns: {required_cols}")
            else:
                preds = model.predict(df_input[required_cols])
                label_map = {0: "Low", 1: "Nominal", 2: "High"}
                df_input['predicted_confidence'] = [label_map[p] for p in preds]

                st.success("✅ Predictions complete.")
                st.dataframe(df_input)

                csv = df_input.to_csv(index=False).encode('utf-8')
                st.download_button("Download Results", csv, "wildfire_predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"Error processing file: {e}")
