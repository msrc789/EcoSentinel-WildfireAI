# 🔥 EcoSentinel — Wildfire AI Intelligence Dashboard

EcoSentinel is a real-time AI-powered wildfire prediction system that integrates live satellite data processing, machine learning predictions, heatmap visualization, and REST API deployment — all hosted in the cloud.

## 🚀 Features

- ✅ **Live confidence prediction** (Low / Nominal / High)
- ✅ **Interactive wildfire heatmap** using real location data
- ✅ **Batch CSV uploads** for bulk predictions
- ✅ **REST API** powered by FastAPI, hosted on Render
- ✅ **Streamlit Dashboard** UI for public demo access

---

## 📊 Tech Stack

- **Backend**: Python, FastAPI, scikit-learn, joblib
- **Frontend**: Streamlit, Folium, Streamlit-Folium
- **Deployment**: Render (Free Tier)
- **ML**: RandomForestClassifier trained on NASA FIRMS-like data

---

## 🧠 How It Works

1. 🔄 Data Pipeline:
   - Preprocess wildfire records (`data_pipeline.py`)
   - Normalize and structure input features

2. 🤖 Model:
   - Trained Random Forest on `scan`, `track`, `latitude`, `longitude`
   - Output: Fire confidence level

3. 🌐 API:
   - `/predict` endpoint accepts JSON → returns confidence class

4. 📺 Streamlit Dashboard:
   - Single Prediction (via API)
   - Heatmap Viewer (Folium)
   - Batch Upload + Download Results

---

## 📦 Setup Instructions

```bash
git clone https://github.com/your-username/EcoSentinel-WildfireAI.git
cd EcoSentinel-WildfireAI

# Optional: setup virtual environment
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
streamlit run dashboard/app.py# EcoSentinel-WildfireAI
