import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

DATA_PATH = "data/processed_wildfire_data.csv"
MODEL_PATH = "models/wildfire_rf_model.pkl"

def load_data():
    df = pd.read_csv(DATA_PATH)

    # Features based on your real dataset
    features = ['scan', 'track', 'latitude', 'longitude']
    df = df[features + ['confidence']]

    # Encode 'confidence' (text or number) into classification labels
    def map_confidence(val):
        if isinstance(val, str):
            val = val.strip().lower()
            if val == 'nominal':
                return 1
            elif val == 'low':
                return 0
            elif val == 'high':
                return 2
        elif isinstance(val, (int, float)):
            return min(int(val / 34), 2)  # Normalize: 0–100 → 0–2
        return 1  # fallback

    df['confidence_class'] = df['confidence'].apply(map_confidence)
    df = df.drop(columns=['confidence'])

    return df

def train_model(df):
    X = df.drop(columns=['confidence_class'])
    y = df['confidence_class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print("[MODEL] Classification Report:")
    from sklearn.utils.multiclass import unique_labels

    labels = unique_labels(y_test, preds)
    label_names = {0: 'Low', 1: 'Nominal', 2: 'High'}
    target_names = [label_names[i] for i in labels]

    print(classification_report(y_test, preds, target_names=target_names))
    joblib.dump(clf, MODEL_PATH)
    print(f"[MODEL] Saved to {MODEL_PATH}")

def main():
    df = load_data()
    train_model(df)

if __name__ == "__main__":
    main()
