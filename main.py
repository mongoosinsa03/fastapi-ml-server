from fastapi import FastAPI
import joblib

app = FastAPI()

# Load models
scaler = joblib.load("scaler.joblib")
kmeans = joblib.load("kmeans.joblib")

@app.get("/")
def home():
    return {"message": "FastAPI ML server alive!"}

@app.post("/predict")
def predict(features: dict):
    import numpy as np

    X = np.array([list(features.values())]).reshape(1, -1)
    X_scaled = scaler.transform(X)
    cluster = kmeans.predict(X_scaled)[0]

    return {"cluster": int(cluster)}
