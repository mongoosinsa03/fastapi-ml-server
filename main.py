from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import os

app = FastAPI()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH = os.path.join(BASE_DIR, "scaler.joblib")
MODEL_PATH = os.path.join(BASE_DIR, "kmeans.joblib")


try:
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    scaler = None
    print("Scaler load error:", e)

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print("Model load error:", e)

class InputData(BaseModel):
    carbohydrate: float
    protein: float
    fat: float
    vitamin_a: float
    thiamine: float
    riboflavin: float
    vitamin_c: float
    calcium: float
    iron: float


@app.get("/")
def root():
    return {"message": "server running"}

@app.post("/predict")
def predict(data: InputData):
    try:
        if scaler is None:
            return {"error": "Scaler not loaded"}
        if model is None:
            return {"error": "Model not loaded"}

        arr = np.array([[
            data.carbohydrate,
            data.protein,
            data.fat,
            data.vitamin_a,
            data.thiamine,
            data.riboflavin,
            data.vitamin_c,
            data.calcium,
            data.iron
        ]])

        scaled = scaler.transform(arr)
        pred = model.predict(scaled)[0]

        return {"cluster": int(pred)}

    except Exception as e:
        return {"error": str(e)}
