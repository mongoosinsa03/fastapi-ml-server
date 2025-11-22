from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# --------------------------
# Input Model
# --------------------------
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

# --------------------------
# FastAPI App
# --------------------------
app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello world"}

# --------------------------
# Load ML Models
# --------------------------
scaler = joblib.load("scaler.joblib")
kmeans = joblib.load("kmeans.joblib")

# --------------------------
# Prediction API
# --------------------------
@app.post("/predict")
def predict(data: InputData):
    # Convert input to array
    x = np.array([
        [
            data.carbohydrate, data.protein, data.fat,
            data.vitamin_a, data.thiamine, data.riboflavin,
            data.vitamin_c, data.calcium, data.iron
        ]
    ])

    # Scale
    x_scaled = scaler.transform(x)

    # Predict cluster
    cluster = kmeans.predict(x_scaled)[0]

    return {"cluster": int(cluster)}
