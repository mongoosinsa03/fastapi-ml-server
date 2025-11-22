from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

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

scaler = joblib.load("scaler.joblib")
model = joblib.load("kmeans.joblib")

@app.get("/")
def root():
    return {"message": "Hello world"}

@app.post("/predict")
def predict(data: InputData):
    try:
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
