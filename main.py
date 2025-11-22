from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

# 모델 로드
scaler = joblib.load("scaler.joblib")
model = joblib.load("kmeans.joblib")

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
    return {"message": "Hello World"}   # ← 반드시 dict로 반환해야 함!


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
        pred = int(model.predict(scaled)[0])

        return {"cluster": pred}

    except Exception as e:
        return {"error": str(e)}
