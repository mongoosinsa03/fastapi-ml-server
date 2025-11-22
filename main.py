from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

# Load models
scaler = joblib.load("scaler.joblib")
kmeans = joblib.load("kmeans.joblib")

class NutritionInput(BaseModel):
    features: dict

@app.post("/predict")
def predict_cluster(input_data: NutritionInput):
    ordered_keys = [
        "탄수화물(g)",
        "단백질(g)",
        "지방(g)",
        "비타민 A(μg RAE)",
        "티아민(mg)",
        "리보플라빈(mg)",
        "비타민 C(mg)",
        "칼슘(mg)",
        "철(mg)"
    ]

    x = np.array([[input_data.features[k] for k in ordered_keys]])
    x_scaled = scaler.transform(x)
    cluster = int(kmeans.predict(x_scaled)[0])
    return {"cluster": cluster}
