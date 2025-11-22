from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

app = FastAPI()

# Load models
scaler = pickle.load(open("scaler.pkl", "rb"))
kmeans = pickle.load(open("kmeans.pkl", "rb"))

# Input schema
class NutritionInput(BaseModel):
    features: dict   # {"탄수화물(g)": 12, ... }

# Predict endpoint
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

    # Convert to vector
    x = np.array([[input_data.features[k] for k in ordered_keys]])
    x_scaled = scaler.transform(x)
    cluster = int(kmeans.predict(x_scaled)[0])

    return {"cluster": cluster}
