from fastapi import FastAPI
from pydantic import BaseModel, Field

import joblib
import pandas as pd

app = FastAPI(
    title="Crop Recommendation API",
    description="API for recommending crops based on soil and environmental conditions.",
    version="1.0.0",
)

class CropInput(BaseModel):
    Nitrogen: int = Field(ge=0)
    Phosphorus: int = Field(ge=0)
    Potassium: int = Field(ge=0)
    Temperature: float
    Humidity: float = Field(ge=0, le=100)
    pH_Value: float = Field(ge=0, le=14)
    Rainfall: float = Field(ge=0)

def load_model():
    return joblib.load("models/crop_recommendation_model.joblib")



@app.post("/predict")
def predict_crop(data: CropInput):
    input_data = pd.DataFrame([data.model_dump()])

    model = load_model()
    prediction = model.predict(input_data)

    return {
        "recommended_crop": prediction[0]
    }