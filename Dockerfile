FROM python:3.11-slim

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

COPY models/crop_recommendation_model.joblib models/crop_recommendation_model.joblib
