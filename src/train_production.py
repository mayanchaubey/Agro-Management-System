import joblib

from src.data_preparation import load_data
from src.models import get_models
from src.train import build_pipeline
from pathlib import Path

X, y = load_data()

models = get_models()

random_forest = models["random_forest"]

model = build_pipeline(random_forest)
model.fit(X, y)

model_path = "models/crop_recommendation_model.joblib"
Path("models").mkdir(parents=True, exist_ok=True)

joblib.dump(model, model_path)

print(f"Model saved to: {model_path}")
