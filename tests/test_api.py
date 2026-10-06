from fastapi.testclient import TestClient

from api.main import app
import joblib

from src.data_preparation import load_data
from src.models import get_models
from src.train import build_pipeline




client = TestClient(app)

import pytest


@pytest.fixture
def test_model(tmp_path, monkeypatch):
    X, y = load_data()

    model = build_pipeline(get_models()["random_forest"])
    model.fit(X, y)

    model_path = tmp_path / "test_model.joblib"
    joblib.dump(model, model_path)

    monkeypatch.setattr(
        "api.main.load_model",
        lambda: joblib.load(model_path),
    )

    return model


def test_predict_crop():
    response = client.post(
        "/predict",
        json={
            "Nitrogen": 90,
            "Phosphorus": 42,
            "Potassium": 43,
            "Temperature": 20.8,
            "Humidity": 82.0,
            "pH_Value": 6.5,
            "Rainfall": 202.9,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "recommended_crop" in data
    assert isinstance(data["recommended_crop"], str)


def test_invalid_humidity():
    response = client.post(
        "/predict",
        json={
            "Nitrogen": 90,
            "Phosphorus": 42,
            "Potassium": 43,
            "Temperature": 20.8,
            "Humidity": 150,
            "pH_Value": 6.5,
            "Rainfall": 202.9,
        },
    )

    assert response.status_code == 422