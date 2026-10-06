from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


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