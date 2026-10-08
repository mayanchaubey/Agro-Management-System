import joblib

from src.data_preparation import load_data
from src.models import get_models
from src.train import build_pipeline


def test_saved_model_can_predict(tmp_path):
    X, y = load_data()

    model = build_pipeline(get_models()["random_forest"])
    model.fit(X, y)

    model_path = tmp_path / "crop_recommendation_model.joblib"
    joblib.dump(model, model_path)

    loaded_model = joblib.load(model_path)

    sample = X.iloc[[0]]
    prediction = loaded_model.predict(sample)

    assert len(prediction) == 1
    assert prediction[0] in y.unique()