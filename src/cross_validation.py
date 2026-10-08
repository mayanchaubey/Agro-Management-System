import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_validate


def cross_validate_model(model, X, y):
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = {
        "accuracy": "accuracy",
        "macro_f1": "f1_macro",
        "weighted_f1": "f1_weighted"
    }

    results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    return {
        "cv_accuracy_mean": np.mean(results["test_accuracy"]),
        "cv_accuracy_std": np.std(results["test_accuracy"]),
        "cv_macro_f1_mean": np.mean(results["test_macro_f1"]),
        "cv_macro_f1_std": np.std(results["test_macro_f1"]),
        "cv_weighted_f1_mean": np.mean(results["test_weighted_f1"]),
        "cv_weighted_f1_std": np.std(results["test_weighted_f1"])
    }

if __name__ == "__main__":
    from data_preparation import load_data
    from models import get_models
    from train import build_pipeline

    X, y = load_data()

    models = get_models()

    model = build_pipeline(models["random_forest"])

    results = cross_validate_model(model, X, y)

    for name, value in results.items():
        print(f"{name}: {value:.4f}")
    print(f"cv_accuracy_mean: {results['cv_accuracy_mean']:.4f}")