import mlflow
from sklearn.metrics import accuracy_score, f1_score

from src.data_preparation import load_data, split_data
from src.train import build_pipeline

import mlflow.sklearn

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from src.models import get_models
import time
import mlflow.sklearn

from evaluate import (
    calculate_metrics,
    create_classification_report,
    save_confusion_matrix,
)

from cross_validation import cross_validate_model



mlflow.set_experiment("crop-recommendation")


if __name__ == "__main__":
    X, y = load_data()

    X_train, X_test, y_train, y_test = split_data(X, y)

    models = get_models()

    for model_name, classifier in models.items():

        with mlflow.start_run(run_name=model_name):

            model = build_pipeline(classifier)

            cv_results = cross_validate_model(model, X, y)

            print(f"\n{model_name}")
            print(f"cv_accuracy_mean: {cv_results['cv_accuracy_mean']:.4f}")
            print(f"cv_accuracy_std: {cv_results['cv_accuracy_std']:.4f}")
            print(f"cv_macro_f1_mean: {cv_results['cv_macro_f1_mean']:.4f}")
            print(f"cv_macro_f1_std: {cv_results['cv_macro_f1_std']:.4f}")

            for metric_name, value in cv_results.items():
                mlflow.log_metric(metric_name, value)

            print(f"\n{model_name}")

            for metric_name, value in cv_results.items():
                print(f"{metric_name}: {value:.4f}")

            start_time = time.time()

            model.fit(X_train, y_train)

            training_time = time.time() - start_time

            y_pred = model.predict(X_test)

            metrics = calculate_metrics(y_test, y_pred)

            # Parameters
            mlflow.log_param("model", model_name)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)

            # Metrics
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)

            mlflow.log_metric("training_time", training_time)

            # Classification report
            report = create_classification_report(y_test, y_pred)
            report.to_csv("classification_report.csv")
            mlflow.log_artifact("classification_report.csv")

            # Confusion matrix
            confusion_matrix_path = f"confusion_matrix_{model_name}.png"

            save_confusion_matrix(
                y_test,
                y_pred,
                confusion_matrix_path,
            )

            mlflow.log_artifact(confusion_matrix_path)

            # Model
            mlflow.sklearn.log_model(
                model,
                name="crop_recommendation_model",
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )

            print(f"\n{model_name}")

            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")

            print(f"training_time: {training_time:.4f}s")