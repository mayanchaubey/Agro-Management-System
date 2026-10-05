from sklearn.metrics import (
    accuracy_score,
    classification_report,
)

from data_preparation import load_data, split_data
from train import build_pipeline

import pandas as pd

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    report = classification_report(
    y_test,
    y_pred,
    output_dict=True
    )

    report_df = pd.DataFrame(report).T
    print(report_df)
    ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    xticks_rotation=90
    )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    X, y = load_data()

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = build_pipeline()
    model.fit(X_train, y_train)

    evaluate_model(model, X_test, y_test)