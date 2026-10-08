import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)


def calculate_metrics(y_test, y_pred):
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_precision": precision_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
    }


def create_classification_report(y_test, y_pred):
    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    return pd.DataFrame(report).T


def save_confusion_matrix(y_test, y_pred, path):
    fig, ax = plt.subplots(figsize=(12, 8))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        xticks_rotation=90,
        ax=ax,
    )

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)