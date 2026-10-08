from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data_preparation import load_data, split_data


def build_pipeline(classifier):
    return Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", classifier)
])


if __name__ == "__main__":
    X, y = load_data()

    X_train, X_test, y_train, y_test = split_data(X, y)

    pipeline = build_pipeline()

    pipeline.fit(X_train, y_train)

    print("Model training completed!")