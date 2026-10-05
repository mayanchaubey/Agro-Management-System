import pandas as pd
from sklearn.model_selection import train_test_split

from data_validation import FEATURE_COLUMNS, TARGET_COLUMN, validate_data

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def load_data():
    df = pd.read_csv("data/raw/Crop_Yield_Prediction.csv")
    df = validate_data(df)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    return X, y

def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

def build_pipeline():
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="linear"))
    ])

    return model

if __name__ == "__main__":
    X, y = load_data()

    X_train, X_test, y_train, y_test = split_data(X, y)

    pipeline = build_pipeline()

    pipeline.fit(X_train, y_train)

    print("Model training completed!")