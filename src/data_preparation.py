import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_validation import (FEATURE_COLUMNS, TARGET_COLUMN, validate_data,)


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