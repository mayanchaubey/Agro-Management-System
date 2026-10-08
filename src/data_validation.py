import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series


FEATURE_COLUMNS = [
    "Nitrogen",
    "Phosphorus",
    "Potassium",
    "Temperature",
    "Humidity",
    "pH_Value",
    "Rainfall"
]

TARGET_COLUMN = "Crop"

class CropSchema(pa.DataFrameModel):
    Nitrogen: Series[int] = pa.Field(ge=0)
    Phosphorus: Series[int] = pa.Field(ge=0)
    Potassium: Series[int] = pa.Field(ge=0)

    Temperature: Series[float]
    Humidity: Series[float] = pa.Field(ge=0, le=100)
    pH_Value: Series[float] = pa.Field(ge=0, le=14)
    Rainfall: Series[float] = pa.Field(ge=0)

    Crop: Series[str]
    Yield: Series[int]

def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    return CropSchema.validate(df)

if __name__ == "__main__":
    df = pd.read_csv("data/raw/Crop_Yield_Prediction.csv")

    #df.loc[0, "Humidity"] = 150
    #to test the validation, you can uncomment the above line to introduce an invalid value for Humidity.

    validated_df = validate_data(df)

    print("Data validation passed!")
    print(validated_df.shape)