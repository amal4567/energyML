import pandas as pd
import numpy as np

df = pd.read_csv("data/cleaned_dataset.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])

df["month"] = df["timestamp"].dt.month
df["day"] = df["timestamp"].dt.day
df["hour"] = df["timestamp"].dt.hour
df["weekday"] = df["timestamp"].dt.weekday

df["is_weekend"] = (
    df["weekday"] >= 5
).astype(int)

df["building_age"] = (
    2025 - df["year_built"]
)

df["log_square_feet"] = np.log1p(
    df["square_feet"]
)

df["log_meter_reading"] = np.log1p(
    df["meter_reading"]
)

df.drop(
    columns=["timestamp"],
    inplace=True
)

df.to_csv(
    "data/featured_dataset.csv",
    index=False
)

print("Feature Engineering Finished")
