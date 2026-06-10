import pandas as pd
import joblib

from sklearn.model_selection import (
    cross_val_score
)

df = pd.read_csv(
    "data/model_dataset.csv"
)

X = df.drop(
    columns=[
        "meter_reading",
        "log_meter_reading"
    ]
)

y = df["log_meter_reading"]

model = joblib.load(
    "models/best_model.pkl"
)

scores = cross_val_score(

    model,
    X,
    y,

    cv=5,

    scoring="r2"
)

print("Scores :", scores)

print(
    "Mean R2 :",
    scores.mean()
)

print(
    "Std :",
    scores.std()
)
