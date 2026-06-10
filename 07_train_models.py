import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv("data/model_dataset.csv")

print("Dataset Shape :", df.shape)

# =====================================
# FEATURES / TARGET
# =====================================

X = df.drop(
    columns=[
        "meter_reading",
        "log_meter_reading"
    ]
)

y = df["log_meter_reading"]

# =====================================
# TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Train Shape :", X_train.shape)
print("Test Shape :", X_test.shape)

# =====================================
# MODELS
# =====================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        ),

    "XGBoost":
        XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ),

    "LightGBM":
        LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=8,
            random_state=42
        )
}

# =====================================
# TRAINING
# =====================================

results = []

best_model = None
best_r2 = -999

for name, model in models.items():

    print("\n" + "="*50)
    print(name)
    print("="*50)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    print("MAE :", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2  :", round(r2, 4))

    results.append([
        name,
        mae,
        rmse,
        r2
    ])

    if r2 > best_r2:

        best_r2 = r2

        best_model = model

        best_name = name

# =====================================
# RESULTS TABLE
# =====================================

os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)
results_df = pd.DataFrame(

    results,

    columns=[
        "Model",
        "MAE",
        "RMSE",
        "R2"
    ]
)

results_df = results_df.sort_values(
    by="R2",
    ascending=False
)

print("\n")
print("="*60)
print("FINAL RESULTS")
print("="*60)

print(results_df)

# =====================================
# SAVE RESULTS
# =====================================

results_df.to_csv(
    "results/model_comparison.csv",
    index=False
)

# =====================================
# SAVE BEST MODEL
# =====================================


joblib.dump(
    best_model,
    "models/best_model.pkl"
)

print("\nBest Model :", best_name)
print("Best R2 :", round(best_r2, 4))

print("\nModel saved successfully")
