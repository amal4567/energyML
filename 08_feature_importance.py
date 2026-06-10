import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Charger données
df = pd.read_csv("data/model_dataset.csv")

X = df.drop(
    columns=[
        "meter_reading",
        "log_meter_reading"
    ]
)

# Charger meilleur modèle
model = joblib.load(
    "models/best_model.pkl"
)

# Importance
importance = pd.DataFrame({

    "Feature": X.columns,
    "Importance": model.feature_importances_

})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance.head(20))

# Graphique
plt.figure(figsize=(10, 8))

plt.barh(
    importance["Feature"].head(15)[::-1],
    importance["Importance"].head(15)[::-1]
)

plt.title(
    "Top 15 Most Important Features"
)

plt.xlabel("Importance")

plt.tight_layout()

plt.show()
