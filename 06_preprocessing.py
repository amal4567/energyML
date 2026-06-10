import pandas as pd

df = pd.read_csv(
    "data/featured_dataset.csv"
)

df = pd.get_dummies(
    df,
    columns=["primary_use"],
    drop_first=True
)

df.to_csv(
    "data/model_dataset.csv",
    index=False
)

print("Encoding Finished")
