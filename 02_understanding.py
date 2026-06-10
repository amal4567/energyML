import pandas as pd

df = pd.read_csv(
    "data/final_energy_dataset.csv"
)

print(df.head())

print(df.shape)

print(df.info())

print(df.describe())
