import pandas as pd

print("Loading datasets...")

train = pd.read_csv("data/train.csv")

weather = pd.read_csv("data/weather_train.csv")

building = pd.read_csv("data/building_metadata.csv")

print("Sampling 15000 rows...")

train = train.sample(
    n=15000,
    random_state=42
)

print(train.shape)

print("Merge building metadata...")

df = train.merge(
    building,
    on="building_id",
    how="left"
)

print(df.shape)

print("Merge weather data...")

df = df.merge(
    weather,
    on=["site_id", "timestamp"],
    how="left"
)

print(df.shape)

df.to_csv(
    "data/final_energy_dataset.csv",
    index=False
)

print("Dataset saved.")
