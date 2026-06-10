import pandas as pd

df = pd.read_csv(
    "data/final_energy_dataset.csv"
)

# supprimer floor_count

df.drop(
    columns=["floor_count"],
    inplace=True
)

# year_built

df["year_built"] = df["year_built"].fillna(
    df["year_built"].median()
)

# colonnes météo

weather_cols = [
    "air_temperature",
    "cloud_coverage",
    "dew_temperature",
    "precip_depth_1_hr",
    "sea_level_pressure",
    "wind_direction",
    "wind_speed"
]

for col in weather_cols:

    df[col] = df[col].fillna(
        df[col].median()
    )

print(df.isnull().sum())

df.to_csv(
    "data/cleaned_dataset.csv",
    index=False
)

print("Cleaning finished")
