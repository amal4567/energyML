import pandas as pd

train = pd.read_csv("train.csv", nrows=5)
weather = pd.read_csv("weather_train.csv", nrows=5)
building = pd.read_csv("building_metadata.csv", nrows=5)

print("TRAIN")
print(train.columns.tolist())

print("\nWEATHER")
print(weather.columns.tolist())

print("\nBUILDING")
print(building.columns.tolist())
