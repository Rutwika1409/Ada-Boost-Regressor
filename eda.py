import pandas as pd
import os

os.makedirs("data", exist_ok=True)

df = pd.read_csv("data/housing.csv")

print("\nShape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicates:")
print(df.duplicated().sum())

df.drop_duplicates(inplace=True)

df["total_bedrooms"] = df["total_bedrooms"].fillna(
    df["total_bedrooms"].median()
)

df["rooms_per_household"] = (
    df["total_rooms"] / df["households"]
)

df["bedrooms_per_room"] = (
    df["total_bedrooms"] / df["total_rooms"]
)

df["population_per_household"] = (
    df["population"] / df["households"]
)

print("\nFinal Shape:")
print(df.shape)

df.to_csv(
    "data/cleaned_housing.csv",
    index=False
)

print("\nCleaned dataset saved successfully.")