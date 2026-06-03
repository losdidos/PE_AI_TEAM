import pandas as pd

df = pd.read_csv("Datasetfor2025_modified.csv", sep=";")

# Convert to datetime first
df["DateUTC"] = pd.to_datetime(df["DateUTC"])

# Format as: day month year hour minute
df["DateUTC"] = df["DateUTC"].dt.strftime("%d-%m-%Y %H:%M")

df.to_csv("converted.csv", sep=";", index=False)