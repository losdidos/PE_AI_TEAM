import pandas as pd

# Load weatherdata
df = pd.read_csv("./Sprint 4/weather-data/weather-data-household-original.csv", sep=",")

# Parse datetime (your format)
df["DateUTC"] = pd.to_datetime(df["time"], format="%Y-%m-%dT%H:%M")
df = df.drop(columns=["time"])
# Sort
df = df.sort_values("DateUTC")

# Set datetime as index 
df = df.set_index("DateUTC")

# Resample + interpolate
df_15min = df.resample("15min").interpolate(method="linear")

# Back to normal column
df_15min = df_15min.reset_index()

# Save
df_15min.to_csv("./Sprint 4/weather-data/weather-data-household-interpolated.csv", sep=",", index=False)