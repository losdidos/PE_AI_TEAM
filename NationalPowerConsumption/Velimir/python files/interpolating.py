import pandas as pd

# Load merged.csv
df = pd.read_csv("./merged.csv", sep=";")

# Parse datetime (your format)
df["DateUTC"] = pd.to_datetime(df["DateUTC"], format="%Y-%m-%d %H:%M:%S")

# Sort
df = df.sort_values("DateUTC")

# Combine rows that share the same DateUTC (mean of numeric columns)
df = df.groupby("DateUTC", as_index=False).mean()

# Set datetime as index (now unique)
df = df.set_index("DateUTC")

# Resample + interpolate
df_15min = df.resample("15min").interpolate(method="linear")

# Back to normal column
df_15min = df_15min.reset_index()

# Save
df_15min.to_csv("./merged1.csv", sep=";", index=False)