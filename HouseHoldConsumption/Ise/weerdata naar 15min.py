import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv("sprint3/data/open-meteo-51.49N0.16W23m (1).csv", skiprows=3)
df = df.drop(columns=["Unnamed: 3", "Unnamed: 4", "Unnamed: 5"], errors="ignore")
df["time"] = pd.to_datetime(df["time"])
df = df.set_index("time").sort_index()


df_15min = df.resample("15min").mean()
df_15min = df_15min.interpolate(method="time")


df_H = pd.read_csv("sprint3/data/House_6.csv")

# Detect the time column name automatically and parse it
time_col = df_H.columns[0]
df_H[time_col] = pd.to_datetime(df_H[time_col])
df_H = df_H.set_index(time_col).sort_index()


df_merged = df_H.join(df_15min, how="left")


df_merged.to_csv("sprint3/data/whederdataHous_6.csv")

print(f"Merged shape: {df_merged.shape}")
print(df_merged.head())