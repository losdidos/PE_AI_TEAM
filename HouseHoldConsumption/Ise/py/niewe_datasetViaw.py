import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("sprint2/niewsetData/clean_dataset 1(in).csv")

print(df.head())
print(df.describe())
print(df.isnull().sum())

# volledige tijdreeks maken (elke 15 min)
full_range = pd.date_range(
    start=df['timestamp'].min(),
    end=df['timestamp'].max(),
    freq='15min'
)

# ontbrekende timestamps vinden
missing = full_range.difference(df['timestamp'])

print(missing)
print("Aantal ontbrekende momenten:", len(missing))