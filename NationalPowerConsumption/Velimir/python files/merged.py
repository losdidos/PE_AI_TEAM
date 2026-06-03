from pathlib import Path
import pandas as pd

csv1_path = "./csv files/merged1.csv"
csv2_path = "./csv files/quarter_dataframe_2.csv"

csv1 = pd.read_csv(csv1_path, sep=";")
csv2 = pd.read_csv(csv2_path, sep=",")

# Add the column from csv2 as a new column in csv1
csv1["temperature_2m"] = csv2["temperature_2m"].values  

# Optional: sort by date if needed
csv1["DateUTC"] = pd.to_datetime(csv1["DateUTC"], format="%Y-%m-%d %H:%M:%S")
csv1 = csv1.sort_values("DateUTC")

csv1.to_csv("./merged.csv", sep=";", index=False)