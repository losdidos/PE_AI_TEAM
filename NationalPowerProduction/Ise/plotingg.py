import pandas as pd
import matplotlib.pyplot as plt

# CSV laden
df = pd.read_csv("filtered_BE_1V2.csv")

# Kolomnamen
df.columns = [
    "type",
    "datetime",
    "start",
    "end",
    "country",
    "code",
    "value",
    "value2"
]

# Datum converteren
df["datetime"] = pd.to_datetime(
    df["datetime"],
    format="%d-%m-%Y %H:%M"
)

# Plot
plt.figure(figsize=(12,6))

plt.plot(df["datetime"], df["value"])

plt.xlabel("Tijd")
plt.ylabel("Value")
plt.title("Time Series Load Values")

plt.xticks(rotation=45)
plt.grid(True)

plt.tight_layout()
plt.show()