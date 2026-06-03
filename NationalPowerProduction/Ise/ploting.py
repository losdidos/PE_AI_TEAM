import pandas as pd
import matplotlib.pyplot as plt

# CSV laden
df = pd.read_csv(
    "sprint1/filtered_BE_1V2.csv",
    sep=";"
)

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
plt.ylabel("Load Value")
plt.title("BE Load Time Series")

plt.xticks(rotation=45)
plt.grid(True)

plt.tight_layout()
plt.show()