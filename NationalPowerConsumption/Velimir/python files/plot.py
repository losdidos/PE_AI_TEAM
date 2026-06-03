import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./merged.csv", sep =";")

# Convert DateUTC to datetime
df["DateUTC"] = pd.to_datetime(df["DateUTC"], format="%Y-%m-%d %H:%M:%S")

# Sort just in case
df = df.sort_values("DateUTC")

# Plot
plt.figure()
plt.plot(df["DateUTC"], df["Value_norm"])

plt.xlabel("Date")
plt.ylabel("Electricity Load")
plt.title("Electricity Load Over Time")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()