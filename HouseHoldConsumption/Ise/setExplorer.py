import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

allCsv = [
    "endsprint2/data/cleerResampelData/set1/CleerResampel_House_0.csv",
    "endsprint2/data/cleerResampelData/set1/CleerResampel_House_1.csv",
    "endsprint2/data/cleerResampelData/set1/CleerResampel_House_2.csv",
    "endsprint2/data/cleerResampelData/set1/CleerResampel_House_3.csv",
    "endsprint2/data/cleerResampelData/set1/CleerResampel_House_4.csv",
    "endsprint2/data/cleerResampelData/set1/CleerResampel_House_5.csv",
    "endsprint2/data/cleerResampelData/set1/CleerResampel_House_6.csv",

]

dfs = []


# Lees CSVs
for file in allCsv:
    df = pd.read_csv(file)



    df["Time"] = pd.to_datetime(df["Time"])  # Zorg dat datetime correct is
    df = df.set_index("Time")
    dfs.append(df)
    print(f"Read {file}")
    ms = df.isnull().sum()
    print(f"The number of mising velus is {ms}")




n = len(dfs)

# correcte globale tijd
min_time = min(df.index.min() for df in dfs)
max_time = max(df.index.max() for df in dfs)

# subplots met gedeelde x-as
fig, ax = plt.subplots(n, 1, figsize=(15, 4 * n), sharex=True)




for i, df in enumerate(dfs):
    df = df.sort_index()

    ax[i].plot(df.index, df["Aggregate"], alpha=0.6)
    ax[i].set_title(f"House {i + 1}")
    ax[i].set_ylabel("Aggregate")
    ax[i].set_xlim(min_time, max_time)

ax[-1].set_xlabel("Time")

plt.tight_layout()
plt.show()
