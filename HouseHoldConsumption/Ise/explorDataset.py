import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

allCsv = [
    "sprint3/set1/House_1.csv",
    "sprint3/set1/House_2.csv",
    "sprint3/set1/House_3.csv",
    "sprint3/set1/House_4.csv",
    "sprint3/set1/House_5.csv",
    "sprint3/set1/House_6.csv",
    "sprint3/set1/House_7.csv",


]

dfs = []



for file in allCsv:
    df = pd.read_csv(file)



    df["Time"] = pd.to_datetime(df["Time"])
    df = df.set_index("Time")
    dfs.append(df)
    print(f"Read {file}")
    ms = df.isnull().sum()
    print(f"The number of mising velus is {ms}")




n = len(dfs)




min_time = min(df.index.min() for df in dfs)
max_time = max(df.index.max() for df in dfs)


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
