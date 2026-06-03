import pandas as pd
import matplotlib.pyplot as plt
"""
"endsprint2/data/House_1.csv",
    "endsprint2/data/House_2.csv",
    "endsprint2/data/House_3.csv",
    #"endsprint2/data/House_4.csv",
    "endsprint2/data/House_5.csv",
    "endsprint2/data/House_6.csv",
    "endsprint2/data/House_7.csv",
    
    #"endsprint2/data/House_9.csv",
    #"endsprint2/data/House_10.csv",
    #"endsprint2/data/House_11.csv",
    "endsprint2/data/House_12.csv",
    "endsprint2/data/House_13.csv",
    "endsprint2/data/House_15.csv",
    "endsprint2/data/House_16.csv",
    "endsprint2/data/House_18.csv",
    "endsprint2/data/House_19.csv",
    "endsprint2/data/House_20.csv",
    #"endsprint2/data/House_21.csv"


"""

allCsv = [

    "endsprint2/data/House_5.csv",

]

dfs = []

# Lees CSVs
for file in allCsv:
    df = pd.read_csv(file)

    df["Time"] = pd.to_datetime(df["Time"])
    df = df.sort_values("Time")

    print(f"start resampling {file}")


    df = df.set_index("Time")

    df_15_Mi = df.resample('15min').mean()

    df_15_Mi = df_15_Mi.loc["2014-04-01":"2015-03-01"]

    dfs.append(df_15_Mi)

    print(f"Read {file}")
    ms = df_15_Mi.isnull().sum()
    print(f"Missing values:\n{ms}\n")

n = len(dfs)


min_time = min(df.index.min() for df in dfs)
max_time = max(df.index.max() for df in dfs)

fig, ax = plt.subplots(n, 1, figsize=(15, 4 * n), sharex=True)

# FIKS: maak altijd een lijst van axen
if n == 1:
    ax = [ax]


for i, df in enumerate(dfs):
    df.to_csv(f"endsprint2/data/CleerResampel_House_{i}.csv")


for i, df in enumerate(dfs):
    df = df.sort_index()


    ax[i].plot(df.index, df["Aggregate"], alpha=0.6)
    ax[i].set_title(f"House {i + 1}")
    ax[i].set_ylabel("Aggregate")
    ax[i].set_xlim(min_time, max_time)

ax[-1].set_xlabel("Time")

plt.tight_layout()
plt.show()