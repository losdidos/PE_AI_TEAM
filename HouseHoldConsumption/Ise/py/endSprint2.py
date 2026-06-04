import pandas as pd
import matplotlib.pyplot as plt

"""

    4
    11
    12
    13
    

"""





allCsv = [
    "endsprint2/data/House_1.csv",
    "endsprint2/data/House_2.csv",
    "endsprint2/data/House_3.csv",
    #"endsprint2/data/House_4.csv",
    "endsprint2/data/House_5.csv",
    "endsprint2/data/House_6.csv",
    "endsprint2/data/House_7.csv",
    "endsprint2/data/House_8.csv",
    #"endsprint2/data/House_9.csv",
    #"endsprint2/data/House_10.csv",
    #"endsprint2/data/House_11.csv",
    #"endsprint2/data/House_12.csv",
    "endsprint2/data/House_13.csv",
    "endsprint2/data/House_15.csv",
    "endsprint2/data/House_16.csv",
    #"endsprint2/data/House_17.csv",
    #"endsprint2/data/House_18.csv",
    #"endsprint2/data/House_19.csv",
    #"endsprint2/data/House_20.csv",
    "endsprint2/data/House_21.csv"



]


"""
"endsprint2/data/House_1.csv",
    "endsprint2/data/House_2.csv",
    "endsprint2/data/House_3.csv",
    "endsprint2/data/House_4.csv",
    "endsprint2/data/House_5.csv",
    "endsprint2/data/House_6.csv",
    "endsprint2/data/House_7.csv",
    "endsprint2/data/House_8.csv",
    "endsprint2/data/House_9.csv",
    "endsprint2/data/House_10.csv",
    "endsprint2/data/House_11.csv",
    "endsprint2/data/House_12.csv",
    "endsprint2/data/House_13.csv",
    "endsprint2/data/House_15.csv",
    "endsprint2/data/House_16.csv",
    "endsprint2/data/House_17.csv",
    "endsprint2/data/House_18.csv",
    "endsprint2/data/House_19.csv",
    "endsprint2/data/House_20.csv",
    "endsprint2/data/House_21.csv"


"""





dfs = []


# Lees CSVs
for file in allCsv:
    df = pd.read_csv(file)

    df = df.set_index("Time")

    df["Time"] = pd.to_datetime(df["Time"])  # Zorg dat datetime correct is
    dfs.append(df)
    print(f"Read {file}")
    ms = df.isnull().sum()
    print(f"The number of mising velus is {ms}")




n = len(dfs)


min_time = min(df["Time"].min() for df in dfs)
max_time = max(df["Time"].max() for df in dfs)


fig, ax = plt.subplots(n, 1, figsize=(15, 4 * n), sharex=True) # subplots

for i, df in enumerate(dfs):
    df = df.sort_values("Time")

    ax[i].plot(df["Time"], df["Aggregate"], alpha=0.6)
    ax[i].set_title(f"House {i + 1}")
    ax[i].set_ylabel("Aggregate")
    ax[i].set_xlim(min_time, max_time)

ax[-1].set_xlabel("Time")

plt.tight_layout()
plt.show()

