import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def vind_missing_blokken(series):
    missing = series.isna()
    groepen = (missing != missing.shift()).cumsum()

    blokken = []
    for _, groep in series.groupby(groepen):
        if groep.isna().all():
            blokken.append((groep.index[0], groep.index[-1]))

    return blokken



def interpolate_sliding(df, kolom):
    df = df.copy()

    blokken = vind_missing_blokken(df[kolom])

    offsets = [
        pd.Timedelta(days=1),
        pd.Timedelta(days=7),
        pd.Timedelta(days=14),
        pd.Timedelta(days=21),
    ]

    for start, eind in blokken:
        idx_missing = df.loc[start:eind].index

        for ts in idx_missing:
            waarden = []

            for offset in offsets:
                voor = ts - offset
                na = ts + offset

                if voor in df.index and not pd.isna(df.loc[voor, kolom]):
                    waarden.append(df.loc[voor, kolom])

                if na in df.index and not pd.isna(df.loc[na, kolom]):
                    waarden.append(df.loc[na, kolom])

            if len(waarden) > 0:
                df.loc[ts, kolom] = np.mean(waarden)

    return df


def interpolate_all_columns(df):
    df_filled = df.copy()

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df_filled[col] = interpolate_sliding(df, col)[col]

    return df_filled




allCsv = [
    "sprint3/set1/CleerResampel_House_0.csv",
    "sprint3/set1/CleerResampel_House_1.csv",
    "sprint3/set1/CleerResampel_House_2.csv",
    "sprint3/set1/CleerResampel_House_3.csv",
    "sprint3/set1/CleerResampel_House_4.csv",
    "sprint3/set1/CleerResampel_House_5.csv",
    "sprint3/set1/CleerResampel_House_6.csv",
]

dfs = []



for i, file in enumerate(allCsv):
    df = pd.read_csv(file)

    df["Time"] = pd.to_datetime(df["Time"])
    df = df.set_index("Time")

    # resample naar 15 min
    df2 = df.asfreq("15min")

    # alle kolommen invullen
    df2_filled = interpolate_all_columns(df2)

    # extra kolom voor visualisatie (Aggregate)
    df2["Aggregate_filled"] = df2_filled["Aggregate"]

    dfs.append(df2)

    # opslaan
    df2_filled.to_csv(f"sprint3/data/House_{i}.csv")





n = len(dfs)

min_time = min(df.index.min() for df in dfs)
max_time = max(df.index.max() for df in dfs)

fig, ax = plt.subplots(n, 1, figsize=(15, 4 * n), sharex=True)

for i, df in enumerate(dfs):
    df = df.sort_index()



    ax[i].plot(df.index, df["Aggregate"], alpha=0.4, label="Original")



    ax[i].plot(df.index, df["Aggregate_filled"], linewidth=2, label="Filled")



    missing = df["Aggregate"].isna()
    ax[i].scatter(
        df.index[missing],
        df["Aggregate_filled"][missing],
        label="Imputed",
        zorder=3
    )

    ax[i].set_title(f"House {i + 1}")
    ax[i].set_ylabel("Aggregate")
    ax[i].set_xlim(min_time, max_time)
    ax[i].legend()

ax[-1].set_xlabel("Time")

plt.tight_layout()
plt.show()