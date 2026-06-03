import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("./Sprint 4/household-data-appliances-new/CleerResampel_House_7.csv")

df["Time"] = pd.to_datetime(df["Time"])
df = df.sort_values("Time")
df = df.set_index("Time")

df_original = df.copy()
df_filled = df.copy()

#15 minutes data => 96 intervals
steps_in_day = 96
week = 7

columns = df_filled.columns

for col in columns:
    shift_list = []
    for d in range(1, week+1):
        #shift to previous day
        shift_list.append(df_original[col].shift(steps_in_day * d))
        #shift to next day
        shift_list.append(df_original[col].shift(-steps_in_day * d))
    #put the data side by side in a table and mean it
    average_same_day = pd.concat(shift_list, axis=1).mean(axis=1)

    df_filled[col] = df_filled[col].fillna(average_same_day)

#fallback if more than 7 days are missing

df_filled[columns] = df_filled[columns].interpolate(method="time")

plt.figure()
plt.plot(df_filled["Appliance1"])
plt.show()