import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

output_folder = "./Sprint 4/household-data-appliances-interpolated/"
allCSVS = ["./Sprint 4/household-data-appliances-new/CleerResampel_House_7.csv",
           "./Sprint 4/household-data-appliances-new/CleerResampel_House_8.csv",
           "./Sprint 4/household-data-appliances-new/CleerResampel_House_9.csv",
           "./Sprint 4/household-data-appliances-new/CleerResampel_House_10.csv",
           "./Sprint 4/household-data-appliances-new/CleerResampel_House_11.csv",
           "./Sprint 4/household-data-appliances-new/CleerResampel_House_12.csv",
           "./Sprint 4/household-data-appliances-new/CleerResampel_House_13.csv"]

dfs= []

for file in allCSVS:
    df = pd.read_csv(file)
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

    

    dfs.append(df_filled)
    #Saving the csvs
    filename = os.path.basename(file)
    output_path = os.path.join(output_folder, filename)
    df_filled.to_csv(output_path)




col = 1
rows = len(dfs)
fig, ax = plt.subplots(rows,col,constrained_layout = True)

for i in range (rows):
    ax[i].set_title(f"House {i}")
    ax[i].plot(dfs[i].index, dfs[i]["Aggregate"])
    

plt.show()