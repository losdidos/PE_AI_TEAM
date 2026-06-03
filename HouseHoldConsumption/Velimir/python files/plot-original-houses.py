import pandas as pd
from matplotlib import pyplot as plt


allCSVS = ["./Sprint 4/household-data-appliances-new/CleerResampel_House_7.csv",
           "./Sprint 4/household-data-appliances-new/CleerResampel_House_8.csv",
           "./Sprint 4/household-data-appliances-new/CleerResampel_House_9.csv",
           "./Sprint 4/household-data-appliances-new/CleerResampel_House_10.csv",
           "./Sprint 4/household-data-appliances-new/CleerResampel_House_11.csv",
           "./Sprint 4/household-data-appliances-new/CleerResampel_House_12.csv",
           "./Sprint 4/household-data-appliances-new/CleerResampel_House_13.csv"]

dfs = []

for file in allCSVS:
    df = pd.read_csv(file)
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.sort_values("Time")
    dfs.append(df)


col = 1
rows = len(dfs)
fig, ax = plt.subplots(rows,col,constrained_layout = True)

for i in range (rows):
    ax[i].set_title(f"House {i}")
    ax[i].plot(dfs[i]["Time"], dfs[i]["Aggregate"])
    

plt.show()
'''


plt.figure()
plt.plot(df["Time"], df["Aggregate"])


plt.xlabel("Time")
plt.ylabel("Aggregate")

plt.show()
'''

