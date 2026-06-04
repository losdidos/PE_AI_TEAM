import pandas as pd
import matplotlib.pyplot as plt

# Inlezen
df = pd.read_csv("endsprint2/data/House_5.csv")

# Tijd instellen
df['Time'] = pd.to_datetime(df['Time'])
df = df.set_index('Time').sort_index()

# Verwijder kolom die je niet nodig hebt
df = df.drop(columns=["Unix"])

# Resample naar 15 minuten (zonder aggregatie!)
df_15min = df.resample('15min').asfreq()

# Interpolatie (BELANGRIJK)
df_15min = df_15min.interpolate(method='time')

# Check
print(df_15min.head())
print(df_15min.isna().sum())

plt.figure()

tijd =  df_15min.index
waarden = df_15min['Aggregate']

plt.plot(tijd, waarden)

# === OPMAAK ===
plt.xlabel("Tijd")
plt.ylabel("Waarden")
plt.title("Mijn simpele plot")
plt.grid()


plt.show()