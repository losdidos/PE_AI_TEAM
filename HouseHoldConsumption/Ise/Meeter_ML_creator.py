import pandas as pd
import matplotlib.pyplot as plt

df2 = pd.read_csv("sprint2/meterData/output.csv")



df2.columns = df2.columns.str.strip()



df2["DateTime"] = pd.to_datetime(df2["DateTime"])



nan_count = df2.isnull().sum()
print("=== NaN waarden per kolom ===")
print(nan_count)



df2["KWH/hh (per half hour)"] = pd.to_numeric(df2["KWH/hh (per half hour)"], errors="coerce")

nan_na_conversie = df2["KWH/hh (per half hour)"].isnull().sum()
print(f"\nNaN na 'Null' string conversie: {nan_na_conversie}")



verwacht = pd.date_range(start=df2["DateTime"].min(),
                         end=df2["DateTime"].max(),
                         freq="30min")

ontbrekend = verwacht.difference(df2["DateTime"])
print(f"\n=== Missende metingen ===")
print(f"Verwacht:   {len(verwacht)}")
print(f"Aanwezig:   {len(df2)}")
print(f"Ontbrekend: {len(ontbrekend)}")

if len(ontbrekend) > 0:
    print("\nEerste 10 ontbrekende tijdstippen:")
    print(ontbrekend[:10])



"""
plt.figure(figsize=(14, 5))
plt.plot(df2["DateTime"], df2["KWH/hh (per half hour)"], linewidth=0.8)

plt.title("Energieverbruik MAC000002")
plt.xlabel("DateTime")
plt.ylabel("KWH/hh (per half hour)")
plt.tight_layout()
plt.show()
"""


