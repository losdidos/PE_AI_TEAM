import pandas as pd
import matplotlib.pyplot as plt



df = pd.read_csv(r"C:\Users\NemoL\Documents\ThomasMore\TweedeJaar\troimester2_2\Practis enterpris\Data\sprint2\CC_LCL-FullData.csv")

print(df.columns.tolist())

df2 = df[df.LCLid == "MAC000002"].copy()

df2.columns = df2.columns.str.strip()
df2.drop(columns=["stdorToU"], inplace=True)

df2["DateTime"] = pd.to_datetime(df2["DateTime"])
df2 = df2.sort_values("DateTime").reset_index(drop=True)

df2["KWH/hh (per half hour)"] = pd.to_numeric(df2["KWH/hh (per half hour)"], errors="coerce")

print("\n=== NaN waarden per kolom ===")
print(df2.isnull().sum())



volledige_reeks = pd.DataFrame({"DateTime": pd.date_range(
    start=df2["DateTime"].min(),
    end=df2["DateTime"].max(),
    freq="30min"
)})

df2 = volledige_reeks.merge(df2, on="DateTime", how="left")



ontbrekende_mask = df2["KWH/hh (per half hour)"].isnull()

print(f"\n=== Missende metingen ===")
print(f"Verwacht:   {len(volledige_reeks)}")
print(f"Ontbrekend: {ontbrekende_mask.sum()}")

df2 = df2.set_index("DateTime")
df2["KWH/hh (per half hour)"] = df2["KWH/hh (per half hour)"].interpolate(method="time")
df2 = df2.reset_index()

print(f"NaN na interpolatie: {df2['KWH/hh (per half hour)'].isnull().sum()}")
print(f"\nRijen na opvullen: {len(df2)}")
print(df2.head(5))
print(df2.shape)
print(df2.columns.tolist())



df2.to_csv("User_MAC000002_data.csv", index=False)
print("\nSaved to User_MAC000002_data.csv")



gefabriceerd = df2[ontbrekende_mask]

plt.figure(figsize=(14, 5))
plt.plot(df2["DateTime"], df2["KWH/hh (per half hour)"],
         linewidth=0.8, color="steelblue", label="Gemeten")

plt.scatter(gefabriceerd["DateTime"], gefabriceerd["KWH/hh (per half hour)"],
            color="red", s=20, zorder=5, label=f"Geïnterpoleerd ({len(gefabriceerd)})")

plt.title("Energieverbruik MAC000002")
plt.xlabel("Datum")
plt.ylabel("KWH per half uur")
plt.legend()
plt.tight_layout()
plt.show()