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

print(f"\n=== Missende metingen (30 min) ===")
print(f"Verwacht:   {len(volledige_reeks)}")
print(f"Ontbrekend: {ontbrekende_mask.sum()}")

df2 = df2.set_index("DateTime")
df2["KWH/hh (per half hour)"] = df2["KWH/hh (per half hour)"].interpolate(method="time")
df2 = df2.reset_index()

print(f"NaN na interpolatie: {df2['KWH/hh (per half hour)'].isnull().sum()}")
print(f"\nRijen na opvullen (30 min): {len(df2)}")
print(df2.head(5))



df2.to_csv("User_MAC000002_data.csv", index=False)
print("\nSaved to User_MAC000002_data.csv")



duplicaten = df2.duplicated(subset="DateTime", keep=False)
if duplicaten.any():
    print(f"\nDuplicaten gevonden: {duplicaten.sum()} rijen → gemiddelde genomen")
    df2 = df2.groupby("DateTime", as_index=False).mean(numeric_only=True)



df2 = df2.set_index("DateTime")



new_index_15 = pd.date_range(
    start=df2.index.min(),
    end=df2.index.max(),
    freq="15min"
)



df_15 = df2["KWH/hh (per half hour)"].reindex(df2.index.union(new_index_15))
df_15 = df_15.interpolate(method="time")
df_15 = df_15.reindex(new_index_15)



df_15 = df_15 / 2

df_15 = df_15.reset_index()
df_15.columns = ["DateTime", "KWH/hh (per kwartier)"]

print(f"\n=== 15-min data ===")
print(f"Rijen: {len(df_15)}")
print(df_15.head(8))
print(df_15.columns.tolist())



df_15.to_csv("User_MAC000002_15min.csv", index=False)
print("\nSaved to User_MAC000002_15min.csv")



gefabriceerd = df2.reset_index()[ontbrekende_mask]

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)



axes[0].plot(df2.reset_index()["DateTime"], df2.reset_index()["KWH/hh (per half hour)"],
             linewidth=0.8, color="steelblue", label="Gemeten (30 min)")
axes[0].scatter(gefabriceerd["DateTime"], gefabriceerd["KWH/hh (per half hour)"],
                color="red", s=20, zorder=5, label=f"Geïnterpoleerd ({len(gefabriceerd)})")
axes[0].set_title("Energieverbruik MAC000002 – 30 minuten")
axes[0].set_ylabel("KWH per half uur")
axes[0].legend()



axes[1].plot(df_15["DateTime"], df_15["KWH/hh (per kwartier)"],
             linewidth=0.6, color="darkorange", label="15 min (geïnterpoleerd)")
axes[1].set_title("Energieverbruik MAC000002 – 15 minuten")
axes[1].set_xlabel("Datum")
axes[1].set_ylabel("KWH per kwartier")
axes[1].legend()

plt.tight_layout()
plt.show()
