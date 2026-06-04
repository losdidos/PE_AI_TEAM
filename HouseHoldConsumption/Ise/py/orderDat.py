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
df_15 = df_15 / 2  # energiebehoud

df_15 = df_15.reset_index()
df_15.columns = ["DateTime", "KWH/hh (per kwartier)"]

print(f"\n=== 15-min energie data ===")
print(f"Rijen: {len(df_15)}")
print(df_15.head(8))


weather = pd.read_csv(
    "weather_london.csv",
    names=["DateTime", "temperature_2m", "relative_humidity_2m", "snowfall", "rain"],
    header=0  # verwijder als er geen header is in het bestand
)



weather["DateTime"] = pd.to_datetime(weather["DateTime"], utc=True).dt.tz_localize(None)
weather = weather.sort_values("DateTime").reset_index(drop=True)

print(f"\n=== Weerdata geladen ===")
print(f"Rijen: {len(weather)}")
print(f"Bereik: {weather['DateTime'].min()} → {weather['DateTime'].max()}")
print(weather.head(5))


weather = weather.set_index("DateTime")

weer_kolommen = ["temperature_2m", "relative_humidity_2m", "snowfall", "rain"]



new_index_weather = pd.date_range(
    start=weather.index.min(),
    end=weather.index.max(),
    freq="15min"
)

weather_15 = weather.reindex(weather.index.union(new_index_weather))
weather_15 = weather_15.interpolate(method="time")
weather_15 = weather_15.reindex(new_index_weather)
weather_15 = weather_15.reset_index()
weather_15.rename(columns={"index": "DateTime"}, inplace=True)

print(f"\n=== Weerdata na 15-min interpolatie ===")
print(f"Rijen: {len(weather_15)}")
print(weather_15.head(8))



df_merged = pd.merge(df_15, weather_15, on="DateTime", how="inner")

print(f"\n=== Samengevoegde dataset ===")
print(f"Rijen: {len(df_merged)}")
print(df_merged.head(8))
print(f"\nKolommen: {df_merged.columns.tolist()}")
print(f"\nNaN per kolom:\n{df_merged.isnull().sum()}")



df_merged.to_csv("User_MAC000002_15min_met_weer.csv", index=False)
print("\nSaved to User_MAC000002_15min_met_weer.csv")



fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)



axes[0].plot(df_merged["DateTime"], df_merged["KWH/hh (per kwartier)"],
             linewidth=0.6, color="darkorange", label="Energie (15 min)")
axes[0].set_title("Energieverbruik MAC000002 – 15 minuten")
axes[0].set_ylabel("KWH per kwartier")
axes[0].legend()



axes[1].plot(df_merged["DateTime"], df_merged["temperature_2m"],
             linewidth=0.8, color="crimson", label="Temperatuur (°C)")
axes[1].set_title("Temperatuur – 15 minuten")
axes[1].set_ylabel("°C")
axes[1].legend()



axes[2].plot(df_merged["DateTime"], df_merged["relative_humidity_2m"],
             linewidth=0.8, color="steelblue", label="Luchtvochtigheid (%)")
ax2_twin = axes[2].twinx()
ax2_twin.bar(df_merged["DateTime"], df_merged["rain"],
             width=0.01, color="navy", alpha=0.4, label="Regen (mm)")
axes[2].set_title("Luchtvochtigheid & Neerslag – 15 minuten")
axes[2].set_xlabel("Datum")
axes[2].set_ylabel("Vochtigheid (%)")
ax2_twin.set_ylabel("Regen (mm)")
axes[2].legend(loc="upper left")
ax2_twin.legend(loc="upper right")

plt.tight_layout()
plt.show()