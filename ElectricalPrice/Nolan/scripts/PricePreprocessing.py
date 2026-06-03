import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf





df = pd.read_csv(r"C:\Users\nolan\OneDrive\Documents\GitHub\PE2-AI\PriceMWHdata\prices_BE_2024-01-01_2025-12-31_15min.csv") 
df_temp = pd.read_csv(r"C:\Users\nolan\OneDrive\Documents\GitHub\PE2-AI\ReadyDataSets\quarter_dataframe.csv")

df_temp = df_temp.rename(columns={"date": "DateUTC"}) 
df_temp['DateUTC'] = pd.to_datetime(df_temp['DateUTC'], utc=True)


price = df['Price_EUR_MWh']



for price in df['Price_EUR_MWh']:
    if price < 200:
        df.loc[df['Price_EUR_MWh'] == price, 'Price_EUR_MWh'] = 200
    if price > 300:
        df.loc[df['Price_EUR_MWh'] == price, 'Price_EUR_MWh'] = 300
        


print(df.info())
print(df.describe())

df['DateUTC'] = pd.to_datetime(df['DateUTC'], utc=True)

plt.figure(figsize=(14, 5))
plt.plot(df['DateUTC'], df['Price_EUR_MWh'])
plt.ylabel('Price (EUR/MWh)')
plt.show()




print("\n=== Gap Detection ===")
expected_freq = pd.Timedelta('1h')  # adjust if not hourly
df['time_diff'] = df['DateUTC'].diff()
gaps = df[df['time_diff'] > expected_freq][['DateUTC', 'time_diff']]

if gaps.empty:
    print("No gaps found — data is continuous.")
else:
    print(f"{len(gaps)} gap(s) found:")
    print(gaps.to_string())





fig, axes = plt.subplots(2, 1, figsize=(14, 8))

plot_acf(price, lags=169, ax=axes[0], alpha=0.05)
axes[0].set_title('ACF — up to 169 lags (daily + weekly seasonality)')
axes[0].axvline(x=24,  color='red',    linestyle='--', linewidth=0.8, label='24h (daily)')
axes[0].axvline(x=168, color='orange', linestyle='--', linewidth=0.8, label='168h (weekly)')
axes[0].legend()

plot_pacf(price, lags=169, ax=axes[1], alpha=0.05, method='ywm')
axes[1].set_title('PACF — up to 169 lags')
axes[1].axvline(x=24,  color='red',    linestyle='--', linewidth=0.8, label='24h (daily)')
axes[1].axvline(x=168, color='orange', linestyle='--', linewidth=0.8, label='168h (weekly)')
axes[1].legend()

plt.tight_layout()
plt.show()





merge = pd.merge(df, df_temp, on="DateUTC", how="inner") # inner merge op DateUTC kolom , alleen de rijen die in beide dataframes voorkomen worden behouden

print(merge.head(10))



#merge.to_csv('Price_Weather_MergedExtraFeatures.csv', index=False)


