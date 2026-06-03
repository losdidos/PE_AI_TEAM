import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


df_2024 = pd.read_csv("dataPower\\monthly_hourly_load_values_2024.csv", sep="\t") # omdat file tab seperated is al geinterpolate door code ise
df_2025 = pd.read_csv("dataPower\\monthly_hourly_load_values_2025.csv", sep="\t") # omdat file tab seperated is al geinterploate door code ise
df = pd.concat([df_2024, df_2025], ignore_index=True)

df_temp = pd.read_csv("dataPower\\quarter_dataframe_2.csv")
df_temp.columns = df_temp.columns.str.strip()
df_temp["date"] = pd.to_datetime(df_temp["date"], errors="coerce")
df_temp = df_temp.rename(columns={"date": "DateUTC"})

df.columns = df.columns.str.strip() # normalizeren van kolomnamen door spaties te verwijderen
df["CountryCode"] = df["CountryCode"].astype(str).str.strip().str.upper()

df["DateUTC"] = pd.to_datetime(df["DateUTC"], format="%d-%m-%Y %H:%M", errors="coerce") # beste voor messy data errors



df = df[df["CountryCode"] == "BE"].copy()
df = df.set_index("DateUTC")

df = df.drop(columns=["CreateDate", "UpdateDate" , "MeasureItem" , "CountryCode" , "Cov_ratio" , "Value_ScaleTo100"], errors="ignore") # error negeren en laat de original value hoe hij was

print(df.info())
df = df.sort_index()




print("First index:", df.index.min())
print("Last index:", df.index.max())


value_hourly = df["Value"].groupby(df.index).mean()
series_15min = value_hourly.resample("15min").interpolate(method="linear")






#adf test (stationary of niet)

load_values = series_15min.dropna() # selecteer alleen value en NaN weg


adf_result = adfuller(load_values, autolag="AIC") #runs ADF test en returnt een tuple 
print(f"ADF Statistic: {adf_result[0]:.6f}") # print de ADF statistic .6f voor decimal met 6 nummer ma comma
print(f"P-value: {adf_result[1]:.6f}") # print de p value


if adf_result[1] < 0.05:
    print("data is stationary (p < 0.05)")
else:
    print("data is not stationary (p >= 0.05)")



#kpss test (stationary of niet)
# de test is verward met seasonality dus zegt not stationary

13

kpss_result = kpss(load_values, regression="c", nlags="auto")

print(f"KPSS Statistic: {kpss_result[0]:.6f}")
print(f"P-value: {kpss_result[1]:.6f}")




#seasonal differencing

# Seasonal differencing (remove daily pattern) 
# diff.24 neemt het verschil tussen 24 uur dus nu vergelijkt hij de waarde op het zelfde uur en laat het dagelijks patroon er uit
load_values_diff = load_values.diff(24).dropna()  

# Now test the differenced data
adf_result_diff = adfuller(load_values_diff, autolag="AIC")
print("ADF on differenced data:", adf_result_diff[1])

kpss_result_diff = kpss(load_values_diff, regression="c", nlags="auto")
print("KPSS on differenced data:", kpss_result_diff[1])

# error van out of range dus data is 100% stationary




fig, axes = plt.subplots(2, 1, figsize=(12, 8)) # define de grote van plot 1 en 2

# ACF plotting
plot_acf(load_values_diff, lags=40, ax=axes[0])

#namen
axes[0].set_title("Autocorrelation Function (ACF)")
axes[0].set_xlabel("Lag")
axes[0].set_ylabel("ACF")

# PACF plotting
plot_pacf(load_values_diff, lags=40, ax=axes[1])

#namen
axes[1].set_title("Partial Autocorrelation Function (PACF)")
axes[1].set_xlabel("Lag")
axes[1].set_ylabel("PACF")






plt.tight_layout()
plt.show()



print(series_15min.head(3))
print(series_15min.tail(3))

series_15min_df = series_15min.rename("Value").reset_index()
series_15min_df = series_15min_df.rename(columns={"index": "DateUTC"})

merge = pd.merge(series_15min_df, df_temp, on="DateUTC", how="inner") # inner merge op DateUTC kolom , alleen de rijen die in beide dataframes voorkomen worden behouden

#merge.to_csv('Power_Production_Weather.csv', index=False) #.to_csv used to create csv file

print(merge.head(10))


print(merge.info())
print(merge.isnull().sum())
print(merge.describe())






