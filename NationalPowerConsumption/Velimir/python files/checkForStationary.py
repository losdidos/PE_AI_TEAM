import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


df = pd.read_csv("./merged1.csv", sep=";") # omdat file tab seperated is 
df.columns = df.columns.str.strip() # normalizeren van kolomnamen door spaties te verwijderen

df["DateUTC"] = pd.to_datetime(df["DateUTC"], format="%Y-%m-%d %H:%M:%S", errors="coerce") # beste voor messy data errors

df = df.set_index("DateUTC")
df = df.sort_index()


print(df.head(20))
print(df.shape)
print(df.columns.tolist())

#adf test (stationary of niet)

load_values = df["Value"].dropna() # selecteer alleen value en NaN weg


adf_result = adfuller(load_values, autolag="AIC") #runs ADF test en returnt een tuple 

print(f"ADF Statistic: {adf_result[0]:.6f}") # print de ADF statistic .6f voor decimal met 6 nummer ma comma
print(f"P-value: {adf_result[1]:.6f}") # print de p value


if adf_result[1] < 0.05:
    print("data is stationary (p < 0.05)")
else:
    print("data is not stationary (p >= 0.05)")



#kpss test (stationary of niet)
# de test is verward met seasonality dus zegt not stationary


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

