import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


df = pd.read_csv("monthly_hourly_load_values_2024.csv", sep="\t")

df["DateUTC"] = pd.to_datetime(df["DateUTC"], format="%d-%m-%Y %H:%M")

df = df[df["CountryCode"] == "BE"].copy()

df = df.drop(columns=["CreateDate", "UpdateDate"], errors="ignore")

df = df.set_index("DateUTC")
df = df.sort_index()


print(df.head(20))
print(df.shape)
print(df.columns.tolist())



load_values = df["Value"].dropna()


adf_result = adfuller(load_values, autolag="AIC") # ADF test en returnt een tuple

print(f"ADF Statistic: {adf_result[0]:.6f}")
print(f"P-value: {adf_result[1]:.6f}")


if adf_result[1] < 0.05:
    print("data is stationary (p < 0.05)")
else:
    print("data is not stationary (p >= 0.05)")






kpss_result = kpss(load_values, regression="c", nlags="auto")

print(f"KPSS Statistic: {kpss_result[0]:.6f}")
print(f"P-value: {kpss_result[1]:.6f}")




load_values_diff = load_values.diff(24).dropna()  

adf_result_diff = adfuller(load_values_diff, autolag="AIC")
print("ADF on differenced data:", adf_result_diff[1])

kpss_result_diff = kpss(load_values_diff, regression="c", nlags="auto")
print("KPSS on differenced data:", kpss_result_diff[1])





fig, axes = plt.subplots(2, 1, figsize=(12, 8))

plot_acf(load_values_diff, lags=40, ax=axes[0])


axes[0].set_title("Autocorrelation Function (ACF)")
axes[0].set_xlabel("Lag")
axes[0].set_ylabel("ACF")


plot_pacf(load_values_diff, lags=40, ax=axes[1])


axes[1].set_title("Partial Autocorrelation Function (PACF)")
axes[1].set_xlabel("Lag")
axes[1].set_ylabel("PACF")

plt.tight_layout()
plt.show()