
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX


data = pd.read_csv("Power_Production_Weather.csv")


data["DateUTC"] = pd.to_datetime(data["DateUTC"])


data = data.sort_values("DateUTC")


data.set_index("DateUTC", inplace=True)


data["Value"] = pd.to_numeric(data["Value"], errors="coerce")


data = data.dropna()




arima_model = SARIMAX(
    data["Value"],
    order=(1,1,1),   # (p,d,q)
    seasonal_order=(0,0,0,0)
).fit()

print("\nARIMA AIC:", arima_model.aic)


sarima_model = SARIMAX(
    data["Value"],
    order=(1,1,1),
    seasonal_order=(1,1,1,24)
).fit()

print("SARIMA AIC:", sarima_model.aic)




if arima_model.aic < sarima_model.aic:# Vergelijk AIC
    print("\nARIMA is beter (lagere AIC)")
else:
    print("\nSARIMA is beter (lagere AIC)")