import pandas as pd
import statsmodels.api as sm


data = pd.read_csv("Power_Production_Weather.csv")


data["Value"] = pd.to_numeric(data["Value"], errors="coerce")
data["temperature_2m"] = pd.to_numeric(data["temperature_2m"], errors="coerce")


data = data.dropna()


y = data["Value"]
X = data["temperature_2m"]


X = sm.add_constant(X)


model = sm.OLS(y, X).fit()


print("AIC:", model.aic)


print(model.summary())