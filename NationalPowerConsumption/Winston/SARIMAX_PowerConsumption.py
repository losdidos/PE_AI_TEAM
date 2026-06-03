import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
#from fbprophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

def adf_test(timeseries):
    print("Results of Augmented Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag='AIC') # autolag helps choose the right number of lags

    # Format the output
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value

    print(dfoutput)

    # Quick Interpretation
    if dftest[1] <= 0.05:
        print("\n=> Conclusion: Reject Null Hypothesis (Data is Stationary)")
    else:
        print("\n=> Conclusion: Fail to Reject Null Hypothesis (Data is Non-Stationary)")

data = pd.read_csv(r'./ReadyDataSets/electricity_consumption_ready.csv', sep=",")
data['DateUTC'] = pd.to_datetime(data['DateUTC'])
data = data.set_index('DateUTC') #Set the index to DateUTC

data = data.resample('1h').mean()
#feature_cols = [col for col in data.columns if col not in ['DateUTC', 'Value']]
#X = data[feature_cols]
y = data['Value']

split = int(len(y) * 0.8)
#X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
print("checkpoint 1")

#print(data.head())
#data['Value_diff'] = data['Value'] - data['Value'].shift(1)
#adf_test(data['Value_diff'].dropna())

model = SARIMAX(y_train, order=(1,0,1), seasonal_order=(1,0,1,24)) #This is the seasonal Arima model, 672 = weekly cycle
print("checkpoint 2")
model_fit = model.fit(disp=False, maxiter=100) #Fit the model
print("checkpoint 2.1")
#freq = pd.Timedelta(minutes=15)
freq = pd.Timedelta(hours=1)
steps = 672
forecast_index = pd.date_range(start=y_train.index[-1] + freq, periods=steps, freq=freq)
forecast = model_fit.forecast(steps=steps) #Create a forecast based on how the model fit your data
forecast = pd.Series(forecast.values, index=forecast_index)
recent = y_train.iloc[-steps:] #Grab the most recent data (4 x 7 x 24 =  672) --> 672 entries = 7 days
print("checkpoint 3")

plt.figure(figsize=(10,5))

plt.plot(recent, label='Last 7 Days')
plt.plot(forecast, label='Forecast')

plt.legend()
plt.show()