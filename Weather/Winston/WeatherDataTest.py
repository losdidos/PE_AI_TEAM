import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import warnings
import openmeteo_requests
import requests_cache
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import DecomposeResult
from statsmodels.graphics.tsaplots import plot_acf
from retry_requests import retry
warnings.filterwarnings('ignore')

#----------------------------------------------------------------------------------------
#								Connect to Open-Meto API
#----------------------------------------------------------------------------------------

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
	"latitude": 51.0257,
	"longitude": 4.4776,
	"start_date": "2024-01-01",
	"end_date": "2025-09-30",
	"daily": ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min", "daylight_duration"],
	"hourly": ["temperature_2m", "direct_radiation", "sunshine_duration", "is_day", "temperature_2m_spread"],
	"timezone": "GMT",
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation: {response.Elevation()} m asl")
print(f"Timezone: {response.Timezone()}{response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
#hourly_direct_radiation = hourly.Variables(1).ValuesAsNumpy()
#hourly_sunshine_duration = hourly.Variables(2).ValuesAsNumpy() not using these columns
#hourly_is_day = hourly.Variables(3).ValuesAsNumpy()

hourly_data = {"date": pd.date_range(
	#start = pd.to_datetime(hourly.Time() + response.UtcOffsetSeconds(), unit = "s", utc = True),
	#end =  pd.to_datetime(hourly.TimeEnd() + response.UtcOffsetSeconds(), unit = "s", utc = True),
	start = pd.to_datetime(hourly.Time() + response.UtcOffsetSeconds(), unit = "s"),
	end =  pd.to_datetime(hourly.TimeEnd() + response.UtcOffsetSeconds(), unit = "s"),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}

hourly_data["temperature_2m"] = hourly_temperature_2m
#hourly_data["direct_radiation"] = hourly_direct_radiation
#hourly_data["sunshine_duration"] = hourly_sunshine_duration      not using these columns
#hourly_data["is_day"] = hourly_is_day

hourly_dataframe = pd.DataFrame(data = hourly_data)
#print("\nHourly data\n", hourly_dataframe)


#----------------------------------------------------------------------------------------
#							Resample Data -> 15 mins intervals + save to CSV
#----------------------------------------------------------------------------------------

# Make sure 'date' is datetime
hourly_dataframe['date'] = pd.to_datetime(hourly_dataframe['date'])
hourly_dataframe = hourly_dataframe.set_index('date')
quarter_dataframe = hourly_dataframe.resample('15min').interpolate() #resample to 15 min interval and interpolate
#quarter_dataframe.to_csv('quarter_dataframe_2.csv') #.to_csv used to create csv file
print(quarter_dataframe)
print("CSV file successfully created!")


#----------------------------------------------------------------------------------------
#										ADF test
#----------------------------------------------------------------------------------------

#Declaration of ADF test
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

adf_test(hourly_dataframe['temperature_2m'])



#----------------------------------------------------------------------------------------
#										ACF & PACF test
#----------------------------------------------------------------------------------------

#pd.plotting.autocorrelation_plot(hourly_dataframe["temperature_2m"])
#plt.show()

plot_acf(hourly_dataframe["temperature_2m"], lags=40)
plt.xlabel("Lag (hours)", fontsize=20)
plt.ylabel("Autocorrelation", fontsize=20)
plt.title("Autocorrelation of Hourly Temperature", fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.show()


#----------------------------------------------------------------------------------------
#								Stationarity clarification
#----------------------------------------------------------------------------------------
adf_test(hourly_dataframe['temperature_2m'])
six_hourly = hourly_dataframe.resample('6h').mean() #Resample naar 6 uur en gebruik de mean om het op te vullen
six_hourly['temperature_2m'].plot(title="Temperature Every 6 Hours")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.show() #Show the temperature every 6 hours in a plot

decomposition = seasonal_decompose(hourly_dataframe["temperature_2m"], model='additive', two_sided=True, extrapolate_trend=0) #Decompose the hourly dataframe into: 1. trend, 2. seasonal, 3. residual/noise

plt.figure(figsize=(14,6))
plt.plot(hourly_dataframe.index, hourly_dataframe["temperature_2m"], label="Observed", color='blue') #Plot the hourly_dataframe
plt.plot(decomposition.trend.index, decomposition.trend, color='red', label="Trend (shows non-stationarity)", linewidth=2) #Plot the trend, obtained from seasonal decompose
plt.title("Observed Temperature vs Trend (Non-Stationary Series)", fontsize=20)
plt.xlabel("Date", fontsize=20)
plt.ylabel("Temperature (°C)", fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.legend()
plt.show() #Show the temperature every hour + overlay the trend on top to prove non-stationarity