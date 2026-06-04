import openmeteo_requests

import pandas as pd
import requests_cache
from retry_requests import retry



cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)



url = "https://archive-api.open-meteo.com/v1/archive"
params = {
	"latitude": 51.5085,
	"longitude": -0.1257,
	"start_date": "2012-10-12",
	"end_date": "2014-02-28",
	"hourly": ["temperature_2m", "relative_humidity_2m", "snowfall", "rain"],
}
responses = openmeteo.weather_api(url, params = params)



response = responses[0]
print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation: {response.Elevation()} m asl")
print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")



hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
hourly_snowfall = hourly.Variables(2).ValuesAsNumpy()
hourly_rain = hourly.Variables(3).ValuesAsNumpy()

hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	end =  pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}

hourly_data["temperature_2m"] = hourly_temperature_2m
hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
hourly_data["snowfall"] = hourly_snowfall
hourly_data["rain"] = hourly_rain

hourly_dataframe = pd.DataFrame(data = hourly_data)
print("\nHourly data\n", hourly_dataframe)


hourly_dataframe.to_csv(r"C:\Users\NemoL\Documents\ThomasMore\TweedeJaar\troimester2_2\Practis enterpris\Data\sprint2\datfile reader\weather_london.csv", index=False)

