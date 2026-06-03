import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

df = pd.read_csv(r"./ReadyDataSets/electricity_consumption_ready.csv")
df['DateUTC'] = pd.to_datetime(df['DateUTC'])

df['hour'] = df['DateUTC'].dt.hour #Get hour and make column for it
df['dayofweek'] = df['DateUTC'].dt.dayofweek #Get day of the week and add column for it
df['month'] = df['DateUTC'].dt.month #Ditto
df['lag1'] = df['Value'].shift(1) #Create a column with 15 minute lag
#df['lag24'] = df['Value'].shift(24) #6 hour lag, but currently not used in code

df = df.dropna().reset_index(drop=True) #After shifting, we might have NaN entries, so we have to drop them + drop the index

feature_cols = ['hour', 'dayofweek', 'month', 'lag1'] #Choose the feature columns // Dont forget to add lag24 if needed
X = df[feature_cols]
y = df['Value'] 
dates = df['DateUTC']

freq_minutes = 15
steps_per_day = 24 * 60 // freq_minutes #96 datapoints in a full day
n_days = 7
n_future_steps = n_days * steps_per_day

last_date = df['DateUTC'].iloc[-1]
future_dates = pd.date_range(start= last_date + pd.Timedelta(minutes= freq_minutes), periods= n_future_steps, freq= f'{freq_minutes}min') #Create correct dates for the next 7 days with 96 datapoints per day

future_df = pd.DataFrame({'DateUTC': future_dates}) #Insert the dates into the a dataframe
future_df['hour'] = future_df['DateUTC'].dt.hour
future_df['dayofweek'] = future_df['DateUTC'].dt.dayofweek
future_df['month'] = future_df['DateUTC'].dt.month

future_df['lag1'] = df['Value'].iloc[-1] #Set lag to the last known value

split = int(len(y) * 0.8) #Index of the last index in 80% of observations
X_train, X_test = X[:split], X[split:] #Pick X_train as first 80% and X_test as last 20%
y_train, y_test = y[:split], y[split:] #Pick y_train as first 80% and y_test as last 20%
dates_train, dates_test = dates[:split], dates[split:] #Pick dates_train as first 80% and dates_test as last 20%

regressor = RandomForestRegressor(n_estimators=200, random_state=42, oob_score=True) #Create a forest with 200 trees + use random seed 42 + allow the model to estimate performance (Out-of-bag score)
regressor.fit(X_train, y_train) #Fit the model

print(f'OOB Score: {regressor.oob_score_:.4f}') #Print out of bag score (R^2 estimate)

X_test_recursive = X_test.copy()
y_pred_recursive = []
future_predictions = []

#Forecast on test set
for i in range(len(X_test_recursive)): #For each row
    pred = regressor.predict(X_test_recursive.iloc[i].values.reshape(1, -1))[0] #Predict energy consumption for current row + format to 2D array
    y_pred_recursive.append(pred) #Save prediction
    if i + 1 < len(X_test_recursive):
        X_test_recursive.iloc[i + 1, X_test_recursive.columns.get_loc('lag1')] = pred #Store the prediction in the dataframe, but store it in the next row of the lag1 column

y_pred_recursive = np.array(y_pred_recursive) #Convert to numpy array to gain access to evaluation metrics

#Get evaluation metrics
mae_test = mean_absolute_error(y_test, y_pred_recursive)
r2_test = r2_score(y_test, y_pred_recursive)

print(f'Test MAE (recursive forecast): {mae_test:.4f}')
print(f'Test R² (recursive forecast): {r2_test:.4f}')

#Predict future
for i in range(len(future_df)):
    pred = regressor.predict(future_df.iloc[i][feature_cols].values.reshape(1, -1))[0] #feature_cols is passed here to match the desired output (the new dataframe doesnt know the lags yet)
    future_predictions.append(pred)
    if i + 1 < len(future_df):
        future_df.iloc[i + 1, future_df.columns.get_loc('lag1')] = pred #Update the lag for the next iteration

future_df['predicted_Value'] = future_predictions #Save predictions

plt.figure(figsize=(15,6))
plt.plot(df['DateUTC'], df['Value'], label='Historical', color='blue')
plt.plot(dates_test, y_pred_recursive, label='Test Predictions', color='green')
plt.plot(future_df['DateUTC'], future_df['predicted_Value'], label='Future Forecast', color='red')
plt.title(f"Random Forest Forecast with Test Predictions", fontsize=20)
plt.xlabel("Date", fontsize=20)
plt.ylabel("Electricity Consumption (MW)", fontsize=20)
plt.legend(fontsize=20)
plt.xticks(rotation=45)
plt.tick_params(axis='both', labelsize=20)

plt.tight_layout()
plt.show()

min_val = min(min(y_test), min(y_pred_recursive))
max_val = max(max(y_test), max(y_pred_recursive))

plt.scatter(y_test, y_pred_recursive, color='blue', alpha=0.2, s=10)

plt.plot([min_val, max_val],
         [min_val, max_val],
         color="red")

plt.text(
    0.05, 0.95,
    f"$R^2$ = {r2_test:.3f}\nMAE = {mae_test:.3f}",
    transform=plt.gca().transAxes,
    fontsize=20,
    verticalalignment='top',
    bbox=dict(boxstyle="round", alpha=0.2)
)


plt.xlabel("Actual (MW)", fontsize=20)
plt.ylabel("Predicted (MW)", fontsize=20)
plt.title("Predicted vs Actual Power Consumption", fontsize=20)
plt.tick_params(axis='both', labelsize=20)

plt.savefig("RandomForest_PowerConsumption_ParityPlot.png")
plt.show()