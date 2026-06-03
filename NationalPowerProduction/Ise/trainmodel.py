import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

PARAMS = {
    'n_estimators':    500,
    'learning_rate':   0.05,
    'max_depth':       3,
    'subsample':       0.8,
    'colsample_bytree':0.8,
    'random_state':    2,#98 = 1.6215/ 6 = 1.537/2 = 1.4246/ 1 = 1.900 / 0 = 1.894
    'min_child_weight':10,
}

df = pd.read_csv('Power_Production_Weather.csv')
df['DateUTC'] = pd.to_datetime(df['DateUTC'])
df = df.sort_values('DateUTC').reset_index(drop=True)

print(f"Loaded {len(df)} rows of data")

# zorgen dat het model de date time begrijpt , dus omzetten naar cos en sin zodat deze punten vergeleken kunnen worden en het model de afstand tusssen de uuren kan begrijpen


dates = df['DateUTC']
hour = dates.dt.hour + dates.dt.minute / 60
day_of_week = dates.dt.dayofweek
month = dates.dt.month
day_of_year = dates.dt.dayofyear

X = pd.DataFrame({
    'hour_sin': np.sin(2 * np.pi * hour / 24),
    'hour_cos': np.cos(2 * np.pi * hour / 24),
    'dow_sin': np.sin(2 * np.pi * day_of_week / 7),
    'dow_cos': np.cos(2 * np.pi * day_of_week / 7),
    'month_sin': np.sin(2 * np.pi * month / 12),
    'month_cos': np.cos(2 * np.pi * month / 12),
    'doy_sin': np.sin(2 * np.pi * day_of_year / 366),
    'doy_cos': np.cos(2 * np.pi * day_of_year / 366),
    'is_weekend': (day_of_week >= 5).astype(int),
})

y = df['Value'].values

print(f"Created {X.shape[1]} features")




# split data set in train en test sets , 20 - 80

split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]

y_train, y_test = y[:split], y[split:]

test_dates = df['DateUTC'].iloc[split:]







model = XGBRegressor(
    n_estimators=350,
    learning_rate=0.1,
    max_depth=20,
    subsample=0.6,
    colsample_bytree=0.8,
    random_state=2,
    n_jobs=-1,
)
model.fit(X_train, y_train)


# vergelijk de voorspellingen met test data / bereken MAE en R


y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nResults on test data:")
print(f"  MAE: {mae}")
print(f"  R²: {r2}")

#plotting the test set vs predicted values


plt.figure(figsize=(14, 5))
plt.plot(test_dates.values, y_test, label='Actual', linewidth=2)
plt.plot(test_dates.values, y_pred, label='XGBoost Forecast', linestyle='--', linewidth=1.5)
plt.title('Power Production Forecast (ALL Test Data)')
plt.xlabel('Date')
plt.ylabel('Power (kW)')
plt.legend()
plt.tight_layout()

plt.show()

print("\nDone!")






