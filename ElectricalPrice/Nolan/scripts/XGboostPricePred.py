import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score 
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import plot_tree, plot_importance



def make_time_features(dates):
    dates = pd.to_datetime(dates)
    hour = dates.dt.hour + dates.dt.minute / 60
    day_of_week = dates.dt.dayofweek
    month = dates.dt.month  
    day_of_year = dates.dt.dayofyear

    return pd.DataFrame({
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




df = pd.read_csv(r'C:\Users\nolan\OneDrive\Documents\GitHub\PE2-AI\ReadyDataSets\Price_Weather_MergedExtraFeatures.csv')
df['DateUTC'] = pd.to_datetime(df['DateUTC'])
df = df.sort_values('DateUTC').reset_index(drop=True)

print(f"Loaded {len(df)} rows of data")

# zorgen dat het model de date time begrijpt , dus omzetten naar cos en sin zodat deze punten vergeleken kunnen worden en het model de afstand tusssen de uuren kan begrijpen


dates = df['DateUTC']
X = make_time_features(dates)
X['temperature'] = df['temperature_2m'].values
X['sunshine_duration'] = df['sunshine_duration'].values
X['is_day'] = df['is_day'].values

price = df['Price_EUR_MWh'].values

for price in df['Price_EUR_MWh']:
    if price < -200:
        df.loc[df['Price_EUR_MWh'] == price, 'Price_EUR_MWh'] = -200
    if price > 300:
        df.loc[df['Price_EUR_MWh'] == price, 'Price_EUR_MWh'] = 300


y = df['Price_EUR_MWh'].values

print(f"Created {X.shape[1]} features")



# split data set in train en test sets , 20 - 80

split = int(len(df) * 0.8)
Xtrain, Xtest = X.iloc[:split], X.iloc[split:]

ytrain, ytest = y[:split], y[split:]

testdates = df['DateUTC'].iloc[split:]

tscv = TimeSeriesSplit(n_splits=5)
'''
param_grid = {
    'n_estimators':     [200, 300, 500],
    'learning_rate':    [0.01, 0.05, 0.1],
    'max_depth':        [3, 5, 7],
    'subsample':        [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
}

grid_search = GridSearchCV(
    estimator  = XGBRegressor(random_state=42, n_jobs=-1, objective='reg:squarederror'),
    param_grid = param_grid,
    cv         = tscv,
    scoring    = 'neg_mean_absolute_error',
    verbose    = 2,        # shows progress
    n_jobs     = -1
)

grid_search.fit(Xtrain, ytrain)

print(f"\nBest params: {grid_search.best_params_}")
print(f"Best CV MAE: {-grid_search.best_score_:.2f} EUR/MWh")

model = grid_search.best_estimator_
'''
# Model training , values aan de hand van claude
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    objective='reg:squarederror',
    eval_metric='mae',  
)

model.fit(
    Xtrain, ytrain,
    eval_set=[(Xtrain, ytrain), (Xtest, ytest)],
    verbose=False
)





# vergelijk de voorspellingen met test data / bereken MAE en R2



ypred = model.predict(Xtest)
mae = mean_absolute_error(ytest, ypred)
r2 = r2_score(ytest, ypred)

print(f"\nResults on test data:")
print(f"  MAE: {mae}")
print(f"  R²: {r2}")
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(testdates.values, ytest, label='Actual',    linewidth=0.8, color='steelblue')
ax.plot(testdates.values, ypred, label='Predicted', linewidth=0.8, color='orange', alpha=0.8)
ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
ax.set_title('Price Forecast vs Actual (test set)')
ax.set_xlabel('Date')
ax.set_ylabel('EUR/MWh')
ax.legend()
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(ypred, ytest, alpha=0.3, s=5, color='steelblue')
ax.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], 
        color='red', linewidth=1, linestyle='--', label='Perfect fit')
ax.set_xlabel('Predicted (EUR/MWh)')
ax.set_ylabel('Actual (EUR/MWh)')
ax.set_title('Actual vs Predicted')
ax.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plot_importance(model)
plt.tight_layout()
plt.show()


"""
#vragen voor voorspelling te hebben
temp = float(input('Expected temperature:').strip())

start_date_str = input('Start date:').strip()
start_date = pd.to_datetime(start_date_str)

periods = 2 * 24 * 4  # 2 dagen
future_dates = pd.date_range(start=start_date, periods=periods, freq='15min')
Xfuture = make_time_features(pd.Series(future_dates))
Xfuture['temperature'] = temp  
yfuture = model.predict(Xfuture)

forecast = pd.DataFrame({
    'DateUTC': future_dates,
    'PredictedValue': yfuture,
})

print('\nForecast:')
print(forecast.head(72).to_string(index=False))
"""