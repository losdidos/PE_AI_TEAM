import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
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




df = pd.read_csv('./ReadyDataSets/Power_Production_Weather.csv')
df['DateUTC'] = pd.to_datetime(df['DateUTC'])
df = df.sort_values('DateUTC').reset_index(drop=True)

print(f"Loaded {len(df)} rows of data")

# zorgen dat het model de date time begrijpt , dus omzetten naar cos en sin zodat deze punten vergeleken kunnen worden en het model de afstand tusssen de uuren kan begrijpen


dates = df['DateUTC']
X = make_time_features(dates)
X['temperature'] = df['temperature_2m'].values


y = df['Value'].values

print(f"Created {X.shape[1]} features")



# split data set in train en test sets , 20 - 80

split = int(len(df) * 0.8)
Xtrain, Xtest = X.iloc[:split], X.iloc[split:]

ytrain, ytest = y[:split], y[split:]

testdates = df['DateUTC'].iloc[split:]



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


# QQ / actual vs predicted scatter
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(ytest, ypred, alpha=0.3, s=5, color='steelblue')
lims = [min(ytest.min(), ypred.min()), max(ytest.max(), ypred.max())]
ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect fit')
ax.text(0.05, 0.92,
        f"R²:  {r2:.4f}\nMAE: {mae:.2f}",
        transform=ax.transAxes, fontsize=11,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title('Predicted vs Actual (test set)')
ax.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plot_importance(model)
plt.tight_layout()
plt.show()

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