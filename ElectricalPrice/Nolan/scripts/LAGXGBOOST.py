import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

import joblib

#Load & sort 
df = pd.read_csv('./ReadyDataSets/Price_Weather_MergedExtraFeatures.csv')
df['DateUTC'] = pd.to_datetime(df['DateUTC'])
df = df.sort_values('DateUTC').reset_index(drop=True)

# outliers
df['Price_EUR_MWh'] = df['Price_EUR_MWh'].clip(-200, 300)

# Flag near zero
df['is_near_zero'] = df['Price_EUR_MWh'].between(-1, 1).astype(int)
print(f"Near-zero rows: {df['is_near_zero'].sum()} ({df['is_near_zero'].mean()*100:.1f}%)")

# Lag features look BACKWARDS only
INTERVALS_PER_DAY = 96  # 15-min data

def add_lag_features(df):
    df = df.copy()
    df['lag_1d']  = df['Price_EUR_MWh'].shift(INTERVALS_PER_DAY)
    df['lag_2d']  = df['Price_EUR_MWh'].shift(INTERVALS_PER_DAY * 2)
    df['lag_3d']  = df['Price_EUR_MWh'].shift(INTERVALS_PER_DAY * 3)
    df['lag_7d']  = df['Price_EUR_MWh'].shift(INTERVALS_PER_DAY * 7)

    # .shift(1) before rolling means we NEVER include the current row's price
    shifted = df['Price_EUR_MWh'].shift(1)
    df['roll_24h_mean']  = shifted.rolling(INTERVALS_PER_DAY).mean()
    df['roll_24h_std']   = shifted.rolling(INTERVALS_PER_DAY).std()
    df['roll_7d_mean']   = shifted.rolling(INTERVALS_PER_DAY * 7).mean()
    df['roll_7d_std']    = shifted.rolling(INTERVALS_PER_DAY * 7).std()
    df['price_momentum'] = shifted.rolling(16).mean() - shifted.rolling(96).mean()

    return df

df = add_lag_features(df)
df = df.dropna().reset_index(drop=True)

# Time features
def make_time_features(dates):
    dates = pd.to_datetime(dates)
    hour        = dates.dt.hour + dates.dt.minute / 60
    day_of_week = dates.dt.dayofweek
    month       = dates.dt.month
    day_of_year = dates.dt.dayofyear
    return pd.DataFrame({
        'hour_sin':   np.sin(2 * np.pi * hour / 24),
        'hour_cos':   np.cos(2 * np.pi * hour / 24),
        'dow_sin':    np.sin(2 * np.pi * day_of_week / 7),
        'dow_cos':    np.cos(2 * np.pi * day_of_week / 7),
        'month_sin':  np.sin(2 * np.pi * month / 12),
        'month_cos':  np.cos(2 * np.pi * month / 12),
        'doy_sin':    np.sin(2 * np.pi * day_of_year / 366),
        'doy_cos':    np.cos(2 * np.pi * day_of_year / 366),
        'is_weekend': (day_of_week >= 5).astype(int),
    })

X = make_time_features(df['DateUTC'])
X['temperature']       = df['temperature_2m'].values
X['sunshine_duration'] = df['sunshine_duration'].values
X['is_day']            = df['is_day'].values
X['is_near_zero']      = df['is_near_zero'].values
X['lag_1d']            = df['lag_1d'].values
X['lag_2d']            = df['lag_2d'].values
X['lag_3d']            = df['lag_3d'].values
X['lag_7d']            = df['lag_7d'].values
X['roll_24h_mean']     = df['roll_24h_mean'].values
X['roll_24h_std']      = df['roll_24h_std'].values
X['roll_7d_mean']      = df['roll_7d_mean'].values
X['roll_7d_std']       = df['roll_7d_std'].values
X['price_momentum']    = df['price_momentum'].values

y = df['Price_EUR_MWh'].values

# Train / Val / Test split 
n = len(df)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

X_train, y_train = X.iloc[:train_end],        y[:train_end]
X_val,   y_val   = X.iloc[train_end:val_end],  y[train_end:val_end]
X_test,  y_test  = X.iloc[val_end:],           y[val_end:]

dates_val  = df['DateUTC'].iloc[train_end:val_end]
dates_test = df['DateUTC'].iloc[val_end:]

print(f"\nTrain: {len(X_train):,} rows  ({df['DateUTC'].iloc[0].date()} → {df['DateUTC'].iloc[train_end-1].date()})")
print(f"Val:   {len(X_val):,} rows  ({df['DateUTC'].iloc[train_end].date()} → {df['DateUTC'].iloc[val_end-1].date()})")
print(f"Test:  {len(X_test):,} rows  ({df['DateUTC'].iloc[val_end].date()} → {df['DateUTC'].iloc[-1].date()})")


#does not improve preformance


'''

tscv = TimeSeriesSplit(n_splits=5)

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

grid_search.fit(X_train, y_train)

print(f"\nBest params: {grid_search.best_params_}")
print(f"Best CV MAE: {-grid_search.best_score_:.2f} EUR/MWh")

model = grid_search.best_estimator_




'''



#Train
model = XGBRegressor(
    n_estimators=2000,       
    learning_rate=0.02,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.1,
    random_state=42,
    n_jobs=-1,
    objective='reg:squarederror',
    eval_metric='mae',
    early_stopping_rounds=50,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=50
)



# Evaluate
y_val_pred  = model.predict(X_val)
y_test_pred = model.predict(X_test)

print(f"\nVal   — MAE: {mean_absolute_error(y_val, y_val_pred):.2f}  R²: {r2_score(y_val, y_val_pred):.4f}")
print(f"Test  — MAE: {mean_absolute_error(y_test, y_test_pred):.2f}  R²: {r2_score(y_test, y_test_pred):.4f}")


#Scatter: predicted vs actual
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test, y_test_pred, alpha=0.3, s=5, color='steelblue')
lims = [min(y_test.min(), y_test_pred.min()), max(y_test.max(), y_test_pred.max())]
ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect fit')
ax.text(0.05, 0.92,
        f"R²:  {r2_score(y_test, y_test_pred):.4f}\nMAE: {mean_absolute_error(y_test, y_test_pred):.2f}",
        transform=ax.transAxes, fontsize=11,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.set_xlabel('Actual (EUR/MWh)')
ax.set_ylabel('Predicted (EUR/MWh)')
ax.set_title('XGBoost — Predicted vs Actual (test set)')
ax.legend()
plt.tight_layout()
plt.show()

# Time series: predicted vs actual
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(dates_test.values, y_test,      label='Actual',    color='steelblue', linewidth=0.8)
ax.plot(dates_test.values, y_test_pred, label='Predicted', color='orange',    linewidth=0.8, alpha=0.8)
ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
ax.text(0.01, 0.97,
        f"MAE: {mean_absolute_error(y_test, y_test_pred):.2f}  |  R²: {r2_score(y_test, y_test_pred):.4f}",
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.set_title('XGBoost — Predicted vs Actual (test set)')
ax.set_xlabel('Date')
ax.set_ylabel('EUR/MWh')
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()   

#Feature importance
plt.figure(figsize=(8, 6))
plot_importance(model)
plt.tight_layout()
plt.show()


#Export model
'''
joblib.dump(value={
    'model': model,
    'feature_columns': X.columns.tolist()
}, filename='xgboost_price_model.pkl')

model.save_model('xgb_price_model.json')

'''