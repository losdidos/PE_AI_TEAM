import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib

# ── Load & sort ──────────────────────────────────────────────────────────────
df = pd.read_csv(r'C:\Users\renzo\PyCharmMiscProject\.venv\pracitce enterprise\cost_data\Price_Weather_MergedExtraFeatures.csv')
df['DateUTC'] = pd.to_datetime(df['DateUTC'])
df = df.sort_values('DateUTC').reset_index(drop=True)

df['Price_EUR_MWh'] = df['Price_EUR_MWh'].clip(-200, 300)
df['is_near_zero']  = df['Price_EUR_MWh'].between(-1, 1).astype(int)

# ── Lag features ─────────────────────────────────────────────────────────────
INTERVALS_PER_DAY = 96

def add_lag_features(df):
    df = df.copy()
    df['lag_1d']  = df['Price_EUR_MWh'].shift(INTERVALS_PER_DAY)
    df['lag_2d']  = df['Price_EUR_MWh'].shift(INTERVALS_PER_DAY * 2)
    df['lag_3d']  = df['Price_EUR_MWh'].shift(INTERVALS_PER_DAY * 3)
    df['lag_7d']  = df['Price_EUR_MWh'].shift(INTERVALS_PER_DAY * 7)
    shifted = df['Price_EUR_MWh'].shift(1)
    df['roll_24h_mean']  = shifted.rolling(INTERVALS_PER_DAY).mean()
    df['roll_24h_std']   = shifted.rolling(INTERVALS_PER_DAY).std()
    df['roll_7d_mean']   = shifted.rolling(INTERVALS_PER_DAY * 7).mean()
    df['roll_7d_std']    = shifted.rolling(INTERVALS_PER_DAY * 7).std()
    df['price_momentum'] = shifted.rolling(16).mean() - shifted.rolling(96).mean()
    return df

df = add_lag_features(df)
df = df.dropna().reset_index(drop=True)

# ── Time features ─────────────────────────────────────────────────────────────
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

# ── Split ─────────────────────────────────────────────────────────────────────
n         = len(df)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

X_train, y_train = X.iloc[:train_end],       y[:train_end]
X_val,   y_val   = X.iloc[train_end:val_end], y[train_end:val_end]
X_test,  y_test  = X.iloc[val_end:],          y[val_end:]
dates_test       = df['DateUTC'].iloc[val_end:]

# ── Train ─────────────────────────────────────────────────────────────────────
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

y_test_pred = model.predict(X_test)

print(f"\nTest — MAE: {mean_absolute_error(y_test, y_test_pred):.2f}  R²: {r2_score(y_test, y_test_pred):.4f}")

# ── Opslaan ───────────────────────────────────────────────────────────────────
SAVE_PATH = r'C:\Users\renzo\PyCharmMiscProject\.venv\pracitce enterprise\powercost_final_model_joblib\xgb_price_model.joblib'

joblib.dump({
    "model":      model,
    "dates_test": dates_test,
    "y_test":     y_test,
    "y_pred":     y_test_pred,
}, SAVE_PATH)

