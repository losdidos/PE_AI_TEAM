import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import plot_importance
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import TextBox, Button
import joblib

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


# --- Data laden ---
df = pd.read_csv(r'C:\Users\renzo\PyCharmMiscProject\.venv\pracitce enterprise\Power_Production_Weather.csv')
df['DateUTC'] = pd.to_datetime(df['DateUTC'])
df = df.sort_values('DateUTC').reset_index(drop=True)
print(f"Loaded {len(df)} rows of data")

dates = df['DateUTC']
X = make_time_features(dates)
X['temperature'] = df['temperature_2m'].values
y = df['Value'].values
print(f"Created {X.shape[1]} features")

# --- Train/test split ---
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y[:split], y[split:]
dates_test = df['DateUTC'].iloc[split:]

# --- Model trainen ---
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
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)

# --- Evaluatie ---
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nResults on test data:")
print(f"  MAE: {mae:.4f}")
print(f"  R²:  {r2:.4f}")

# --- Feature importance ---
plt.figure(figsize=(8, 5))
plot_importance(model)
plt.tight_layout()
plt.show()

#store model

SAVE_PATH = r'C:\Users\renzo\PyCharmMiscProject\.venv\pracitce enterprise\powergeneration_final_model_joblib\powergenartion_final_model.joblib'

joblib.dump({
    "model":      model,
    "dates_test": dates_test,
    "y_test":     y_test,
    "y_pred":     y_pred,
}, SAVE_PATH)



