import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# 1. SETTINGS
# =========================
FILE_PATH = "./electricity_consumption_ready.csv"  
FREQ_MINUTES = 15
STEPS_PER_HOUR = 60 // FREQ_MINUTES               # 4
STEPS_PER_DAY = 24 * STEPS_PER_HOUR               # 96
STEPS_PER_WEEK = 7 * STEPS_PER_DAY                # 672

TEST_DAYS = 7
TEST_SIZE = TEST_DAYS * STEPS_PER_DAY             # last 7 days for testing

FUTURE_DAYS = 30
FUTURE_STEPS = FUTURE_DAYS * STEPS_PER_DAY        # 1 month forecast

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =========================
# 2. LOAD DATA
# =========================
df = pd.read_csv(FILE_PATH, sep=";")

df["DateUTC"] = pd.to_datetime(df["DateUTC"])
df = df.sort_values("DateUTC").reset_index(drop=True)

# Keep only needed columns
df = df[["DateUTC", "Value", "temperature_2m"]].copy()

# =========================
# 3. FEATURE ENGINEERING
# =========================
# Calendar features
df["hour"] = df["DateUTC"].dt.hour
df["minute"] = df["DateUTC"].dt.minute
df["dayofweek"] = df["DateUTC"].dt.dayofweek
df["month"] = df["DateUTC"].dt.month
df["dayofyear"] = df["DateUTC"].dt.dayofyear

# Cyclical time encoding
hour_decimal = df["hour"] + df["minute"] / 60.0
df["hour_sin"] = np.sin(2 * np.pi * hour_decimal / 24)
df["hour_cos"] = np.cos(2 * np.pi * hour_decimal / 24)

df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# Lag features - ONLY past values
df["lag_1"] = df["Value"].shift(1)                    # 15 min ago
df["lag_4"] = df["Value"].shift(4)                    # 1 hour ago
df["lag_96"] = df["Value"].shift(96)                  # 1 day ago
df["lag_672"] = df["Value"].shift(672)                # 1 week ago

# Rolling features - shift first to avoid leakage
df["roll_4"] = df["Value"].shift(1).rolling(4).mean()
df["roll_96"] = df["Value"].shift(1).rolling(96).mean()
df["roll_672"] = df["Value"].shift(1).rolling(672).mean()

# Optional temperature rolling features
df["temp_lag_1"] = df["temperature_2m"].shift(1)
df["temp_roll_4"] = df["temperature_2m"].shift(1).rolling(4).mean()
df["temp_roll_96"] = df["temperature_2m"].shift(1).rolling(96).mean()

# Drop NaNs created by lagging/rolling
df = df.dropna().reset_index(drop=True)

# =========================
# 4. TRAIN / TEST SPLIT
# =========================
train_df = df.iloc[:-TEST_SIZE].copy()
test_df = df.iloc[-TEST_SIZE:].copy()

feature_cols = [
    "temperature_2m",
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    "lag_1", "lag_4", "lag_96", "lag_672",
    "roll_4", "roll_96", "roll_672",
    "temp_lag_1", "temp_roll_4", "temp_roll_96"
]

X_train = train_df[feature_cols].copy()
y_train = train_df["Value"].copy()

X_test = test_df[feature_cols].copy()
y_test = test_df["Value"].copy()

# =========================
# 5. SCALE FEATURES
# =========================
x_scaler = StandardScaler()
X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)

# Scale target too - helps neural nets train better
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

# =========================
# 6. BUILD MODEL
# =========================
model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.15),
    Dense(64, activation="relu"),
    Dropout(0.10),
    Dense(32, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

# =========================
# 7. TRAIN
# =========================
history = model.fit(
    X_train_scaled,
    y_train_scaled,
    validation_split=0.1,
    epochs=30,
    batch_size=128,
    callbacks=[early_stopping],
    verbose=1
)

# =========================
# 8. TEST PREDICTION
# =========================
y_pred_scaled = model.predict(X_test_scaled, verbose=0).flatten()
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nTest results:")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

# =========================
# 9. RECURSIVE 1-MONTH FORECAST
# =========================
# We forecast one step at a time and feed predictions back in.
history_df = df.copy()
future_predictions = []

for step in range(FUTURE_STEPS):
    next_time = history_df["DateUTC"].iloc[-1] + pd.Timedelta(minutes=FREQ_MINUTES)

    # Simple temperature assumption:
    # use temperature from 24 hours ago if available, otherwise last known temperature
    if len(history_df) >= STEPS_PER_DAY:
        temp_next = history_df["temperature_2m"].iloc[-STEPS_PER_DAY]
    else:
        temp_next = history_df["temperature_2m"].iloc[-1]

    hour = next_time.hour
    minute = next_time.minute
    dayofweek = next_time.dayofweek
    month = next_time.month
    dayofyear = next_time.dayofyear

    hour_decimal = hour + minute / 60.0

    next_row = {
        "DateUTC": next_time,
        "temperature_2m": temp_next,
        "hour": hour,
        "minute": minute,
        "dayofweek": dayofweek,
        "month": month,
        "dayofyear": dayofyear,
        "hour_sin": np.sin(2 * np.pi * hour_decimal / 24),
        "hour_cos": np.cos(2 * np.pi * hour_decimal / 24),
        "dow_sin": np.sin(2 * np.pi * dayofweek / 7),
        "dow_cos": np.cos(2 * np.pi * dayofweek / 7),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
    }

    # Value-based features from history only
    next_row["lag_1"] = history_df["Value"].iloc[-1]
    next_row["lag_4"] = history_df["Value"].iloc[-4]
    next_row["lag_96"] = history_df["Value"].iloc[-96]
    next_row["lag_672"] = history_df["Value"].iloc[-672]

    next_row["roll_4"] = history_df["Value"].iloc[-4:].mean()
    next_row["roll_96"] = history_df["Value"].iloc[-96:].mean()
    next_row["roll_672"] = history_df["Value"].iloc[-672:].mean()

    next_row["temp_lag_1"] = history_df["temperature_2m"].iloc[-1]
    next_row["temp_roll_4"] = history_df["temperature_2m"].iloc[-4:].mean()
    next_row["temp_roll_96"] = history_df["temperature_2m"].iloc[-96:].mean()

    X_next = pd.DataFrame([next_row])[feature_cols]
    X_next_scaled = x_scaler.transform(X_next)

    pred_scaled = model.predict(X_next_scaled, verbose=0).flatten()[0]
    pred_value = y_scaler.inverse_transform([[pred_scaled]])[0, 0]

    future_predictions.append(pred_value)

    row_to_append = next_row.copy()
    row_to_append["Value"] = pred_value

    history_df = pd.concat([history_df, pd.DataFrame([row_to_append])], ignore_index=True)

future_dates = pd.date_range(
    start=df["DateUTC"].iloc[-1] + pd.Timedelta(minutes=FREQ_MINUTES),
    periods=FUTURE_STEPS,
    freq=f"{FREQ_MINUTES}min"
)

# =========================
# 10. PLOTS
# =========================

# Loss curve
plt.figure(figsize=(12, 5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Test set actual vs predicted
plt.figure(figsize=(14, 6))
plt.plot(test_df["DateUTC"], y_test.values, label="Actual")
plt.plot(test_df["DateUTC"], y_pred, label="Predicted")
plt.title("Test Set: Actual vs Predicted")
plt.xlabel("Date")
plt.ylabel("Electricity Consumption")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Recent history + test prediction + future forecast
recent_steps = 14 * STEPS_PER_DAY  # last 14 days
recent_df = df.iloc[-recent_steps:].copy()

plt.figure(figsize=(16, 7))
plt.plot(recent_df["DateUTC"], recent_df["Value"], label="Recent Actual")
plt.plot(test_df["DateUTC"], y_pred, label="Test Prediction")
plt.plot(future_dates, future_predictions, label="Future Forecast (30 days)")
plt.title("Recent Actual + Test Prediction + 30-Day Forecast")
plt.xlabel("Date")
plt.ylabel("Electricity Consumption")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()