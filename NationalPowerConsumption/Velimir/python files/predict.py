
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

import torch
import torch.nn as nn


# 1. SETTINGS 

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()
CSV_PATH = BASE_DIR / "electricity_consumption_ready-copy.csv"
MODEL_PATH = BASE_DIR / "model.pth"
SCALER_PATH = BASE_DIR / "scaler.pkl"

DATE_COL = "DateUTC"
TARGET_COL = "Value"

STEPS_PER_HOUR = 4
STEPS_PER_DAY = 24 * STEPS_PER_HOUR

# MUST MATCH TRAINING SCRIPT
LOOKBACK = 3 * STEPS_PER_DAY   # 288

# 1 month forecast
FUTURE_STEPS = 30 * STEPS_PER_DAY   # 2880

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# 2. DEBUG PATHS
# -----------------------------
print("Script folder:", BASE_DIR)
print("CSV exists:", CSV_PATH.exists(), CSV_PATH)
print("Model exists:", MODEL_PATH.exists(), MODEL_PATH)
print("Scaler exists:", SCALER_PATH.exists(), SCALER_PATH)


# -----------------------------
# 3. MODEL CLASS
# MUST MATCH THE TRAINED MODEL EXACTLY
# -----------------------------
class MLPForecaster(nn.Module):
    def __init__(self, lookback, n_features):
        super().__init__()
        input_dim = lookback * n_features

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        return self.net(x)


# -----------------------------
# 4. LOAD MODEL + SCALER
# -----------------------------
model = MLPForecaster(lookback=LOOKBACK, n_features=5)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

scaler = joblib.load(SCALER_PATH)

print("Model and scaler loaded!")


# -----------------------------
# 5. LOAD DATA
# -----------------------------
df = pd.read_csv(CSV_PATH)
df[DATE_COL] = pd.to_datetime(df[DATE_COL])
df = df.sort_values(DATE_COL).reset_index(drop=True)
df = df[[DATE_COL, TARGET_COL]].copy()


# -----------------------------
# 6. ADD SAME TIME FEATURES
# -----------------------------
def add_time_features(frame):
    frame = frame.copy()
    frame["hour"] = frame[DATE_COL].dt.hour
    frame["minute"] = frame[DATE_COL].dt.minute
    frame["dayofweek"] = frame[DATE_COL].dt.dayofweek

    frame["step_in_day"] = frame["hour"] * 4 + (frame["minute"] // 15)

    frame["sin_day"] = np.sin(2 * np.pi * frame["step_in_day"] / 96)
    frame["cos_day"] = np.cos(2 * np.pi * frame["step_in_day"] / 96)
    frame["sin_week"] = np.sin(2 * np.pi * frame["dayofweek"] / 7)
    frame["cos_week"] = np.cos(2 * np.pi * frame["dayofweek"] / 7)

    return frame


df = add_time_features(df)

# SAME SCALING AS TRAINING
df["Value_scaled"] = scaler.transform(df[[TARGET_COL]]).flatten()

feature_cols = ["Value_scaled", "sin_day", "cos_day", "sin_week", "cos_week"]


# -----------------------------
# 7. CHECK ENOUGH HISTORY
# -----------------------------
if len(df) < LOOKBACK:
    raise ValueError(f"Not enough rows in CSV. Need at least {LOOKBACK}, got {len(df)}.")


# -----------------------------
# 8. RECURSIVE 1-MONTH FORECAST
# -----------------------------
history_df = df.copy()
history_features = history_df[feature_cols].values.astype(np.float32).tolist()

last_timestamp = history_df[DATE_COL].iloc[-1]

future_dates = pd.date_range(
    start=last_timestamp + pd.Timedelta(minutes=15),
    periods=FUTURE_STEPS,
    freq="15min"
)

future_preds_scaled = []

for future_dt in future_dates:
    current_window = np.array(history_features[-LOOKBACK:], dtype=np.float32)
    current_window_tensor = torch.tensor(current_window, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        next_pred_scaled = model(current_window_tensor).cpu().numpy()[0, 0]

    future_preds_scaled.append(next_pred_scaled)

    hour = future_dt.hour
    minute = future_dt.minute
    dayofweek = future_dt.dayofweek
    step_in_day = hour * 4 + (minute // 15)

    sin_day = np.sin(2 * np.pi * step_in_day / 96)
    cos_day = np.cos(2 * np.pi * step_in_day / 96)
    sin_week = np.sin(2 * np.pi * dayofweek / 7)
    cos_week = np.cos(2 * np.pi * dayofweek / 7)

    history_features.append([next_pred_scaled, sin_day, cos_day, sin_week, cos_week])

future_preds_scaled = np.array(future_preds_scaled).reshape(-1, 1)
future_preds = scaler.inverse_transform(future_preds_scaled).flatten()


# -----------------------------
# 9. SAVE 1-MONTH FORECAST
# -----------------------------
forecast_df = pd.DataFrame({
    "DateUTC": future_dates,
    "Predicted_Value": future_preds
})

forecast_output_path = BASE_DIR / "one_month_forecast.csv"
forecast_df.to_csv(forecast_output_path, index=False)

print(f"\n1-month forecast saved to: {forecast_output_path}")
print("\nFirst 10 rows of forecast:")
print(forecast_df.head(10))


# -----------------------------
# 10. LOAD SAVED FORECAST AGAIN
# -----------------------------
loaded_forecast_df = pd.read_csv(forecast_output_path)
loaded_forecast_df["DateUTC"] = pd.to_datetime(loaded_forecast_df["DateUTC"])

print("\nLoaded saved forecast:")
print(loaded_forecast_df.head(10))


# -----------------------------
# 11. PLOT LAST PART OF HISTORY + 1 MONTH FORECAST
# -----------------------------
history_plot_steps = min(14 * STEPS_PER_DAY, len(df))  # show last 14 days

past_dates = df[DATE_COL].iloc[-history_plot_steps:]
past_values = df[TARGET_COL].iloc[-history_plot_steps:]

plt.figure(figsize=(14, 6))
plt.plot(past_dates, past_values, label="Historical Actual")
plt.plot(loaded_forecast_df["DateUTC"], loaded_forecast_df["Predicted_Value"], label="1-Month Forecast")
plt.axvline(df[DATE_COL].iloc[-1], linestyle="--", label="Forecast Start")

plt.title("One-Month Ahead Forecast")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
