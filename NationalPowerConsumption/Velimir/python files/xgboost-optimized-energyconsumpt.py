#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import holidays  # pip install holidays


# 1. LOAD DATA
#%%
df = pd.read_csv("./csv files/electricity_consumption_ready-copy.csv")

df["DateUTC"] = pd.to_datetime(df["DateUTC"])
df = df.sort_values("DateUTC").reset_index(drop=True)
df = df[["DateUTC", "Value", "temperature_2m"]]


# 2. SETTINGS

FREQ = 4
MONTH_STEPS = 30 * 24 * FREQ   # 2880 steps = ~1 month


# 3. FEATURE ENGINEERING

BE_HOLIDAYS = holidays.Belgium(years=range(2015, 2030))

def create_features(data):
    d = data.copy()

    # Calendar
    d["dayofweek"] = d["DateUTC"].dt.dayofweek
    d["month"] = d["DateUTC"].dt.month
    d["quarter"] = d["DateUTC"].dt.quarter
    d["is_weekend"] = (d["dayofweek"] >= 5).astype(int)

    # Cyclical hour
    hour = d["DateUTC"].dt.hour + d["DateUTC"].dt.minute / 60.0
    d["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    d["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # Holiday flag
    d["is_holiday"] = d["DateUTC"].dt.date.map(lambda dt: 1 if dt in BE_HOLIDAYS else 0)

    # Lag features
    d["lag_96"] = d["Value"].shift(96)
    d["lag_672"] = d["Value"].shift(672)

    # Rolling means
    d["roll_mean_day"] = d["Value"].shift(96).rolling(96).mean()
    d["roll_mean_week"] = d["Value"].shift(672).rolling(672).mean()

    return d


df_feat = create_features(df)
df_feat = df_feat.dropna().reset_index(drop=True)


# 4. TRAIN / VALIDATION / TEST SPLIT
# Last month = test
# Month before that = validation
# Everything before = training

test = df_feat.iloc[-MONTH_STEPS:]
val = df_feat.iloc[-2 * MONTH_STEPS:-MONTH_STEPS]
train = df_feat.iloc[:-2 * MONTH_STEPS]

FEATURE_COLS = [
    "hour_sin", "hour_cos",
    "dayofweek", "month", "quarter",
    "is_weekend", "is_holiday",
    "lag_96", "lag_672",
    "roll_mean_day", "roll_mean_week",
    "temperature_2m"
]

X_train = train[FEATURE_COLS].copy()
y_train = train["Value"].copy()

X_val = val[FEATURE_COLS].copy()
y_val = val["Value"].copy()

X_test = test[FEATURE_COLS].copy()
y_test = test["Value"].copy()

dates_val = val["DateUTC"]
dates_test = test["DateUTC"]


# 5. SCALING
# IMPORTANT:
# Fit ONLY on train, then transform validation/test.
# Do NOT fit_transform the test set.

X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_scaled = X_scaler.fit_transform(X_train)
X_val_scaled = X_scaler.transform(X_val)
X_test_scaled = X_scaler.transform(X_test)

y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_val_scaled = y_scaler.transform(y_val.values.reshape(-1, 1)).ravel()
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).ravel()


# 6. TRAIN MODEL

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="reg:squarederror"
)

model.fit(
    X_train_scaled,
    y_train_scaled,
    eval_set=[
        (X_train_scaled, y_train_scaled),
        (X_val_scaled, y_val_scaled)
    ],
    verbose=False
)


# 7. VALIDATION PREDICTION

y_val_pred_scaled = model.predict(X_val_scaled)
y_val_pred = y_scaler.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).ravel()

val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
val_mae = mean_absolute_error(y_val, y_val_pred)
val_r2 = r2_score(y_val, y_val_pred)

print("Validation results (previous month)")
print(f"  RMSE : {val_rmse:.2f}")
print(f"  MAE  : {val_mae:.2f}")
print(f"  R²   : {val_r2:.4f}")


# 8. TEST / BACKTEST PREDICTION

y_pred_scaled = model.predict(X_test_scaled)
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nBacktest results (last month)")
print(f"  RMSE : {rmse:.2f}")
print(f"  MAE  : {mae:.2f}")
print(f"  R²   : {r2:.4f}")


# 9. RECURSIVE FORECAST (next month)

history = df.copy()
last_date = history["DateUTC"].iloc[-1]

future_predictions = []

for i in range(MONTH_STEPS):
    next_date = last_date + pd.Timedelta(minutes=15)

    # Temperature proxy: same slot from 30 days ago
    temp = history["temperature_2m"].iloc[-(96 * 30)]

    new_row = pd.DataFrame({
        "DateUTC": [next_date],
        "Value": [np.nan],
        "temperature_2m": [temp]
    })

    temp_history = pd.concat([history, new_row], ignore_index=True)
    temp_feat = create_features(temp_history)

    X_next = temp_feat.iloc[[-1]][FEATURE_COLS]
    X_next_scaled = X_scaler.transform(X_next)

    next_value_scaled = model.predict(X_next_scaled)[0]
    next_value = y_scaler.inverse_transform([[next_value_scaled]])[0, 0]

    future_predictions.append({
        "DateUTC": next_date,
        "Predicted": next_value
    })

    history.loc[len(history)] = [next_date, next_value, temp]
    last_date = next_date

    if (i + 1) % 500 == 0:
        print(f"  Forecast step {i + 1}/{MONTH_STEPS}")

forecast_df = pd.DataFrame(future_predictions)
print("Forecast complete.")


# 10. FEATURE IMPORTANCE

importance = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values()


# 11. PLOTS ONLY:
# - QQ plot
# - Scatter plot
# - Two weeks zoomed actual vs predicted

TWO_WEEKS_STEPS = 14 * 24 * FREQ   # 14 days * 24 hours * 4 samples/hour = 1344 rows

# Residuals for QQ plot
residuals = y_test - y_pred

fig = plt.figure(figsize=(18, 16))
fig.suptitle("Electricity Consumption — Evaluation Plots", fontsize=16, fontweight="bold", y=0.98)

gs = fig.add_gridspec(3, 1, hspace=0.45)

# ---------------------------------------------------------
# Plot 1: QQ plot of residuals
# ---------------------------------------------------------
ax1 = fig.add_subplot(gs[0, 0])

stats.probplot(residuals, dist="norm", plot=ax1)

ax1.set_title("QQ Plot of Backtest Residuals")
ax1.set_xlabel("Theoretical Quantiles")
ax1.set_ylabel("Ordered Residuals")

# ---------------------------------------------------------
# Plot 2: Scatter plot predicted vs actual
# ---------------------------------------------------------
ax2 = fig.add_subplot(gs[1, 0])

ax2.scatter(y_pred, y_test, alpha=0.3, s=10, edgecolors="none", label="Data points")

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())

ax2.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--",
    linewidth=1.5,
    label="Perfect prediction"
)

metrics_text = (
    f"RMSE: {rmse:.2f}\n"
    f"MAE:  {mae:.2f}\n"
    f"R²:   {r2:.4f}"
)

ax2.text(
    0.05, 0.95, metrics_text,
    transform=ax2.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8)
)

ax2.set_title("Predicted vs Actual — Backtest")
ax2.set_xlabel("Predicted Consumption")
ax2.set_ylabel("Actual Consumption")
ax2.legend()
ax2.set_aspect("equal", adjustable="box")

# ---------------------------------------------------------
# Plot 3: Two weeks zoomed actual vs predicted
# ---------------------------------------------------------
ax3 = fig.add_subplot(gs[2, 0])

dates_zoom = dates_test.iloc[-TWO_WEEKS_STEPS:]
y_test_zoom = y_test.iloc[-TWO_WEEKS_STEPS:]
y_pred_zoom = y_pred[-TWO_WEEKS_STEPS:]

ax3.plot(dates_zoom, y_test_zoom, label="Actual", alpha=0.8)
ax3.plot(dates_zoom, y_pred_zoom, label="Predicted", linestyle="--", alpha=0.8)

ax3.set_title("Backtest Zoom — Last Two Weeks")
ax3.set_xlabel("Date")
ax3.set_ylabel("Consumption")
ax3.legend()
ax3.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()