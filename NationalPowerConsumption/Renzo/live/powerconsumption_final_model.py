import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import TextBox, Button
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib


# 1. LOAD DATA

df = pd.read_csv(r"C:\Users\renzo\PyCharmMiscProject\.venv\pracitce enterprise\electricity_consumption_ready.csv")

df["DateUTC"] = pd.to_datetime(df["DateUTC"])
df = df.sort_values("DateUTC").reset_index(drop=True)
df = df[["DateUTC", "Value", "temperature_2m"]]


# 2. SETTINGS

FREQ        = 4
MONTH_STEPS = 30 * 24 * FREQ


# 3. FEATURE ENGINEERING

def create_features(data):
    d = data.copy()
    d["hour"]       = d["DateUTC"].dt.hour
    d["dayofweek"]  = d["DateUTC"].dt.dayofweek
    d["month"]      = d["DateUTC"].dt.month
    d["quarter"]    = d["DateUTC"].dt.quarter
    d["is_weekend"] = (d["dayofweek"] >= 5).astype(int)
    d["lag_96"]     = d["Value"].shift(96)
    d["lag_672"]    = d["Value"].shift(672)
    d["roll_mean_day"]  = d["Value"].shift(96).rolling(96).mean()
    d["roll_mean_week"] = d["Value"].shift(672).rolling(672).mean()
    return d


df_feat = create_features(df)
df_feat = df_feat.dropna().reset_index(drop=True)


# 4. TRAIN / TEST SPLIT

train = df_feat.iloc[:-MONTH_STEPS]
test  = df_feat.iloc[-MONTH_STEPS:]

FEATURE_COLS = [
    "hour", "dayofweek", "month", "quarter", "is_weekend",
    "lag_96", "lag_672",
    "roll_mean_day", "roll_mean_week",
    "temperature_2m"
]

X_train, y_train = train[FEATURE_COLS], train["Value"]
X_test,  y_test  = test[FEATURE_COLS],  test["Value"]
dates_test = test["DateUTC"]


# 5. TRAIN MODEL

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)


# 6. BACKTEST

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("Backtest results (last month)")
print(f"  RMSE : {rmse:.2f}")
print(f"  MAE  : {mae:.2f}")
print(f"  R²   : {r2:.4f}")




# 7. RECURSIVE FORECAST

history   = df.copy()
last_date = history["DateUTC"].iloc[-1]
future_predictions = []

for i in range(MONTH_STEPS):
    next_date = last_date + pd.Timedelta(minutes=15)
    temp      = history["temperature_2m"].iloc[-(96 * 30)]
    new_row   = pd.DataFrame({"DateUTC": [next_date], "Value": [np.nan], "temperature_2m": [temp]})
    temp_history = pd.concat([history, new_row], ignore_index=True)
    temp_feat    = create_features(temp_history)
    X_next       = temp_feat.iloc[[-1]][FEATURE_COLS]
    next_value   = model.predict(X_next)[0]
    future_predictions.append({"DateUTC": next_date, "Predicted": next_value})
    history.loc[len(history)] = [next_date, next_value, temp]
    last_date = next_date
    if (i + 1) % 500 == 0:
        print(f"  Forecast step {i + 1}/{MONTH_STEPS}")

forecast_df = pd.DataFrame(future_predictions)
print("Forecast complete.")


# 8. FEATURE IMPORTANCE

importance = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values()
plt.figure(figsize=(8, 5))
importance.plot(kind="barh")
plt.title("Feature Importance")
plt.xlabel("Score")
plt.tight_layout()
plt.show()


# 9. FULL OVERVIEW PLOT

plt.figure(figsize=(16, 6))
plt.plot(df["DateUTC"], df["Value"], label="Historical", alpha=0.7)
plt.plot(dates_test, y_pred, label="Backtest Predicted", linestyle="--")
plt.plot(forecast_df["DateUTC"], forecast_df["Predicted"], label="Forecast (next month)", linestyle="--")
plt.axvline(df["DateUTC"].iloc[-MONTH_STEPS], color="grey", linestyle=":", label="Backtest Start")
plt.axvline(df["DateUTC"].iloc[-1],           color="red",  linestyle=":", label="Forecast Start")
plt.legend()
plt.title("Electricity Consumption — Full Overview")
plt.xlabel("Date")
plt.ylabel("Consumption")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 10. ZOOM: BACKTEST

plt.figure(figsize=(16, 6))
plt.plot(dates_test, y_test, label="Actual",    alpha=0.8)
plt.plot(dates_test, y_pred, label="Predicted", linestyle="--", alpha=0.8)
plt.legend()
plt.title("Backtest: Actual vs Predicted (last month)")
plt.xlabel("Date")
plt.ylabel("Consumption")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 11. ZOOM: FORECAST

plt.figure(figsize=(16, 6))
plt.plot(forecast_df["DateUTC"], forecast_df["Predicted"], color="green", label="Forecast")
plt.legend()
plt.title("Forecast: Next Month")
plt.xlabel("Date")
plt.ylabel("Consumption")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 12. SCATTER PLOT

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_pred, y_test, alpha=0.3, s=10, color="#2563EB", edgecolors="none", label="Data points")
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], color="#EF4444", linewidth=1.5, linestyle="--", label="Perfect prediction (y = x)")
textstr = f"RMSE: {rmse:.2f}\nMAE:  {mae:.2f}\nR²:   {r2:.4f}"
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#CBD5E1", alpha=0.8))
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("True Value", fontsize=12)
ax.set_title("Predicted vs Actual — Backtest (last month)", fontsize=14)
ax.legend(fontsize=10)
ax.set_aspect("equal", adjustable="box")
plt.tight_layout()
plt.show()

joblib.dump({
    "model":      model,
    "dates_test": dates_test,
    "y_test":     y_test,
    "y_pred":     y_pred,
    "forecast_df": forecast_df
}, r"C:\Users\renzo\PyCharmMiscProject\.venv\pracitce enterprise\powerconsumption_model\Powerconsumption.joblib")

filename = 'Powerconsumption.joblib'
joblib.dump(model, filename)