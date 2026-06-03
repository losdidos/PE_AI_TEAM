import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

# =========================
# SETTINGS
# =========================
FILE_PATH = "./electricity_consumption_ready.csv"
FREQ_MINUTES = 15
STEPS_PER_HOUR = 60 // FREQ_MINUTES  # 4
STEPS_PER_DAY = 24 * STEPS_PER_HOUR  # 96
STEPS_PER_WEEK = 7 * STEPS_PER_DAY   # 672

# Use more data for training, test on last week
TEST_DAYS = 7
TEST_SIZE = TEST_DAYS * STEPS_PER_DAY

# Forecast next week
FORECAST_DAYS = 7
FORECAST_STEPS = FORECAST_DAYS * STEPS_PER_DAY

# ARIMA settings
ORDER = (5, 1, 0)  # ARIMA(p,d,q) - adjust based on data
SEASONAL_ORDER = (1, 1, 1, STEPS_PER_DAY)  # Seasonal ARIMA

SEED = 42
np.random.seed(SEED)

# =========================
# LOAD AND PREPROCESS DATA
# =========================
df = pd.read_csv(FILE_PATH, sep=";")
df["DateUTC"] = pd.to_datetime(df["DateUTC"])
df = df.sort_values("DateUTC").reset_index(drop=True)

# Keep relevant columns
df = df[["DateUTC", "Value", "temperature_2m"]].copy()

# Set DateUTC as index for time series
df.set_index("DateUTC", inplace=True)

# Split data
train_data = df.iloc[:-TEST_SIZE]
test_data = df.iloc[-TEST_SIZE:]

print(f"Train shape: {train_data.shape}, Test shape: {test_data.shape}")

# =========================
# BUILD SARIMAX MODEL
# =========================
# Fit SARIMAX model on training data with temperature as exogenous
exog_train = train_data[["temperature_2m"]]
exog_test = test_data[["temperature_2m"]]

model = SARIMAX(train_data["Value"], exog=exog_train, order=ORDER, seasonal_order=SEASONAL_ORDER)
model_fit = model.fit(disp=False)

print("Model fitted successfully!")
print(model_fit.summary())

# =========================
# EVALUATE ON TEST SET
# =========================
# Forecast the test period with exogenous
forecast_test = model_fit.forecast(steps=TEST_SIZE, exog=exog_test)

y_pred = forecast_test.values
y_actual = test_data["Value"].values

# Calculate metrics
mae = mean_absolute_error(y_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
r2 = r2_score(y_actual, y_pred)

print("\nTest Set Performance:")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.4f}")

# =========================
# FORECAST NEXT WEEK
# =========================
# Create future exogenous (approximate temperature from last week)
future_exog = []
for i in range(FORECAST_STEPS):
    # Use temperature from same time last week
    temp_idx = -STEPS_PER_WEEK + i if len(df) >= STEPS_PER_WEEK else -1
    future_exog.append(df["temperature_2m"].iloc[temp_idx])

future_exog = pd.DataFrame({"temperature_2m": future_exog})

# Forecast future values
future_forecast = model_fit.forecast(steps=FORECAST_STEPS, exog=future_exog)
future_pred = future_forecast.values

# Create future dates
last_date = df.index[-1]
future_dates = pd.date_range(
    start=last_date + pd.Timedelta(minutes=FREQ_MINUTES),
    periods=FORECAST_STEPS,
    freq=f"{FREQ_MINUTES}min"
)

# =========================
# PLOTS
# =========================

# 1. Model Diagnostics
model_fit.plot_diagnostics(figsize=(12, 8))
plt.tight_layout()
plt.savefig('arima_diagnostics.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Test Set: Actual vs Predicted
test_dates = test_data.index

plt.figure(figsize=(16, 8))
plt.plot(test_dates, y_actual, label='Actual', linewidth=2)
plt.plot(test_dates, y_pred, label='Predicted', linewidth=2, alpha=0.8)
plt.title(f'Test Set: Actual vs Predicted (MAE: {mae:.2f}, R²: {r2:.4f})')
plt.xlabel('Date')
plt.ylabel('Electricity Consumption')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('test_predictions.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Next Week Forecast
plt.figure(figsize=(16, 8))

# Show recent actual data (last 3 days)
recent_days = 3
recent_steps = recent_days * STEPS_PER_DAY
recent_dates = df.index[-recent_steps:]
recent_values = df["Value"].iloc[-recent_steps:]

plt.plot(recent_dates, recent_values, label='Recent Actual', linewidth=2, color='blue')
plt.plot(future_dates, future_pred, label='Next Week Forecast', linewidth=2, color='red', linestyle='--')

plt.title('Next Week Electricity Consumption Forecast')
plt.xlabel('Date')
plt.ylabel('Electricity Consumption')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('next_week_forecast.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Combined View: Recent + Forecast
plt.figure(figsize=(18, 8))

# Last 7 days actual + next 7 days forecast
combined_dates = pd.concat([pd.Series(recent_dates), pd.Series(future_dates)])
combined_values = np.concatenate([recent_values.values, future_pred])

plt.plot(combined_dates, combined_values, linewidth=2)
plt.axvline(x=recent_dates[-1], color='red', linestyle='--', alpha=0.7, label='Forecast Start')
plt.title('7 Days Actual + 7 Days Forecast')
plt.xlabel('Date')
plt.ylabel('Electricity Consumption')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('combined_view.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nModel training and evaluation complete!")
print(f"Plots saved as PNG files in the current directory.")
print(f"Next week forecast covers: {future_dates.min()} to {future_dates.max()}")