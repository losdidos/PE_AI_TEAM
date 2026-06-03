import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


dataframe = pd.read_csv(r"C:\Users\winst\Downloads\School\Thomas More\2de jaar\2de semester\PE2-AI\PE2-AI\clean_household_weather_extended.csv", sep=",")

train_size = int(len(dataframe) * 0.7)
val_size = int(len(dataframe) * 0.15)

training_df = dataframe.iloc[:train_size]
validation_df = dataframe.iloc[train_size:train_size + val_size]
test_df = dataframe.iloc[train_size + val_size:]

X_train = training_df.drop(['aggregate_w'], axis=1)
X_train["timestamp"] = pd.to_datetime(X_train["timestamp"])
X_train["timestamp"] = X_train["timestamp"].astype('int64') // 10**9
y_train = training_df['aggregate_w']

X_val = validation_df.drop(['aggregate_w'], axis=1)
X_val["timestamp"] = pd.to_datetime(X_val["timestamp"])
X_val["timestamp"] = X_val["timestamp"].astype('int64') // 10**9
y_val = validation_df['aggregate_w']

X_test = test_df.drop(['aggregate_w'], axis=1)
X_test["timestamp"] = pd.to_datetime(X_test["timestamp"])
X_test["timestamp"] = X_test["timestamp"].astype('int64') // 10**9
y_test = test_df['aggregate_w']


model = xgb.XGBRegressor(n_estimators=100, max_depth=10, learning_rate=0.1)
model.fit(X_train, y_train)

y_XGB_predictions = model.predict(X_test)

MAE_XGBoost = mean_absolute_error(y_test, y_XGB_predictions)
R2_XGBoost = r2_score(y_test, y_XGB_predictions)

print(f'''
#-------------------------------------------------------------------#
|                            XGBoosts                               |
#-------------------------------------------------------------------#
|                                                                   |
|   Test MAE: {round(MAE_XGBoost, 2)}                                                |
|   Test R2: {round(R2_XGBoost, 2)}                                                    |
|                                                                   |
#-------------------------------------------------------------------#
''')

# Make sure timestamps are truly datetime
test_df = test_df.copy()
test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])

plt.figure(figsize=(14, 6))

# Plot actual + predicted
plt.plot(test_df["timestamp"], y_test, color="blue", label="Actual")
plt.plot(test_df["timestamp"], y_XGB_predictions, color="red", label="Predicted")

# Labels + title
plt.title("Household Power Consumption Timeline", fontsize=20)
plt.xlabel("Date", fontsize=20)
plt.ylabel("Power Consumption (W)", fontsize=20)

# Tick styling
plt.tick_params(axis='both', labelsize=20)

# 🔥 FIX: clean date display (every 6 months)
plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# Rotate for readability
plt.gcf().autofmt_xdate()

# Legend
plt.legend(fontsize=20)

plt.tight_layout()
plt.savefig("XGBoost_household_forwardfill.png", bbox_inches="tight")
plt.show()

# Scatter plot of predictions vs actual
plt.scatter(y_XGB_predictions, y_test, color="blue", alpha=0.5, label="Predictions")

# Add a y=x line for reference
min_val = min(y_test.min(), y_XGB_predictions.min())
max_val = max(y_test.max(), y_XGB_predictions.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', label="Perfect Prediction")

plt.xlabel("Predicted (W)", fontsize=20)
plt.ylabel("Actual (W)", fontsize=20)
plt.title("Actual vs Predicted Household Power Consumption", fontsize=20)
plt.legend(fontsize=20)
plt.tick_params(axis='both', labelsize=20)
plt.savefig("XGBoost_Household_ParityPlot")
plt.show()