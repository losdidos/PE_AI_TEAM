import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit

# Reading CSV
df = pd.read_csv("./Sprint 4/household-data-appliances-interpolated/CleerResampel_House_7.csv", delimiter=",")
df["Time"] = pd.to_datetime(df["Time"], format="%Y-%m-%d %H:%M:%S")
df = df.sort_values("Time")
df = df.set_index("Time")


APPLIANCES_COLUMNS = [c for c in df.columns if c.startswith("Appliance")]

df = df.drop(columns=["Unix"] + APPLIANCES_COLUMNS)

# FEATURES ENGINEERING
def create_features(df):
    """
    Creating features based on time index
    """
    df = df.copy()

    df["quarter"] = df.index.minute // 15
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["isweekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["month"] = df.index.month

    # Lag features
    lag_steps = [1, 4, 96, 672]  # 15 minutes, 1 hour, 1 day, 1 week
    for lag_step in lag_steps:
        df[f"lag_{lag_step}"] = df["Aggregate"].shift(lag_step)

    return df

df = create_features(df)
df = df.dropna().copy()

# TRAIN TEST SPLIT
FEATURES = [c for c in df.columns if c != "Aggregate"]
TARGET = "Aggregate"

tss = TimeSeriesSplit(n_splits=5, test_size=96 * 7, gap=96)

results = []
fold = 0

for train_index, test_index in tss.split(df):
    train = df.iloc[train_index]
    test = df.iloc[test_index]

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    model = XGBRegressor(
        base_score=0.5,
        n_estimators=1000,
        early_stopping_rounds=50,
        random_state=42,
        learning_rate=0.10,
        objective="reg:squarederror",
        max_depth=6
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)

    fold += 1
    print(f"Fold {fold}: MAE={mae:.3f}, R2={r2:.3f}, MAPE={mape:.3f}")

    results.append({
        "fold": fold,
        "mae": mae,
        "r2": r2,
        "MAPE": mape
    })

results_df = pd.DataFrame(results)

print("\nResults by fold:")
print(results_df)

print("\nAverage results:")
print(results_df.mean(numeric_only=True))

#PLOT

plt.figure(figsize=(14,6))
plt.plot(y_test.index, y_test, label="Actual")
plt.plot(y_test.index, predictions, label="Predicted")
plt.title("Actual vs Predicted")
plt.xlabel("Time")
plt.ylabel("Aggregate")
plt.legend()
plt.show()
errors = y_test - predictions
stats.probplot(errors, dist="norm", plot=plt)
plt.title("QQ Plot")
plt.show()
