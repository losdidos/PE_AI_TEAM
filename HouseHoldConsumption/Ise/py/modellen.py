import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor


"""
all data csv

sprint3/data/whederdataHous_1_metMetaData.csv
sprint3/data/whederdataHous_2_metMetaData.csv
sprint3/data/whederdataHous_3_metMetaData.csv
sprint3/data/whederdataHous_5_metMetaData.csv
sprint3/data/whederdataHous_6_metMetaData.csv
sprint3/data/whederdataHous_7_metMetaData.csv
sprint3/data/whederdataHous_8_metMetaData.csv
"""

df = pd.read_csv("sprint3/data/whederdataHous_1_metMetaData.csv")
df["Time"] = pd.to_datetime(df["Time"])
df.set_index("Time", inplace=True)


df = df.drop(columns=[f"Appliance{i}" for i in range(1, 10)], errors="ignore")

target = "Aggregate"


df["hour"]      = df.index.hour
df["dayofweek"] = df.index.dayofweek
df["month"]     = df.index.month

df["lag1"]   = df[target].shift(1)    # 15 minuten
df["lag4"]   = df[target].shift(4)    # 1 uur
df["lag96"]  = df[target].shift(96)   # 1 dag

df["lag2"] = df[target].shift(2)
df["lag3"] = df[target].shift(3)
df["lag8"] = df[target].shift(8)      # 2 uur
df["lag672"] = df[target].shift(672)

df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

feature_cols = [
    "hour",
    "dayofweek",
    "month",
    "lag1",
    "lag4",
    "lag96",
    "temperature_2m (°C)",#temperature_2m (Â°C),relative_humidity_2m (%)
    "relative_humidity_2m (%)",
    "lag672",
    "lag8",
    "lag3",
    "lag2",
    "is_weekend",
    "Appliances Owned",
    "Detached",
    "Semi-detached",
    "Mid-terrace",
    "Size"

]

df = df.dropna()

X = df[feature_cols]
y = df[target]



split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

tscv = TimeSeriesSplit(n_splits=5)



models = {
    "XGBoost": {
        "model": XGBRegressor(),
        "params": {
            "n_estimators": [100, 200, 500, 600],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 6, 7],
            "random_state": [42, 2],
            "subsample": [0.7, 0.8, 0.9],
        }
    },
    "LightGBM": {
        "model": LGBMRegressor(),
        "params": {
            "n_estimators": [100, 300],
            "max_depth": [-1, 5, 10],
            "learning_rate": [0.05, 0.1]
        }
    },

    "RandomForest": {
        "model": RandomForestRegressor(),
        "params": {
            "n_estimators": [100, 300],
            "max_depth": [None, 10, 20]
        }
    },

    "GradientBoosting": {
        "model": GradientBoostingRegressor(),
        "params": {
            "n_estimators": [100, 300],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5]
        }
    },
    "LightGBM": {
        "model": LGBMRegressor(),
        "params": {
            "n_estimators": [100, 300],
            "max_depth": [5, 10],
            "learning_rate": [0.05, 0.1]
        }
    }
}



best_models = {}

for name, mp in models.items():
    print(f"Running GridSearch for {name}...")
    grid = GridSearchCV(
        estimator=mp["model"],
        param_grid=mp["params"],
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    best_models[name] = grid.best_estimator_
    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV RMSE: {-grid.best_score_:.2f}")





result = {}
for name, model in best_models.items():
    y_pred_i = model.predict(X_test)
    rmse_i = np.sqrt(mean_squared_error(y_test, y_pred_i))
    result[name] = {"rmse": rmse_i, "y_pred": y_pred_i}
    print(f"{name} RMSE: {rmse_i:.2f}")

best_name = min(result, key=lambda n: result[n]["rmse"])
y_pred = result[best_name]["y_pred"]
rmse = result[best_name]["rmse"]

print(f"\nBest model: {best_name} — RMSE: {rmse:.2f}")
print("Gemiddelde echte waarde:", y_test.mean())
print("Gemiddelde voorspelling:", y_pred.mean())





residuals = y_test - pd.Series(y_pred, index=y_test.index)





fig, axes = plt.subplots(3, 1, figsize=(14, 12))

axes[0].plot(y_test.index, y_test.values, label="Echt")
axes[0].plot(y_test.index, y_pred, label="Voorspelling")
axes[0].set_title(f"Echt vs Voorspelling ({best_name})")
axes[0].legend()

axes[1].scatter(y_test, y_pred, alpha=0.5)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red")
axes[1].set_title("Scatter: Predicted vs Actual")
axes[1].set_xlabel("Echt")
axes[1].set_ylabel("Voorspelling")

axes[2].plot(y_test.index, residuals)
axes[2].axhline(0, color="red")
axes[2].set_title("Residuals (fouten)")

plt.tight_layout()
plt.show()