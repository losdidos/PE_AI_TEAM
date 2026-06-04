import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit





####_> Extra <_####


from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor


###################

# Data inladen
df = pd.read_csv("sprint3/data/whederdataHous_0.csv")
df["Time"] = pd.to_datetime(df["Time"])
df.set_index("Time", inplace=True)

df = df.drop(columns=["Appliance1"])
df = df.drop(columns=["Appliance2"])
df = df.drop(columns=["Appliance3"])
df = df.drop(columns=["Appliance4"])
df = df.drop(columns=["Appliance5"])
df = df.drop(columns=["Appliance6"])
df = df.drop(columns=["Appliance7"])
df = df.drop(columns=["Appliance8"])
df = df.drop(columns=["Appliance9"])



target = "Aggregate"


df["hour"] = df.index.hour
df["dayofweek"] = df.index.dayofweek
df["month"] = df.index.month


df["lag1"] = df[target].shift(1) # 15 minuten
df["lag4"] = df[target].shift(4) # 1 uur

df["lag8"] = df[target].shift(8) # 2 uur
df["lag96"] = df[target].shift(96) #1 dag

df["lag672"] = df[target].shift(672)# 1 week


df = df.dropna()

X = df.drop(columns=[target])
y = df[target]

# split
split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]


tscv = TimeSeriesSplit(n_splits=5)


"""
"XGBoost": {
        "model": XGBRegressor(),
        "params": {
            "n_estimators": [100, 200, 500, 600],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 6, 7],
            "random_state": [42, 2]
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

"""

# models
modems = {


    "ExtraTrees": {
        "model": ExtraTreesRegressor(random_state=42, n_jobs=-1),
        "params": {
            "n_estimators": [100, 300],
            "max_depth": [None, 10, 20],
        }
    },
    "HistGBM": {
        "model": HistGradientBoostingRegressor(random_state=42),
        "params": {
            "max_iter": [100, 300],
            "learning_rate": [0.05, 0.1],
            "max_depth": [None, 5, 10],
        }
    },




}

best_models = {}


for name, mp in modems.items():
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

    print("Best params:", grid.best_params_)
    print("Best CV score:", -grid.best_score_)



result = {}

for name, model in best_models.items():
    y_pred_i = model.predict(X_test)
    rmse_i = np.sqrt(mean_squared_error(y_test, y_pred_i))
    result[name] = {"rmse": rmse_i, "y_pred": y_pred_i}

#voorspellen met beste model
best_name = min(result, key=lambda n: result[n]["rmse"])
y_pred = result[best_name]["y_pred"]


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)




residuals = y_test - y_pred



fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# 🔹 1. Time series: echt vs voorspelling
axes[0].plot(y_test.index, y_test, label="Echt")
axes[0].plot(y_test.index, y_pred, label="Voorspelling")
axes[0].set_title("Echt vs Voorspelling")
axes[0].legend()

# 🔹 2. Scatter plot (belangrijk!)
axes[1].scatter(y_test, y_pred, alpha=0.5)
axes[1].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             color="red")
axes[1].set_title("Scatter: Predicted vs Actual")
axes[1].set_xlabel("Echt")
axes[1].set_ylabel("Voorspelling")

# 🔹 3. Residuen
axes[2].plot(y_test.index, residuals)
axes[2].axhline(0, color="red")
axes[2].set_title("Residuals (fouten)")

plt.tight_layout()
plt.show()