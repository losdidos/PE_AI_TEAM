import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Data inladen
df = pd.read_csv("sprint3/data/whederdataHous_0.csv")
df["Time"] = pd.to_datetime(df["Time"])
df.set_index("Time", inplace=True)

df["lag1"]   = df[target].shift(1)    # 15 minuten
df["lag4"]   = df[target].shift(4)    # 1 uur
df["lag96"]  = df[target].shift(96)   # 1 dag

df["lag2"] = df["Appliance1"].shift(2)
df["lag3"] = df[target].shift(3)
df["lag8"] = df[target].shift(8)      # 2 uur
df["lag672"] = df[target].shift(672)
df = df.drop(columns=[f"Appliance{i}" for i in range(1, 10)], errors="ignore")

target = "Aggregate"

# Tijd features
df["hour"] = df.index.hour
df["dayofweek"] = df.index.dayofweek
df["month"] = df.index.month

# Lag features (HEEL belangrijk)
df["lag1"]   = df[target].shift(1)    # 15 minuten
df["lag4"]   = df[target].shift(4)    # 1 uur
df["lag96"]  = df[target].shift(96)   # 1 dag

df["lag2"] = df[target].shift(2)
df["lag3"] = df[target].shift(3)
df["lag8"] = df[target].shift(8)      # 2 uur
df["lag672"] = df[target].shift(672)

df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

# Drop NaN door lag
df = df.dropna()

# Features en target
X = df.drop(columns=[target])
y = df[target]

# Train/test split (time series!)
split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

# Model
model = XGBRegressor(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

#Voorspellen
y_pred = model.predict(X_test)

#Evaluatie
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)
print("Gemiddelde echte waarde:", y_test.mean())
print("Gemiddelde voorspelling:", y_pred.mean())

# Plot
plt.figure(figsize=(12,6))
plt.plot(y_test.index, y_test, label="Echt")
plt.plot(y_test.index, y_pred, label="Voorspelling")
plt.legend()
plt.show()