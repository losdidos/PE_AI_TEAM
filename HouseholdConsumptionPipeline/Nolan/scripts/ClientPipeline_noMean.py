import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import load
import matplotlib.pyplot as plt




# static input , change ith db pull


occ = input("Occupancy: ")
con_year = input("Construction Year: ")
appl = input("Number of Appliances: ")
house_type = input("House Type (0 atatched or 1 detached): ")
bedrooms = input("Number of Bedrooms: ")

inp = [occ, con_year, appl, house_type, bedrooms]   


# load models
knn_model = load(r"C:\Users\nolan\OneDrive\Documents\GitHub\PE2-AI\HOUSEHOLDPIPELINE\Top7_XGBoost_Active\models\knn_classifier.joblib")
knn_scaler = load(r"C:\Users\nolan\OneDrive\Documents\GitHub\PE2-AI\HOUSEHOLDPIPELINE\Top7_XGBoost_Active\models\knn_scaler.joblib")
xgb6  = load(r"C:\Users\nolan\OneDrive\Documents\GitHub\PE2-AI\HOUSEHOLDPIPELINE\models_no_mean_meta\xgboost_model_house_06.joblib")
xgb15 = load(r"C:\Users\nolan\OneDrive\Documents\GitHub\PE2-AI\HOUSEHOLDPIPELINE\models_no_mean_meta\xgboost_model_house_15.joblib")
xgb18 = load(r"C:\Users\nolan\OneDrive\Documents\GitHub\PE2-AI\HOUSEHOLDPIPELINE\models_no_mean_meta\xgboost_model_house_18.joblib")
xgb19 = load(r"C:\Users\nolan\OneDrive\Documents\GitHub\PE2-AI\HOUSEHOLDPIPELINE\models_no_mean_meta\xgboost_model_house_19.joblib")
xgb20 = load(r"C:\Users\nolan\OneDrive\Documents\GitHub\PE2-AI\HOUSEHOLDPIPELINE\models_no_mean_meta\xgboost_model_house_20.joblib")

# KNN pick

inp_scaled   = knn_scaler.transform([inp])
house_to_use = int(knn_model.predict(inp_scaled)[0])
print(house_to_use)



# pick xgboost (kon switch zijn )
xgb_models = {6: xgb6, 15: xgb15, 18: xgb18, 19: xgb19, 20: xgb20}
model = xgb_models[house_to_use]




#pred loop


# load matched house 
DATA_DIR = Path(r"C:\Users\nolan\OneDrive\Documents\GitHub\PE2-AI\HOUSEHOLDPIPELINE\Top7_XGBoost_Active\data_source")
df = pd.read_csv(DATA_DIR / f"CLEAN_House{house_to_use}_15min_interpolated_with_metadata.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Lag features
df["lag_15m"] = df["aggregate_w"].shift(1)
df["lag_1h"] = df["aggregate_w"].shift(4)
df["lag_3h"] = df["aggregate_w"].shift(12)
df["lag_6h"] = df["aggregate_w"].shift(24)
df["lag_12h"] = df["aggregate_w"].shift(48)
df["lag_1d"] = df["aggregate_w"].shift(96)
df["lag_2d"] = df["aggregate_w"].shift(96 * 2)
df["lag_3d"] = df["aggregate_w"].shift(96 * 3)
df["lag_7d"] = df["aggregate_w"].shift(96 * 7)


# Rate of change
shifted = df["aggregate_w"].shift(1)
df["diff_15m"] = shifted.diff(1)
df["diff_1h"] = shifted.diff(4)
df["diff_6h"] = shifted.diff(24)

# Time features
hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

dow = df["timestamp"].dt.dayofweek
df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

month = df["timestamp"].dt.month
df["month_sin"] = np.sin(2 * np.pi * month / 12)
df["month_cos"] = np.cos(2 * np.pi * month / 12)

df["is_weekend"] = (dow >= 5).astype(int)
df["hour_of_day"] = df["timestamp"].dt.hour


df = df.dropna().reset_index(drop=True)

# Features
features = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_weekend", "hour_of_day",
    "lag_15m", "lag_1h", "lag_3h", "lag_6h", "lag_12h", "lag_1d", "lag_2d", "lag_3d", "lag_7d",
    "diff_15m", "diff_1h", "diff_6h",
]


X = df[features]
y = df["aggregate_w"]



df = df.dropna().reset_index(drop=True)



# predict onwards from last data point
STEPS_AHEAD = 96 * 3  

buffer = df["aggregate_w"].iloc[-700:].tolist()
last_ts = df["timestamp"].iloc[-1]

future_rows = []
for i in range(STEPS_AHEAD):
    next_ts = last_ts + pd.Timedelta(minutes=15 * (i + 1))
    hour = next_ts.hour + next_ts.minute / 60.0
    dow  = next_ts.dayofweek
    row  = {
        "hour_sin": np.sin(2*np.pi*hour/24), "hour_cos": np.cos(2*np.pi*hour/24),
        "dow_sin":  np.sin(2*np.pi*dow/7),   "dow_cos":  np.cos(2*np.pi*dow/7),
        "month_sin":np.sin(2*np.pi*next_ts.month/12), "month_cos":np.cos(2*np.pi*next_ts.month/12),
        "is_weekend": int(dow >= 5), "hour_of_day": next_ts.hour,
        "lag_15m": buffer[-1],   "lag_1h":  buffer[-4],  "lag_3h":  buffer[-12],
        "lag_6h":  buffer[-24],  "lag_12h": buffer[-48], "lag_1d":  buffer[-96],
        "lag_2d":  buffer[-192], "lag_3d":  buffer[-288],"lag_7d":  buffer[-672],
        "diff_15m": buffer[-1]-buffer[-2], "diff_1h": buffer[-1]-buffer[-5], "diff_6h": buffer[-1]-buffer[-25],
    }
    
    future_rows.append({"timestamp": next_ts, **row})

    # feed prediction back into buffer so next step has correct lags
    pred_val = max(0, float(model.predict(pd.DataFrame([row])[features])[0]))
    buffer.append(pred_val)
    buffer = buffer[-700:]



#predict

future_df = pd.DataFrame(future_rows)
y_pred    = model.predict(future_df[features])






#plot
plt.figure(figsize=(12, 6))
plt.plot(df["timestamp"].iloc[-192:], df["aggregate_w"].iloc[-192:], label="Historical (last 2 days)")
plt.plot(future_df["timestamp"], y_pred, label="Forecast", color="orange")
plt.xlabel("Timestamp")
plt.ylabel("Aggregate Power (W)")
plt.title(f"House {house_to_use}")
plt.legend()
plt.tight_layout()
plt.show()



