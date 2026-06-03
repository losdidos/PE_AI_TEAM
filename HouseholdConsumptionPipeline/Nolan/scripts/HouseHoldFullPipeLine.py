import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import load

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE_DIR / "data_source"
MODELS_DIR = BASE_DIR / "models"



# input for new house
inp = [3, 2005, 30, 1, 3]   
STEPS_AHEAD = 96 * 7        

# Load CSVs 
house_data = {}
for csv_path in DATA_DIR.glob("CLEAN_House*_15min_interpolated_with_metadata.csv"):
    hid = int(csv_path.name.split("CLEAN_House")[1].split("_")[0])
    df_h = pd.read_csv(csv_path)
    df_h["timestamp"] = pd.to_datetime(df_h["timestamp"])
    house_data[hid] = df_h



# Load models
knn_model = load(r"C:\Users\nolan\OneDrive\Documents\GitHub\PE2-AI\HOUSEHOLDPIPELINE\Top7_XGBoost_Active\models\knn_classifier.joblib")
xgb6  = load(r"C:\Users\nolan\OneDrive\Documents\GitHub\PE2-AI\HOUSEHOLDPIPELINE\Top7_XGBoost_Active\models\xgboost_model_house_06.joblib")
xgb15 = load(r"C:\Users\nolan\OneDrive\Documents\GitHub\PE2-AI\HOUSEHOLDPIPELINE\Top7_XGBoost_Active\models\xgboost_model_house_15.joblib")
xgb18 = load(r"C:\Users\nolan\OneDrive\Documents\GitHub\PE2-AI\HOUSEHOLDPIPELINE\Top7_XGBoost_Active\models\xgboost_model_house_18.joblib")
xgb19 = load(r"C:\Users\nolan\OneDrive\Documents\GitHub\PE2-AI\HOUSEHOLDPIPELINE\Top7_XGBoost_Active\models\xgboost_model_house_19.joblib")
xgb20 = load(r"C:\Users\nolan\OneDrive\Documents\GitHub\PE2-AI\HOUSEHOLDPIPELINE\Top7_XGBoost_Active\models\xgboost_model_house_20.joblib")

# KNN pick
house_to_use = int(knn_model.predict([inp])[0])
print(house_to_use)



# Select xgboost model
xgb_models = {6: xgb6, 15: xgb15, 18: xgb18, 19: xgb19, 20: xgb20}
model = xgb_models[house_to_use]




# Get matched house data

df = house_data[house_to_use].dropna(subset=["aggregate_w"]).reset_index(drop=True)

# grab metadata constants from last row

METADATA_COLS = ["occupancy", "appliances_owned", "issues_any"]
metadata_vals = {}
for col in METADATA_COLS:
    if col in df.columns:
        metadata_vals[col] = pd.to_numeric(df[col], errors="coerce").iloc[-1]

#build_features constructs one feature row from the rolling buffer
def build_features(buf, ts):
    v = buf
    hour = ts.hour + ts.minute / 60.0
    dow  = ts.dayofweek
    row = {
        "hour_sin":    np.sin(2 * np.pi * hour / 24),
        "hour_cos":    np.cos(2 * np.pi * hour / 24),
        "dow_sin":     np.sin(2 * np.pi * dow / 7),
        "dow_cos":     np.cos(2 * np.pi * dow / 7),
        "month_sin":   np.sin(2 * np.pi * ts.month / 12),
        "month_cos":   np.cos(2 * np.pi * ts.month / 12),
        "is_weekend":  int(dow >= 5),
        "hour_of_day": ts.hour,
        "lag_15m":  v[-1],   "lag_1h":  v[-4],  "lag_3h":  v[-12],
        "lag_6h":   v[-24],  "lag_12h": v[-48], "lag_1d":  v[-96],
        "lag_2d":   v[-192], "lag_3d":  v[-288],"lag_7d":  v[-672],
        "diff_15m": v[-1] - v[-2],
        "diff_1h":  v[-1] - v[-5],
        "diff_6h":  v[-1] - v[-25],
    }
    for w in [4, 12, 24, 96]:
        window = v[-w:]
        row[f"roll_{w}_mean"] = np.mean(window)
        row[f"roll_{w}_std"]  = np.std(window)
        row[f"roll_{w}_max"]  = np.max(window)
        row[f"roll_{w}_min"]  = np.min(window)
    for col, val in metadata_vals.items():
        row[col] = val
    return pd.DataFrame([row])

# forecast loop
buffer = df["aggregate_w"].iloc[-700:].tolist()
last_ts = df["timestamp"].iloc[-1]

timestamps, predictions = [], []
for i in range(STEPS_AHEAD):
    next_ts  = last_ts + pd.Timedelta(minutes=15 * (i + 1))
    next_val = max(0, float(model.predict(build_features(buffer, next_ts))[0]))
    buffer.append(next_val)
    buffer = buffer[-700:]
    timestamps.append(next_ts)
    predictions.append(next_val)










# Plots
context_ts   = df["timestamp"].iloc[-192:].values
context_vals = df["aggregate_w"].iloc[-192:].values

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))


ax1.plot(context_ts, context_vals, label="Historical (last 2 days)", linewidth=1)
ax1.plot(timestamps, predictions, label=f"Forecast — House {house_to_use}", linewidth=1, linestyle="--", color="orange")
ax1.set_ylabel("Power (W)")
ax1.set_title(f"House {house_to_use} — 1-week forecast")
ax1.legend()


ax2.plot(timestamps, predictions, color="orange", linewidth=1)
ax2.set_ylabel("Power (W)")
ax2.set_xlabel("Time")
ax2.set_title("Forecast only")

plt.tight_layout()
plt.show()

