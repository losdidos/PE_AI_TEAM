import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, r2_score
from joblib import load
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data_source"
TEST_DIR = BASE_DIR / "test_sets"
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models_no_mean_meta"
OUTPUT_DIR = BASE_DIR / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_weekend", "hour_of_day",
    "lag_15m", "lag_1h", "lag_3h", "lag_6h", "lag_12h", "lag_1d", "lag_2d", "lag_3d", "lag_7d",
    "diff_15m", "diff_1h", "diff_6h",
]

HOUSES = [6, 15, 18, 19, 20]

for house_id in HOUSES:
    # Load model
    model_path = MODELS_DIR / f"xgboost_model_house_{house_id:02d}.joblib"
    model = load(model_path)

    # Load full house data and engineer features (same as training)
    csv_path = DATA_DIR / f"CLEAN_House{house_id}_15min_interpolated_with_metadata.csv"
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.dropna(subset=["timestamp", "aggregate_w"]).sort_values("timestamp").reset_index(drop=True)

    df["lag_15m"] = df["aggregate_w"].shift(1)
    df["lag_1h"]  = df["aggregate_w"].shift(4)
    df["lag_3h"]  = df["aggregate_w"].shift(12)
    df["lag_6h"]  = df["aggregate_w"].shift(24)
    df["lag_12h"] = df["aggregate_w"].shift(48)
    df["lag_1d"]  = df["aggregate_w"].shift(96)
    df["lag_2d"]  = df["aggregate_w"].shift(96 * 2)
    df["lag_3d"]  = df["aggregate_w"].shift(96 * 3)
    df["lag_7d"]  = df["aggregate_w"].shift(96 * 7)

    shifted = df["aggregate_w"].shift(1)
    df["diff_15m"] = shifted.diff(1)
    df["diff_1h"]  = shifted.diff(4)
    df["diff_6h"]  = shifted.diff(24)

    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    dow = df["timestamp"].dt.dayofweek
    df["dow_sin"]  = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * dow / 7)
    month = df["timestamp"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["is_weekend"]  = (dow >= 5).astype(int)
    df["hour_of_day"] = df["timestamp"].dt.hour

    df = df.dropna().reset_index(drop=True)

    # Restrict to test block timestamps
    test_block = pd.read_csv(TEST_DIR / f"house_{house_id:02d}_test_predictions_clean_block.csv",
                             parse_dates=["timestamp"])
    test_start = test_block["timestamp"].min()
    test_end   = test_block["timestamp"].max()
    test_mask  = (df["timestamp"] >= test_start) & (df["timestamp"] <= test_end)

    X_test     = df.loc[test_mask, FEATURES]
    y_test     = df.loc[test_mask, "aggregate_w"]
    dates_test = df.loc[test_mask, "timestamp"]

    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)

    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    dates_arr = dates_test.values
    actual_arr = y_test.values

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(f'House {house_id:02d} — XGBoost (models_no_mean_meta)\nMAE: {mae:.1f} W   |   R²: {r2:.4f}',
                 fontsize=14, fontweight='bold')

    # --- Left: zoomed 3-day sample ---
    ax1 = axes[0]
    ax1.plot(dates_arr[:288], actual_arr[:288], label='Actual', color='steelblue', linewidth=1.2)
    ax1.plot(dates_arr[:288], y_pred[:288], label='Predicted', color='tomato', linewidth=1.2, linestyle='--')
    ax1.set_title('First 3 Days of Test Period')
    ax1.set_ylabel('Power (W)')
    ax1.set_xlabel('Timestamp')
    ax1.legend(loc='upper right')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d %H:%M'))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')
    ax1.grid(True, alpha=0.3)

    # --- Right: Predicted vs Actual scatter with 45° perfect-fit line ---
    ax2 = axes[1]
    ax2.scatter(actual_arr, y_pred, alpha=0.2, s=6, color='steelblue', label='Predictions')
    lims = [min(actual_arr.min(), y_pred.min()), max(actual_arr.max(), y_pred.max())]
    ax2.plot(lims, lims, color='tomato', linewidth=1.5, linestyle='--', label='Perfect fit (45°)')
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_xlabel('Actual (W)')
    ax2.set_ylabel('Predicted (W)')
    ax2.set_title('Predicted vs Actual')
    ax2.legend(loc='upper left')
    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUTPUT_DIR / f'house_{house_id:02d}_xgboost_forecast.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'House {house_id:02d} — MAE: {mae:.1f} W, R²: {r2:.4f}  → saved to {out_path}')

print('\nAll plots saved.')
