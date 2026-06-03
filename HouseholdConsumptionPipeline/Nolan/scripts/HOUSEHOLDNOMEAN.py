import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from joblib import dump, load


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data_source"
TEST_DIR = BASE_DIR / "test_sets"





results = []
for csv_path in sorted(DATA_DIR.glob("CLEAN_House*_15min_interpolated_with_metadata.csv")):
    house_id = int(csv_path.name.split("CLEAN_House")[1].split("_")[0])
    test_path = TEST_DIR / f"house_{house_id:02d}_test_predictions_clean_block.csv"
    
    if not test_path.exists():
        print(f"House {house_id:02d} - SKIP (no test block)")
        continue
    
    # Load data
    df = pd.read_csv(csv_path)
    test_block = pd.read_csv(test_path)
    
    # Prep timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.dropna(subset=["timestamp", "aggregate_w"]).sort_values("timestamp").reset_index(drop=True)
    
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
    
    # Metadata
    metadata_cols = ["occupancy", "appliances_owned", "issues_any"]
    for col in metadata_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
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
    
    # Split
    test_block["timestamp"] = pd.to_datetime(test_block["timestamp"])
    test_start = test_block["timestamp"].min()
    test_end = test_block["timestamp"].max()
    
    train_mask = df["timestamp"] < test_start
    test_mask = (df["timestamp"] >= test_start) & (df["timestamp"] <= test_end)
    
    X_pre, y_pre = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    dates_test = df[test_mask]["timestamp"]
    
    train_end = int(len(X_pre) * 0.85)
    X_train, y_train = X_pre.iloc[:train_end], y_pre.iloc[:train_end]
    X_val, y_val = X_pre.iloc[train_end:], y_pre.iloc[train_end:]
    
    # Train
    model = XGBRegressor(
        n_estimators=3000,
        learning_rate=0.01,
        max_depth=8,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=3,
        gamma=0.05,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror",
        eval_metric="mae",
        early_stopping_rounds=100,
    )
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # Evaluate
    y_val_pred = model.predict(X_val)
    y_pred = model.predict(X_test)
    
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    
    print(f"\nHouse {house_id:02d}: Val MAE={val_mae:.2f} R2={val_r2:.4f} | Test MAE={test_mae:.2f} R2={test_r2:.4f}")
    
    results.append({
        "house": house_id,
        "val_mae": val_mae,
        "val_r2": val_r2,
        "test_mae": test_mae,
        "test_r2": test_r2,
    })
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.scatter(y_test, y_pred, alpha=0.3, s=5)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    ax1.set_xlabel("Actual (W)")
    ax1.set_ylabel("Predicted (W)")
    ax1.set_title(f"House {house_id:02d} - MAE: {test_mae:.2f}, R2: {test_r2:.4f}")
    
    ax2.plot(dates_test.values, y_test.values, label="Actual", linewidth=1)
    ax2.plot(dates_test.values, y_pred, label="Predicted", linewidth=1, alpha=0.7)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Power (W)")
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    # Save the model


    dump(model,f"C:\\Users\\nolan\\OneDrive\\Documents\\GitHub\\PE2-AI\\HOUSEHOLDPIPELINE\\models_no_mean_meta\\xgboost_model_house_{house_id:02d}.joblib")
