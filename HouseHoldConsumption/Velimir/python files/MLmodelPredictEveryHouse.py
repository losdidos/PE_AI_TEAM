import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import glob
import os

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit

# CONFIG ─
DATA_DIR = "./Sprint 4/household-data-appliances-interpolated/"
FILE_PATTERN = "CleerResampel_House_*.csv"
PLOT_DIR = "./Sprint 4/pngs/"
os.makedirs(PLOT_DIR, exist_ok=True)

# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────
def create_features(df):
    """Create time-based and lag features."""
    df = df.copy()
    df["quarter"]   = df.index.minute // 15
    df["hour"]      = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["isweekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["month"]     = df.index.month

    lag_steps = [1, 4, 96, 672]   # 15 min, 1 hr, 1 day, 1 week
    for lag in lag_steps:
        df[f"lag_{lag}"] = df["Aggregate"].shift(lag)

    return df


# ── PER-HOUSE PIPELINE ────────────────────────────────────────────────────────
def run_pipeline(csv_path):
    house_name = os.path.splitext(os.path.basename(csv_path))[0]
    print(f"\n{'='*60}")
    print(f"  Processing: {house_name}")
    print(f"{'='*60}")

    # Load & clean
    df = pd.read_csv(csv_path, delimiter=",")
    df["Time"] = pd.to_datetime(df["Time"], format="%Y-%m-%d %H:%M:%S")
    df = df.sort_values("Time").set_index("Time")
    df = df.drop(columns=["Unix"], errors="ignore")

    df = create_features(df)
    df = df.dropna().copy()

    FEATURES = [c for c in df.columns if c != "Aggregate"]
    TARGET   = "Aggregate"

    tss = TimeSeriesSplit(n_splits=5, test_size=96 * 7, gap=96)

    fold_results = []
    last_y_test  = None
    last_preds   = None

    for fold, (train_idx, test_idx) in enumerate(tss.split(df), start=1):
        train, test = df.iloc[train_idx], df.iloc[test_idx]

        X_train, y_train = train[FEATURES], train[TARGET]
        X_test,  y_test  = test[FEATURES],  test[TARGET]

        model = XGBRegressor(
            base_score=0.5,
            n_estimators=1000,
            early_stopping_rounds=50,
            random_state=42,
            learning_rate=0.10,
            objective="reg:squarederror",
            max_depth=6,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        predictions = model.predict(X_test)

        mae  = mean_absolute_error(y_test, predictions)
        r2   = r2_score(y_test, predictions)
        mape = mean_absolute_percentage_error(y_test, predictions)

        print(f"  Fold {fold}: MAE={mae:.3f}  R²={r2:.3f}  MAPE={mape:.3f}")
        fold_results.append({"fold": fold, "mae": mae, "r2": r2, "mape": mape})

        last_y_test = y_test
        last_preds  = predictions

    results_df = pd.DataFrame(fold_results)
    print(f"\n  Averages → MAE={results_df['mae'].mean():.3f}  "
          f"R²={results_df['r2'].mean():.3f}  MAPE={results_df['mape'].mean():.3f}")

    # ── PLOTS (last fold) ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(house_name, fontsize=14, fontweight="bold")

    # Actual vs Predicted
    axes[0].plot(last_y_test.index, last_y_test.values, label="Actual",    linewidth=1)
    axes[0].plot(last_y_test.index, last_preds,          label="Predicted", linewidth=1, alpha=0.8)
    axes[0].set_title("Actual vs Predicted (last fold)")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Aggregate")
    axes[0].legend()
    axes[0].tick_params(axis="x", rotation=30)

    # QQ plot
    errors = last_y_test.values - last_preds
    stats.probplot(errors, dist="norm", plot=axes[1])
    axes[1].set_title("QQ Plot of Residuals")

    plt.tight_layout()
    plot_path = os.path.join(PLOT_DIR, f"{house_name}.png")
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"  Plot saved → {plot_path}")

    results_df["house"] = house_name
    return results_df


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, FILE_PATTERN)))

    if not csv_files:
        raise FileNotFoundError(f"No files found matching {DATA_DIR}{FILE_PATTERN}")

    print(f"Found {len(csv_files)} house file(s):")
    for f in csv_files:
        print(f"  {os.path.basename(f)}")

    all_results = []
    for csv_path in csv_files:
        house_results = run_pipeline(csv_path)
        all_results.append(house_results)

    # ── SUMMARY TABLE ─────────────────────────────────────────────────────────
    summary = (
        pd.concat(all_results)
        .groupby("house")[["mae", "r2", "mape"]]
        .mean()
        .round(4)
        .sort_index()
    )

    print("\n" + "="*60)
    print("  SUMMARY — average metrics per house")
    print("="*60)
    print(summary.to_string())
    print("="*60)