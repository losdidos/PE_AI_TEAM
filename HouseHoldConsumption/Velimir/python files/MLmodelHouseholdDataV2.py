import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit


# 1. LOAD & CLEAN

df = pd.read_csv(
    "./Sprint 4/household-data-appliances-interpolated/CleerResampel_House_7.csv",
    delimiter=","
)
df["Time"] = pd.to_datetime(df["Time"], format="%Y-%m-%d %H:%M:%S")
df = df.sort_values("Time").set_index("Time")

# Drop Unix and zero-variance appliance columns (Appliance7/8/9 are all 0)
APPLIANCE_COLS = [c for c in df.columns if c.startswith("Appliance")]
df = df.drop(columns=["Unix"] + APPLIANCE_COLS)



print(f"Retained appliance columns: {APPLIANCE_COLS}")
print(f"Data shape: {df.shape}  |  {df.index.min()} -> {df.index.max()}\n")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def create_features(df: pd.DataFrame, appliance_cols: list) -> pd.DataFrame:
    df = df.copy()

    # --- Time features (cyclical encoding preserves periodicity) ---
    df["quarter"]     = df.index.minute // 15
    df["hour"]        = df.index.hour
    df["dayofweek"]   = df.index.dayofweek
    df["month"]       = df.index.month
    df["isweekend"]   = df["dayofweek"].isin([5, 6]).astype(int)

    # Cyclical encoding: hour, day-of-week, month
    df["hour_sin"]    = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]     = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"]     = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"]   = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["month"] / 12)

    # Peak-period flags (from hourly average analysis)
    df["morning_peak"] = df["hour"].between(6, 9).astype(int)
    df["evening_peak"] = df["hour"].between(17, 21).astype(int)


    # --- Lag features for Aggregate ---
    # 15 min, 1 h, 2 h, 1 day, 2 days, 1 week
    for lag in [1, 4, 8, 96, 192, 672]:
        df[f"lag_{lag}"] = df["Aggregate"].shift(lag)

    # --- Rolling statistics (shift by 1 to avoid leakage) ---
    agg_shifted = df["Aggregate"].shift(1)
    df["rolling_mean_4"]   = agg_shifted.rolling(4).mean()     # 1-hour window
    df["rolling_std_4"]    = agg_shifted.rolling(4).std()
    df["rolling_mean_96"]  = agg_shifted.rolling(96).mean()    # 1-day window
    df["rolling_std_96"]   = agg_shifted.rolling(96).std()
    df["rolling_mean_672"] = agg_shifted.rolling(672).mean()   # 1-week window



    return df


df = create_features(df, APPLIANCE_COLS)
df = df.dropna().copy()

FEATURES = [c for c in df.columns if c != "Aggregate"]
TARGET    = "Aggregate"

print(f"Total features: {len(FEATURES)}")
print(f"Rows after dropna: {len(df)}\n")

# 3. LOG-TRANSFORM TARGET

def transform_target(y):
    return np.log1p(y)

def inverse_transform_target(y):
    return np.expm1(y)

    
# 4. TIME-SERIES CROSS-VALIDATION

tss = TimeSeriesSplit(n_splits=5, test_size=96 * 7, gap=96)

results     = []
all_y_test  = []
all_y_pred  = []
fold        = 0

for train_index, test_index in tss.split(df):
    train = df.iloc[train_index]
    test  = df.iloc[test_index]

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test,  y_test  = test[FEATURES],  test[TARGET]

    y_train_log = transform_target(y_train)
    y_test_log  = transform_target(y_test)

    model = XGBRegressor(
        n_estimators          = 1000,
        early_stopping_rounds = 50,
        learning_rate         = 0.05,       # lower LR → more trees, better generalisation
        max_depth             = 5,          # shallower → less overfitting
        subsample             = 0.8,        # row subsampling
        colsample_bytree      = 0.8,        # feature subsampling per tree
        min_child_weight      = 5,          # conservative splits on noisy data
        reg_alpha             = 0.1,        # L1 regularisation
        reg_lambda            = 1.0,        # L2 regularisation
        random_state          = 42,
        objective             = "reg:squarederror",
    )

    model.fit(
        X_train, y_train_log,
        eval_set=[(X_test, y_test_log)],
        verbose=False,
    )

    predictions_log = model.predict(X_test)
    predictions     = inverse_transform_target(predictions_log)
    predictions     = np.maximum(predictions, 0)   # clip negatives

    mae  = mean_absolute_error(y_test, predictions)
    r2   = r2_score(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)
    rmse = np.sqrt(np.mean((y_test - predictions) ** 2))

    fold += 1
    print(f"Fold {fold}: MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}  MAPE={mape:.3f}")

    results.append({"fold": fold, "mae": mae, "rmse": rmse, "r2": r2, "mape": mape})
    all_y_test.extend(y_test.values)
    all_y_pred.extend(predictions)

results_df = pd.DataFrame(results)

print("\n── Results by fold ──")
print(results_df.to_string(index=False))

print("\n── Average across folds ──")
print(results_df.mean(numeric_only=True).to_string())


#Features importance
importance = pd.Series(model.feature_importances_, index=FEATURES)
top20 = importance.nlargest(20)
# 6 Plots
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("XGBoost Household Electricity — Optimised Model", fontsize=14, fontweight="bold")

# 6a. Actual vs Predicted (last fold, last 2 weeks for readability)
ax = axes[0, 0]
n_plot = 96 * 14
y_test_plot  = test[TARGET].values[-n_plot:]
pred_plot    = predictions[-n_plot:]
time_idx     = test.index[-n_plot:]
ax.plot(time_idx, y_test_plot,  label="Actual",    alpha=0.8, linewidth=0.8)
ax.plot(time_idx, pred_plot,    label="Predicted", alpha=0.8, linewidth=0.8, linestyle="--")
ax.set_title("Actual vs Predicted (last 2 weeks, fold 5)")
ax.set_xlabel("Time")
ax.set_ylabel("Aggregate (W)")
ax.legend()
ax.tick_params(axis="x", rotation=30)

# 6b. Scatter: actual vs predicted (all folds)
ax = axes[0, 1]
ax.scatter(all_y_test, all_y_pred, alpha=0.05, s=5, color="steelblue")
lim = max(max(all_y_test), max(all_y_pred))
ax.plot([0, lim], [0, lim], "r--", linewidth=1, label="Perfect fit")
ax.set_title("Actual vs Predicted — All Folds")
ax.set_xlabel("Actual (W)")
ax.set_ylabel("Predicted (W)")
ax.legend()

# 6c. Top-20 feature importances
ax = axes[1, 0]
top20.sort_values().plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Top 20 Feature Importances (fold 5)")
ax.set_xlabel("Importance")

# 6d. QQ plot of residuals 
ax = axes[1, 1]
errors = np.array(all_y_test) - np.array(all_y_pred)
stats.probplot(errors, dist="norm", plot=ax)
ax.set_title("QQ Plot of Residuals — All Folds")

plt.tight_layout()
plt.savefig("xgboost_optimized_results.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nPlot saved to xgboost_optimized_results.png")