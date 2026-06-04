

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import os

# GPT = coments

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA INLADEN & SAMENVOEGEN
# ══════════════════════════════════════════════════════════════════════════════

DATA_DIR = "sprint3/data"
HOUSE_IDS = [1, 2, 3, 5, 6, 7, 8, 12,15,16,18,19]   # moet aanpassen mij meer Datasets
TARGET = "Aggregate"

def load_house(house_id: int) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"whederdataHous_{house_id}_metMetaData.csv")
    df = pd.read_csv(path)
    df["Time"] = pd.to_datetime(df["Time"])
    df.set_index("Time", inplace=True)

    # Appliance kolommen droppen
    df = df.drop(columns=[f"Appliance{i}" for i in range(1, 10)], errors="ignore")

    # Huishouden-ID toevoegen zodat je het na merge nog weet
    df["house_id"] = house_id
    return df


all_frames = []
for hid in HOUSE_IDS:
    try:
        all_frames.append(load_house(hid))
        print(f"✓ Huis {hid} geladen")
    except FileNotFoundError:
        print(f"✗ Huis {hid} niet gevonden — overgeslagen")

combined = pd.concat(all_frames, axis=0).sort_index()
print(f"\nGecombineerde dataset: {len(combined):,} rijen, {combined['house_id'].nunique()} huishoudens\n")

# Gecombineerde CSV opslaan (handig voor later gebruik)
combined.reset_index().to_csv("combined_huishoudens.csv", index=False)
print("✓ combined_huishoudens.csv opgeslagen\n")


# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING  (wordt per huishouden gedaan om lags correct te houden)
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    "hour", "dayofweek", "month", "is_weekend",
    "lag1", "lag2", "lag3", "lag4", "lag8", "lag96", "lag672",
    "temperature_2m (°C)",
    "relative_humidity_2m (%)",
    "Appliances Owned", "Detached", "Semi-detached", "Mid-terrace", "Size",
]

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"]       = df.index.hour
    df["dayofweek"]  = df.index.dayofweek
    df["month"]      = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    for lag in [1, 2, 3, 4, 8, 96, 672]:
        df[f"lag{lag}"] = df[TARGET].shift(lag)

    # Alleen feature-kolommen die aanwezig zijn behouden
    available = [c for c in FEATURE_COLS if c in df.columns]
    df = df[available + [TARGET, "house_id"]].dropna()
    return df, available


# Feature engineering per huishouden (lags mogen niet over huishoudens heen lopen)
house_frames = {}
for hid in combined["house_id"].unique():
    subset = combined[combined["house_id"] == hid].copy()
    feat_df, avail_cols = add_features(subset)
    house_frames[hid] = {"df": feat_df, "features": avail_cols}

# Alles samenvoegen voor gezamenlijke train
all_feat = pd.concat([v["df"] for v in house_frames.values()]).sort_index()
features_used = house_frames[list(house_frames.keys())[0]]["features"]

X_all = all_feat[features_used]
y_all = all_feat[TARGET]

# ===============================================================================
# 3. TRAIN / TEST SPLIT  (80 / 20 op tijdsvolgorde)
# ===============================================================================

split_idx = int(len(all_feat) * 0.8)
X_train, X_test = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
y_train, y_test = y_all.iloc[:split_idx], y_all.iloc[split_idx:]
meta_test = all_feat[["house_id"]].iloc[split_idx:]

print(f"Trainset: {len(X_train):,} rijen | Testset: {len(X_test):,} rijen\n")

# ══════════════════════════════════════════════════════════════════════════════
# 4. GRIDSEARCH OVER MEERDERE MODELLEN
# ══════════════════════════════════════════════════════════════════════════════

tscv = TimeSeriesSplit(n_splits=5)

MODELS = {
    "XGBoost": {
        "model": XGBRegressor(tree_method="hist", random_state=42),
        "params": {
            "n_estimators":  [100, 300],
            "learning_rate": [0.05, 0.1],
            "max_depth":     [3, 6],
            "subsample":     [0.8, 1.0],
        },
    },
    "LightGBM": {
        "model": LGBMRegressor(random_state=42, verbose=-1),
        "params": {
            "n_estimators":  [100, 300],
            "learning_rate": [0.05, 0.1],
            "max_depth":     [-1, 6],
        },
    },
    "RandomForest": {
        "model": RandomForestRegressor(random_state=42, n_jobs=-1),
        "params": {
            "n_estimators": [100, 300],
            "max_depth":    [None, 10],
        },
    },
    "GradientBoosting": {
        "model": GradientBoostingRegressor(random_state=42),
        "params": {
            "n_estimators":  [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth":     [3, 5],
        },
    },
}

best_models = {}
cv_scores   = {}

for name, mp in MODELS.items():
    print(f"⏳ GridSearch: {name} ...")
    grid = GridSearchCV(
        estimator=mp["model"],
        param_grid=mp["params"],
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)
    best_models[name] = grid.best_estimator_
    cv_scores[name]   = -grid.best_score_
    print(f"   Beste params : {grid.best_params_}")
    print(f"   CV RMSE      : {cv_scores[name]:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. EVALUATIE OP TESTSET  (globaal + per huishouden)
# ══════════════════════════════════════════════════════════════════════════════

print("\n── Testset RMSE per model ──")
test_scores = {}
test_preds  = {}

for name, model in best_models.items():
    y_pred_all = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_all))
    test_scores[name] = rmse
    test_preds[name]  = pd.Series(y_pred_all, index=y_test.index)
    print(f"  {name:20s}  RMSE = {rmse:.4f}")

best_name = min(test_scores, key=test_scores.get)
print(f"\n🏆 Beste model: {best_name}  (RMSE = {test_scores[best_name]:.4f})\n")

# RMSE per huishouden voor het beste model
y_pred_best = test_preds[best_name]
house_rmse  = {}

print(f"── RMSE per huishouden ({best_name}) ──")
for hid in sorted(meta_test["house_id"].unique()):
    mask = meta_test["house_id"] == hid
    if mask.sum() == 0:
        continue
    rmse_h = np.sqrt(mean_squared_error(y_test[mask], y_pred_best[mask]))
    house_rmse[hid] = rmse_h
    print(f"  Huis {hid}: RMSE = {rmse_h:.4f}  (n={mask.sum():,})")

# ══════════════════════════════════════════════════════════════════════════════
# 6. SUBPLOTS PER HUISHOUDEN
# ══════════════════════════════════════════════════════════════════════════════

n_houses  = len(house_rmse)
n_cols    = 2
n_rows    = (n_houses + n_cols - 1) // n_cols   # ceil

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(16, 5 * n_rows),
    squeeze=False,
)
fig.suptitle(
    f"Echt vs Voorspelling per huishouden — {best_name}  "
    f"(globale RMSE = {test_scores[best_name]:.4f})",
    fontsize=14, fontweight="bold", y=1.01,
)

PLOT_DAYS = 7   # hoeveel dagen zichtbaar in tijdlijn-plot (anders te druk)

for ax_idx, hid in enumerate(sorted(house_rmse.keys())):
    row, col = divmod(ax_idx, n_cols)
    ax = axes[row][col]

    mask      = meta_test["house_id"] == hid
    y_true_h  = y_test[mask]
    y_pred_h  = y_pred_best[mask]
    rmse_h    = house_rmse[hid]

    # Laatste PLOT_DAYS dagen voor leesbaarheid
    cutoff    = y_true_h.index[-1] - pd.Timedelta(days=PLOT_DAYS)
    zoom_mask = y_true_h.index >= cutoff

    ax.plot(y_true_h.index[zoom_mask], y_true_h.values[zoom_mask],
            label="Echt", linewidth=1.2, color="#2196F3")
    ax.plot(y_pred_h.index[zoom_mask], y_pred_h.values[zoom_mask],
            label="Voorspelling", linewidth=1.2, linestyle="--", color="#FF5722", alpha=0.85)

    ax.set_title(f"Huis {hid}  —  RMSE = {rmse_h:.4f}", fontsize=11)
    ax.set_xlabel("Tijd")
    ax.set_ylabel("Verbruik (Wh)")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, alpha=0.3)

# Lege subplots verbergen als aantal huishoudens oneven is
for ax_idx in range(n_houses, n_rows * n_cols):
    row, col = divmod(ax_idx, n_cols)
    axes[row][col].set_visible(False)

plt.tight_layout()
output_plot = "forecast_per_huishouden.png"
plt.savefig(output_plot, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n✓ Plot opgeslagen als '{output_plot}'")

# ══════════════════════════════════════════════════════════════════════════════
# 7. OPTIONEEL: VERGELIJKINGSTABEL ALLE MODELLEN
# ══════════════════════════════════════════════════════════════════════════════

summary = pd.DataFrame({
    "Model":      list(test_scores.keys()),
    "CV RMSE":    [cv_scores[m]   for m in test_scores],
    "Test RMSE":  [test_scores[m] for m in test_scores],
}).sort_values("Test RMSE").reset_index(drop=True)

print("\n── Samenvatting alle modellen ──")
print(summary.to_string(index=False))
summary.to_csv("model_vergelijking.csv", index=False)
print("✓ model_vergelijking.csv opgeslagen")