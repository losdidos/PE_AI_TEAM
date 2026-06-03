

#comments mede mogelijk gemaakt door GPT °°'  + hulp voor demo ven claud
#                                        __

import joblib
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# INSTELLINGEN  ← pas hier aan
# ══════════════════════════════════════════════════════════════════════════════

MODEL_PATH   = "best_model_GradientBoosting.pkl"   # naam van jouw .pkl bestand
HOUSE_ID     = 6                           # welk huishouden demo-en
TARGET       = "Aggregate"
INTERVAL_SEC = 0.5                           # seconden tussen voorspellingen

# ─── STARTDATUM VOORSPELLING ─────────────────────────────────────────────────
# Alles vóór deze datum = aanloop (blauw).
# Alles vanaf deze datum = live voorspelling (oranje stippel).
# Zet op None om automatisch de testset-grens (80/20 split) te gebruiken.
START_DATUM =  "2026-8-6"         # bijv.  "2014-10-01"  of  None

# Lege ruimte rechts van de nieuwste voorspelling (96 kwartieren = 1 dag)
RECHTER_MARGE  = 96
# Max aanloop-punten links zichtbaar (672 = 1 week)
AANLOOP_PUNTEN = 672

# Open-Meteo historische data
WEATHER_START = "2014-03-17"
WEATHER_END   = "2016-01-05"
LATITUDE      = 51.5085
LONGITUDE     = -0.1257

# Alle features uit de training — EXACT dezelfde lijst en volgorde!
# De metadata-kolommen (Appliances Owned, Detached, ...) zijn cruciaal:
# het model werd ermee getraind, dus de demo moet ze ook aanleveren.
FEATURE_COLS = [
    "hour", "dayofweek", "month", "is_weekend",
    "lag1", "lag2", "lag3", "lag4", "lag8", "lag96", "lag672",
    "temperature_2m (°C)",
    "relative_humidity_2m (%)",
    "Appliances Owned", "Detached", "Semi-detached", "Mid-terrace", "Size",
]

# ══════════════════════════════════════════════════════════════════════════════
# 1. MODEL LADEN  +  feature-namen opslaan
# ══════════════════════════════════════════════════════════════════════════════

def load_model(path: str):
    p = Path(path)
    if not p.exists():
        pkls = list(Path(".").glob("best_model_*.pkl"))
        if pkls:
            p = sorted(pkls)[0]
            print(f"⚠  MODEL_PATH niet gevonden, gebruik: {p}")
        else:
            raise FileNotFoundError(
                f"Geen model gevonden op '{path}'. "
                "Zorg dat het .pkl bestand in dezelfde map staat."
            )
    model = joblib.load(p)
    print(f"✓ Model geladen: {p.name}  ({type(model).__name__})")

    # Probeer de feature-namen die het model écht kent op te halen
    feat_names = None
    try:
        feat_names = list(model.feature_names_in_)
        print(f"  Model verwacht {len(feat_names)} features: {feat_names}")
    except AttributeError:
        print("  ⚠  Model heeft geen feature_names_in_ — gebruik FEATURE_COLS als fallback.")
    return model, feat_names

# ══════════════════════════════════════════════════════════════════════════════
# 2. WEERDATA OPHALEN  (uurlijks → 15-min interpolatie)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_weather() -> pd.DataFrame:
    print("⏳ Weerdata ophalen via Open-Meteo API ...")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   LATITUDE,
        "longitude":  LONGITUDE,
        "start_date": WEATHER_START,
        "end_date":   WEATHER_END,
        "hourly":     "temperature_2m,relative_humidity_2m",
        "timezone":   "Europe/London",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()["hourly"]

    df = pd.DataFrame({
        "Time":                      pd.to_datetime(data["time"]),
        "temperature_2m (°C)":      data["temperature_2m"],
        "relative_humidity_2m (%)": data["relative_humidity_2m"],
    }).set_index("Time")

    df_15min = df.resample("15min").interpolate(method="linear")
    print(f"✓ Weerdata: {len(df_15min):,} rijen (15-min) van {WEATHER_START} t/m {WEATHER_END}")
    return df_15min

# ══════════════════════════════════════════════════════════════════════════════
# 3. HUISHOUDDATA LADEN
# ══════════════════════════════════════════════════════════════════════════════

def load_house_data(house_id: int, weather_df: pd.DataFrame) -> pd.DataFrame:
    path = f"sprint3/data/whederdataHous_{house_id}_metMetaData.csv"
    p = Path(path)
    if not p.exists():
        print(f"⚠  Huishoudendata niet gevonden ({path}). Demo draait in simulatiemodus.")
        return None

    df = pd.read_csv(path)
    df["Time"] = pd.to_datetime(df["Time"])
    df.set_index("Time", inplace=True)

    # Appliance-kolommen droppen (zelfde als in training)
    df = df.drop(columns=[f"Appliance{i}" for i in range(1, 10)], errors="ignore")

    # Weerkolommen vervangen door de vers geïnterpoleerde API-versie
    weer_cols = ["temperature_2m (°C)", "relative_humidity_2m (%)"]
    df = df.drop(columns=[c for c in weer_cols if c in df.columns], errors="ignore")
    df = df.join(weather_df[weer_cols], how="left")

    print(f"✓ Huis {house_id} geladen: {len(df):,} rijen")
    print(f"  Kolommen: {list(df.columns)}")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# 4. FEATURE ENGINEERING  — bouwt één feature-rij voor het model
# ══════════════════════════════════════════════════════════════════════════════

def build_features(row: pd.DataFrame, history: list, model_features: list) -> pd.DataFrame:
    """
    row            : één rij van house_df (met alle kolommen incl. metadata)
    history        : lijst van eerder geziene Aggregate-waarden (voor lags)
    model_features : exacte kolom-volgorde die het model verwacht
    """
    feat = {}
    now = row.index[-1]

    # Tijdsfeatures
    feat["hour"]       = now.hour
    feat["dayofweek"]  = now.dayofweek
    feat["month"]      = now.month
    feat["is_weekend"] = int(now.dayofweek >= 5)

    # Lag-features
    for lag in [1, 2, 3, 4, 8, 96, 672]:
        feat[f"lag{lag}"] = history[-lag] if len(history) >= lag else np.nan

    # Weerfeatures
    for col in ["temperature_2m (°C)", "relative_humidity_2m (%)"]:
        feat[col] = float(row[col].iloc[0]) if col in row.columns else np.nan

    # Metadata-features — dit waren de missende features!
    # Ze zijn constant per huishouden, dus gewoon de waarde uit de rij pakken.
    for col in ["Appliances Owned", "Detached", "Semi-detached", "Mid-terrace", "Size"]:
        feat[col] = float(row[col].iloc[0]) if col in row.columns else 0.0

    feat_df = pd.DataFrame([feat])

    # Zorg dat de kolom-volgorde exact overeenkomt met training
    final_cols = [c for c in model_features if c in feat_df.columns]
    missing    = [c for c in model_features if c not in feat_df.columns]
    if missing:
        print(f"  ⚠  Missende features (worden 0): {missing}")
        for c in missing:
            feat_df[c] = 0.0

    return feat_df[model_features]

# ══════════════════════════════════════════════════════════════════════════════
# 5. DEMO KLASSE
# ══════════════════════════════════════════════════════════════════════════════

class LiveDemo:
    def __init__(self, model, model_features, house_df, weather_df):
        self.model         = model
        self.model_features = model_features   # exacte feature-lijst van het model
        self.house_df      = house_df
        self.step          = 0
        self.history       = []

        # ── Split bepalen ────────────────────────────────────────────────────
        if house_df is not None:
            n = len(house_df)

            if START_DATUM is not None:
                # Gebruik de ingestelde startdatum
                start_ts = pd.Timestamp(START_DATUM)
                split = house_df.index.searchsorted(start_ts)
                if split == 0 or split >= n:
                    print(f"⚠  START_DATUM '{START_DATUM}' valt buiten de data. "
                          "Terugval naar 80/20 split.")
                    split = int(n * 0.8)
                else:
                    print(f"✓ Startdatum voorspelling: {START_DATUM}  "
                          f"(index {split:,} van {n:,})")
            else:
                split = int(n * 0.8)
                print(f"✓ Automatische 80/20 split op index {split:,}")

            self.data_slice = house_df.iloc[split:].copy()
            train_slice     = house_df.iloc[max(0, split - 700):split]
            self.history    = list(train_slice[TARGET].values)

        else:
            self.data_slice = weather_df.copy()
            self.history    = [300 + np.random.normal(0, 50) for _ in range(700)]
            split           = 0
            print("✓ Simulatiemodus actief (geen huishouddata gevonden)")

        self.max_steps = len(self.data_slice)

        # ── Aanloop: AANLOOP_PUNTEN echte datapunten vóór de split ───────────
        n_aanloop = min(AANLOOP_PUNTEN, len(self.history))
        aanloop_vals = self.history[-n_aanloop:]

        if house_df is not None:
            aanloop_start = max(0, split - n_aanloop)
            aanloop_idx   = house_df.index[aanloop_start: aanloop_start + n_aanloop]
        else:
            aanloop_idx = pd.date_range(
                end     = self.data_slice.index[0] - pd.Timedelta(minutes=15),
                periods = n_aanloop,
                freq    = "15min",
            )

        self.aanloop_times   = list(aanloop_idx)
        self.aanloop_actuals = list(aanloop_vals)

        # Live-gedeelte
        self.live_times   = []
        self.live_actuals = []
        self.live_preds   = []
        self.errors       = []

        start_str = self.aanloop_times[0].strftime('%d/%m/%Y')
        eind_str  = self.aanloop_times[-1].strftime('%d/%m/%Y')
        print(f"✓ Aanloop:  {len(self.aanloop_times):,} punten  ({start_str} – {eind_str})")
        print(f"✓ Testset:  {self.max_steps:,} datapunten beschikbaar voor demo")

    def next_step(self):
        if self.step >= self.max_steps:
            print("\n🏁 Alle testdata afgespeeld. Demo stopt.")
            return False

        row = self.data_slice.iloc[[self.step]]
        X   = build_features(row, self.history, self.model_features)

        try:
            pred = float(self.model.predict(X)[0])
        except Exception as e:
            print(f"⚠  Voorspelling mislukt op stap {self.step}: {e}")
            pred = float(np.mean(self.history[-10:])) if self.history else 300.0

        actual = (float(row[TARGET].iloc[0]) if TARGET in row.columns
                  else max(50, 300 + 100 * np.sin((row.index[0].hour - 7) * np.pi / 10)
                           + np.random.normal(0, 40)))

        self.history.append(actual)
        self.live_times.append(row.index[0])
        self.live_actuals.append(actual)
        self.live_preds.append(pred)
        self.errors.append(abs(actual - pred))
        self.step += 1
        return True

    def venster(self):
        """Geeft de zichtbare tijdreeks terug (aanloop + live + rechtermarge)."""
        n_live        = len(self.live_times)
        n_aanloop_vis = max(0, AANLOOP_PUNTEN - n_live)

        vis_times   = (self.aanloop_times[-n_aanloop_vis:] + self.live_times
                       if n_aanloop_vis > 0 else list(self.live_times))
        vis_actuals = (self.aanloop_actuals[-n_aanloop_vis:] + self.live_actuals
                       if n_aanloop_vis > 0 else list(self.live_actuals))
        vis_preds   = [np.nan] * n_aanloop_vis + list(self.live_preds)
        total_width = len(vis_times) + RECHTER_MARGE

        return vis_times, vis_actuals, vis_preds, n_aanloop_vis, total_width

    @property
    def mae(self):
        return np.mean(self.errors) if self.errors else 0.0

    @property
    def rmse(self):
        return np.sqrt(np.mean(np.array(self.errors) ** 2)) if self.errors else 0.0

# ══════════════════════════════════════════════════════════════════════════════
# 6. LIVE PLOT
# ══════════════════════════════════════════════════════════════════════════════

def run_live_demo(demo: LiveDemo):
    fig, (ax_main, ax_err) = plt.subplots(
        2, 1, figsize=(15, 8),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.patch.set_facecolor("#0f1117")
    for ax in (ax_main, ax_err):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#aaaaaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    line_actual,  = ax_main.plot([], [], lw=1.8, color="#2196F3")
    line_pred,    = ax_main.plot([], [], lw=2.0, color="#FF5722",
                                 linestyle="--", alpha=0.9)
    line_err_bar, = ax_err.plot( [], [], lw=1.4, color="#4CAF50")
    fill_ref      = [None]

    vline = ax_main.axvline(x=len(demo.aanloop_times) - 1,
                            color="#ffffff", linewidth=1.2,
                            linestyle=":", alpha=0.4, zorder=5)

    leg_hist = mpatches.Patch(color="#2196F3", label="Echt verbruik")
    leg_pred = mpatches.Patch(color="#FF5722", label="Modelvoorspelling")
    leg_fout = mpatches.Patch(color="#4CAF50", alpha=0.35, label="Foutzone")
    ax_main.legend(handles=[leg_hist, leg_pred, leg_fout],
                   loc="upper left", facecolor="#1c2128",
                   labelcolor="white", fontsize=9)

    ax_main.set_ylabel("Verbruik (Wh)", color="#aaaaaa")
    ax_main.grid(True, alpha=0.12, color="#aaaaaa")
    ax_err.set_ylabel("|Fout| (Wh)", color="#aaaaaa", fontsize=9)
    ax_err.set_xlabel("Tijd", color="#aaaaaa")
    ax_err.legend(
        handles=[mpatches.Patch(color="#4CAF50", label="|Fout| (Wh)")],
        loc="upper left", facecolor="#1c2128", labelcolor="white", fontsize=9,
    )
    ax_err.grid(True, alpha=0.12, color="#aaaaaa")

    start_lbl = (f"vanaf {START_DATUM}" if START_DATUM else "80/20 split")
    fig.suptitle(
        f"Live Energie Voorspelling — Huis {HOUSE_ID}  "
        f"(aanloop: 1 week  →  live voorspelling  [{start_lbl}])",
        color="white", fontsize=12, fontweight="bold",
    )
    stats_text = ax_main.text(
        0.99, 0.97, "Aanloop wordt getoond…",
        transform=ax_main.transAxes,
        color="#aaaaaa", fontsize=9, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1c2128", edgecolor="#30363d"),
    )

    def update_xticks(ts, total_width):
        n    = len(ts)
        step = max(1, n // 8)
        pos  = list(range(0, n, step))
        lbl  = [ts[i].strftime("%H:%M\n%d/%m") for i in pos]
        for ax in (ax_main, ax_err):
            ax.set_xlim(0, total_width)
            ax.set_xticks(pos)
            ax.set_xticklabels(lbl, fontsize=7, color="#aaaaaa")

    # Aanloop meteen tonen
    ts  = demo.aanloop_times
    act = demo.aanloop_actuals
    xs  = list(range(len(ts)))
    total_w = len(xs) + RECHTER_MARGE
    line_actual.set_data(xs, act)
    mn, mx = min(act), max(act)
    pad = max((mx - mn) * 0.1, 20)
    ax_main.set_ylim(mn - pad, mx + pad)
    ax_err.set_ylim(0, 50)
    update_xticks(ts, total_w)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.canvas.draw_idle()

    def update(_frame):
        ok = demo.next_step()
        if not ok:
            ani.event_source.stop()
            return line_actual, line_pred, line_err_bar, stats_text

        vis_times, vis_act, vis_preds, n_aanloop_vis, total_width = demo.venster()
        xs = list(range(len(vis_times)))

        vline.set_xdata([n_aanloop_vis - 1, n_aanloop_vis - 1])
        line_actual.set_data(xs, vis_act)

        xs_pred  = [x for x, v in zip(xs, vis_preds) if not np.isnan(v)]
        val_pred = [v for v in vis_preds if not np.isnan(v)]
        line_pred.set_data(xs_pred, val_pred)

        if fill_ref[0]:
            try:
                fill_ref[0].remove()
            except Exception:
                pass
        if xs_pred and val_pred:
            act_live = vis_act[n_aanloop_vis:]
            fill_ref[0] = ax_main.fill_between(
                xs_pred, act_live, val_pred, alpha=0.13, color="#4CAF50",
            )

        err_xs = list(range(n_aanloop_vis, n_aanloop_vis + len(demo.errors)))
        line_err_bar.set_data(err_xs, demo.errors)

        all_y = vis_act + val_pred
        if all_y:
            mn, mx = min(all_y), max(all_y)
            pad = max((mx - mn) * 0.1, 20)
            ax_main.set_ylim(mn - pad, mx + pad)
        if demo.errors:
            ax_err.set_ylim(0, max(demo.errors) * 1.2 + 10)

        update_xticks(vis_times, total_width)

        if val_pred:
            stats_text.set_text(
                f"Stap: {demo.step}  |  "
                f"Echt: {vis_act[-1]:.0f} Wh  |  "
                f"Pred: {val_pred[-1]:.0f} Wh  |  "
                f"MAE: {demo.mae:.1f} Wh  |  "
                f"RMSE: {demo.rmse:.1f} Wh"
            )
        return line_actual, line_pred, line_err_bar, stats_text

    def init():
        line_actual.set_data([], [])
        line_pred.set_data([], [])
        line_err_bar.set_data([], [])
        return line_actual, line_pred, line_err_bar

    ani = animation.FuncAnimation(
        fig, update, init_func=init,
        interval=INTERVAL_SEC * 1000,
        cache_frame_data=False, blit=False, save_count=9999,
    )

    print(f"\n🚀 Demo gestart!")
    print(f"   Aanloop-data zichtbaar (1 week vóór de startdatum).")
    print(f"   Daarna elke {INTERVAL_SEC} seconden een nieuwe voorspelling.")
    print("   Sluit het venster om te stoppen.\n")
    plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  LIVE ENERGIE VOORSPELLING DEMO")
    print("=" * 60)

    model, model_features = load_model(MODEL_PATH)

    # Als het model geen feature_names_in_ heeft: gebruik FEATURE_COLS als fallback
    if model_features is None:
        model_features = FEATURE_COLS
        print(f"  Fallback: gebruik FEATURE_COLS ({len(model_features)} features)")

    weather_df = fetch_weather()
    house_df   = load_house_data(HOUSE_ID, weather_df)

    demo = LiveDemo(model, model_features, house_df, weather_df)
    run_live_demo(demo)