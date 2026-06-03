import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras


def make_time_features(dates):
    dates = pd.to_datetime(dates)
    hour = dates.dt.hour + dates.dt.minute / 60
    day_of_week = dates.dt.dayofweek
    month = dates.dt.month
    day_of_year = dates.dt.dayofyear

    return pd.DataFrame({
        'hour_sin':   np.sin(2 * np.pi * hour / 24),
        'hour_cos':   np.cos(2 * np.pi * hour / 24),
        'dow_sin':    np.sin(2 * np.pi * day_of_week / 7),
        'dow_cos':    np.cos(2 * np.pi * day_of_week / 7),
        'month_sin':  np.sin(2 * np.pi * month / 12),
        'month_cos':  np.cos(2 * np.pi * month / 12),
        'doy_sin':    np.sin(2 * np.pi * day_of_year / 366),
        'doy_cos':    np.cos(2 * np.pi * day_of_year / 366),
        'is_weekend': (day_of_week >= 5).astype(int),
    })



df = pd.read_csv(r'C:\Users\nolan\OneDrive\Documents\GitHub\PE2-AI\ReadyDataSets\Price_Weather_MergedExtraFeatures.csv')
df['DateUTC'] = pd.to_datetime(df['DateUTC'])
df = df.sort_values('DateUTC').reset_index(drop=True)

print(f"Loaded {len(df)} rows")

# ── 2. FEATURES & SCALING 
time_features = make_time_features(df['DateUTC']).values
weather = df[['temperature_2m']].values  
sunshine_duration = df[['sunshine_duration']].values
is_day = df[['is_day']].values
X = np.hstack([time_features, weather,is_day, sunshine_duration])

for price in df['Price_EUR_MWh']:
    if price < -200:
        df.loc[df['Price_EUR_MWh'] == price, 'Price_EUR_MWh'] = -200
    if price > 300:
        df.loc[df['Price_EUR_MWh'] == price, 'Price_EUR_MWh'] = 300


y = df['Price_EUR_MWh'].values                                 

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

print(f"Features: {X.shape[1]}")

# ── 3. SLIDING WINDOW 
# WaveNet needs a power-of-2 window to match the dilation stack nicely
# 2^7 = 128 steps = 32h at 15min
WINDOW = 128

def make_sequences(X, y, window):
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i:i + window])
        ys.append(y[i + window])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = make_sequences(X_scaled, y_scaled, WINDOW)
print(f"Sequence shape: {X_seq.shape}")

# ── 4. TRAIN / TEST SPLIT
split = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]
test_dates = df['DateUTC'].iloc[split + WINDOW:]

print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ── 5. WAVENET MODEL 
# Stack dilated causal Conv1D layers with doubling dilation rates
# Each layer sees exponentially further back in time:
#   dilation 1  → looks 1 step back
#   dilation 2  → looks 2 steps back
#   dilation 4  → looks 4 steps back ... up to 64 = 128 steps total
# Residual connections let gradients flow cleanly through deep stacks

def wavenet_block(x, filters, dilation_rate):
    """Single WaveNet block with residual connection (Géron fig 15-11)"""
    residual = x
    x = keras.layers.Conv1D(
        filters      = filters,
        kernel_size  = 2,
        dilation_rate= dilation_rate,
        padding      = 'causal',        # causal = never looks at future
        activation   = 'relu'
    )(x)
    x = keras.layers.Conv1D(filters, kernel_size=1)(x)  # 1x1 projection
    # residual connection: add input back so gradients flow easily
    if residual.shape[-1] != filters:
        residual = keras.layers.Conv1D(filters, kernel_size=1)(residual)
    return keras.layers.Add()([x, residual])


inputs = keras.layers.Input(shape=(WINDOW, X.shape[1]))
x = inputs

# Initial projection to match filter size
x = keras.layers.Conv1D(32, kernel_size=1)(x)

# Stack of dilated blocks — dilation doubles each time
# This gives a receptive field covering the full 128-step window
for dilation in [1, 2, 4, 8, 16, 32, 64]:
    x = wavenet_block(x, filters=32, dilation_rate=dilation)
    x = keras.layers.Dropout(0.2)(x)

# Only take the last timestep's output for prediction
x = keras.layers.Lambda(lambda t: t[:, -1, :])(x)
x = keras.layers.Dense(16, activation='relu')(x)
outputs = keras.layers.Dense(1)(x)

model = keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='mae')
model.summary()

# ── 6. TRAINING ──────────────────────────────────────────────────────────────
early_stop = keras.callbacks.EarlyStopping(
    monitor             = 'val_loss',
    patience            = 10,
    restore_best_weights= True
)

history = model.fit(
    X_train, y_train,
    epochs           = 50,
    batch_size       = 256,
    validation_split = 0.2,
    callbacks        = [early_stop],
    verbose          = 1
)

# ── 7. EVALUATE ──────────────────────────────────────────────────────────────
y_pred_scaled = model.predict(X_test).flatten()
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

mae = mean_absolute_error(y_true, y_pred)
r2  = r2_score(y_true, y_pred)
print(f"\nResults on test set:")
print(f"  MAE : {mae:.2f} EUR/MWh")
print(f"  R²  : {r2:.4f}")
print(f"\nLSTM baseline — MAE: 22.50 | R²: 0.5425")

# ── 8. PLOTS ─────────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'],     label='Train loss')
plt.plot(history.history['val_loss'], label='Val loss')
plt.title('WaveNet Training History')
plt.xlabel('Epoch')
plt.ylabel('MAE (scaled)')
plt.legend()
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(test_dates.values, y_true, label='Actual',    linewidth=0.8, color='steelblue')
ax.plot(test_dates.values, y_pred, label='Predicted', linewidth=0.8, color='orange', alpha=0.8)
ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
ax.set_title('WaveNet Price Forecast vs Actual (test set)')
ax.set_ylabel('EUR/MWh')
ax.legend()
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_pred, y_true, alpha=0.3, s=5, color='steelblue')
ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
        color='red', linewidth=1, linestyle='--', label='Perfect fit')
ax.set_xlabel('Predicted (EUR/MWh)')
ax.set_ylabel('Actual (EUR/MWh)')
ax.set_title('WaveNet — Actual vs Predicted')
ax.legend()
plt.tight_layout()
plt.savefig(r'C:\Users\nolan\OneDrive\Desktop\PE_AI_TEAM\ElectricalPrice\Nolan\plots\wavenet_scatter.png', dpi=150)
plt.show()