import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------
# 1. SETTINGS
# -----------------------------
CSV_PATH = "./electricity_consumption_ready-copy.csv"

DATE_COL = "DateUTC"
TARGET_COL = "Value"


# 4 samples per hour -> 96 samples per day -> 672 samples per week
STEPS_PER_HOUR = 4
STEPS_PER_DAY = 24 * STEPS_PER_HOUR
FUTURE_STEPS = 30 * STEPS_PER_DAY          # 1 week ahead = 672 steps

# How many previous time steps the model sees
LOOKBACK = 3 * STEPS_PER_DAY              # use previous 7 days to predict next step

# Split percentages (chronological)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Training
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4                       # L2 regularization
PATIENCE = 10                             # early stopping
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)
np.random.seed(42)


# -----------------------------
# 2. LOAD DATA
# -----------------------------
df = pd.read_csv(CSV_PATH)
df[DATE_COL] = pd.to_datetime(df[DATE_COL])
df = df.sort_values(DATE_COL).reset_index(drop=True)

# Keep only needed columns
df = df[[DATE_COL, TARGET_COL]].copy()

print("Data shape:", df.shape)
print("Date range:", df[DATE_COL].min(), "to", df[DATE_COL].max())


# -----------------------------
# 3. TIME-BASED SPLIT
# -----------------------------
n = len(df)

train_end = int(n * TRAIN_RATIO)
val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

train_df = df.iloc[:train_end].copy()
val_df   = df.iloc[train_end:val_end].copy()
test_df  = df.iloc[val_end:].copy()

print(f"Train: {len(train_df)} rows")
print(f"Val:   {len(val_df)} rows")
print(f"Test:  {len(test_df)} rows")



# 4. NORMALIZE USING TRAIN ONLY

#
# We fit the scaler only on training data to avoid leakage.
scaler = StandardScaler()
scaler.fit(train_df[[TARGET_COL]])

df["Value_scaled"] = scaler.transform(df[[TARGET_COL]]).flatten()



# 5. ADD KNOWN TIME FEATURES

# These are safe for the future because they come from the timestamp,
# not from future observed values.
def add_time_features(frame):
    frame = frame.copy()
    frame["hour"] = frame[DATE_COL].dt.hour
    frame["minute"] = frame[DATE_COL].dt.minute
    frame["dayofweek"] = frame[DATE_COL].dt.dayofweek

    # position within day: 0..95
    frame["step_in_day"] = frame["hour"] * 4 + (frame["minute"] // 15)

    # cyclical encoding
    frame["sin_day"] = np.sin(2 * np.pi * frame["step_in_day"] / 96)
    frame["cos_day"] = np.cos(2 * np.pi * frame["step_in_day"] / 96)
    frame["sin_week"] = np.sin(2 * np.pi * frame["dayofweek"] / 7)
    frame["cos_week"] = np.cos(2 * np.pi * frame["dayofweek"] / 7)

    return frame

df = add_time_features(df)

feature_cols = ["Value_scaled", "sin_day", "cos_day", "sin_week", "cos_week"]



# 6. BUILD WINDOWS

# Each sample:
# X = previous LOOKBACK steps
# y = next step
# We will split by the TARGET timestamp to keep chronology correct.

all_features = df[feature_cols].values.astype(np.float32)
all_targets_scaled = df["Value_scaled"].values.astype(np.float32)
all_targets_real = df[TARGET_COL].values.astype(np.float32)
all_dates = df[DATE_COL].values

X_list = []
y_list = []
y_real_list = []
target_date_list = []

for i in range(LOOKBACK, len(df)):
    X_window = all_features[i - LOOKBACK:i]        # past only
    y_target = all_targets_scaled[i]               # next step target
    y_real = all_targets_real[i]
    target_date = all_dates[i]

    X_list.append(X_window)
    y_list.append(y_target)
    y_real_list.append(y_real)
    target_date_list.append(target_date)

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32).reshape(-1, 1)
y_real = np.array(y_real_list, dtype=np.float32).reshape(-1, 1)
target_dates = np.array(target_date_list)

print("Windowed X shape:", X.shape)
print("Windowed y shape:", y.shape)



# 7. SPLIT WINDOWS BY TARGET TIME

train_cutoff_date = df.iloc[train_end][DATE_COL]
val_cutoff_date = df.iloc[val_end][DATE_COL]

train_mask = target_dates < np.datetime64(train_cutoff_date)
val_mask   = (target_dates >= np.datetime64(train_cutoff_date)) & (target_dates < np.datetime64(val_cutoff_date))
test_mask  = target_dates >= np.datetime64(val_cutoff_date)

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val     = X[val_mask], y[val_mask]
X_test, y_test   = X[test_mask], y[test_mask]

y_test_real = y_real[test_mask]
test_dates = target_dates[test_mask]

print("Train windows:", X_train.shape, y_train.shape)
print("Val windows:  ", X_val.shape, y_val.shape)
print("Test windows: ", X_test.shape, y_test.shape)



# 8. DATASET / DATALOADER

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = TimeSeriesDataset(X_train, y_train)
val_ds = TimeSeriesDataset(X_val, y_val)
test_ds = TimeSeriesDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


# 9. MODEL

# This is intentionally not too big to reduce overfitting.
class MLPForecaster(nn.Module):
    def __init__(self, lookback, n_features):
        super().__init__()
        input_dim = lookback * n_features

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x shape: (batch, lookback, features)
        x = x.view(x.size(0), -1)
        return self.net(x)

model = MLPForecaster(lookback=LOOKBACK, n_features=len(feature_cols)).to(DEVICE)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


# -----------------------------
# 10. TRAINING WITH EARLY STOPPING
# -----------------------------
best_val_loss = np.inf
best_state = None
patience_counter = 0

train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    # ---- train ----
    model.train()
    running_train_loss = 0.0

    for xb, yb in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * xb.size(0)

    epoch_train_loss = running_train_loss / len(train_loader.dataset)

    # ---- validation ----
    model.eval()
    running_val_loss = 0.0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            preds = model(xb)
            loss = criterion(preds, yb)
            running_val_loss += loss.item() * xb.size(0)

    epoch_val_loss = running_val_loss / len(val_loader.dataset)

    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)

    print(
        f"Epoch {epoch+1:3d}/{EPOCHS} | "
        f"Train Loss: {epoch_train_loss:.6f} | "
        f"Val Loss: {epoch_val_loss:.6f}"
    )

    # ---- early stopping ----
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        best_state = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

# restore best model
model.load_state_dict(best_state)

# -----------------------------
# SAVE MODEL + SCALER
# -----------------------------
import pickle

# Save model weights
torch.save(model.state_dict(), "model.pth")

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model and scaler saved!")

# ---- validation ----
model.eval()
running_val_loss = 0.0


# -----------------------------
# 11. TEST EVALUATION
# -----------------------------
model.eval()
test_preds_scaled = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        preds = model(xb)
        test_preds_scaled.append(preds.cpu().numpy())

test_preds_scaled = np.vstack(test_preds_scaled)

# Inverse transform back to original Value scale
test_preds = scaler.inverse_transform(test_preds_scaled)
test_actual = y_test_real

mae = mean_absolute_error(test_actual, test_preds)
rmse = np.sqrt(mean_squared_error(test_actual, test_preds))
r2 = r2_score(test_actual, test_preds)

print("\nTest Results")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# -----------------------------
# 12. PLOTS: LOSS CURVES
# -----------------------------
plt.figure(figsize=(12, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# -----------------------------
# 13. PLOTS: ACTUAL VS PREDICTED
# -----------------------------
plt.figure(figsize=(14, 6))
plt.plot(pd.to_datetime(test_dates), test_actual, label="Actual")
plt.plot(pd.to_datetime(test_dates), test_preds, label="Predicted")
plt.title("Actual vs Predicted on Test Set")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# -----------------------------
# 14. SCATTER PLOT
# -----------------------------
plt.figure(figsize=(7, 7))
plt.scatter(test_actual, test_preds, alpha=0.5)
min_v = min(test_actual.min(), test_preds.min())
max_v = max(test_actual.max(), test_preds.max())
plt.plot([min_v, max_v], [min_v, max_v], "--")
plt.title("Scatter Plot: Actual vs Predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.grid(True)
plt.tight_layout()
plt.show()


# -----------------------------
# 15. ONE-WEEK FUTURE FORECAST
# -----------------------------
# We forecast recursively:
# predict next value -> append it -> predict next -> ...
# Time features are known from future timestamps.

# Start from the last LOOKBACK rows of the full dataset
history_df = df.copy()

last_timestamp = history_df[DATE_COL].iloc[-1]
future_dates = pd.date_range(
    start=last_timestamp + pd.Timedelta(minutes=15),
    periods=FUTURE_STEPS,
    freq="15min"
)

future_preds_scaled = []

# rolling history window for recursive forecasting
history_features = history_df[feature_cols].values.astype(np.float32).tolist()

for future_dt in future_dates:
    # build current model input from latest LOOKBACK rows
    current_window = np.array(history_features[-LOOKBACK:], dtype=np.float32)
    current_window_tensor = torch.tensor(current_window, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        next_pred_scaled = model(current_window_tensor).cpu().numpy()[0, 0]

    future_preds_scaled.append(next_pred_scaled)

    # create future cyclical features from timestamp
    hour = future_dt.hour
    minute = future_dt.minute
    dayofweek = future_dt.dayofweek
    step_in_day = hour * 4 + (minute // 15)

    sin_day = np.sin(2 * np.pi * step_in_day / 96)
    cos_day = np.cos(2 * np.pi * step_in_day / 96)
    sin_week = np.sin(2 * np.pi * dayofweek / 7)
    cos_week = np.cos(2 * np.pi * dayofweek / 7)

    # append predicted value + known future time features
    history_features.append([next_pred_scaled, sin_day, cos_day, sin_week, cos_week])

future_preds_scaled = np.array(future_preds_scaled).reshape(-1, 1)
future_preds = scaler.inverse_transform(future_preds_scaled).flatten()


# -----------------------------
# 16. PLOT LAST PART OF SERIES + ONE-WEEK FORECAST
# -----------------------------
# Show last 2 weeks of actual data plus 1 future week prediction
history_plot_steps = 14 * STEPS_PER_DAY

past_dates = df[DATE_COL].iloc[-history_plot_steps:]
past_values = df[TARGET_COL].iloc[-history_plot_steps:]

plt.figure(figsize=(14, 6))
plt.plot(past_dates, past_values, label="Historical Actual")
plt.plot(future_dates, future_preds, label="1-Week Forecast")
plt.axvline(df[DATE_COL].iloc[-1], linestyle="--", label="Forecast Start")
plt.title("One-Week Ahead Forecast")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# -----------------------------
# 17. OPTIONAL: SHOW FIRST FEW FUTURE PREDICTIONS
# -----------------------------
future_forecast_df = pd.DataFrame({
    "DateUTC": future_dates,
    "Predicted_Value": future_preds
})

print("\nFuture 1-week forecast (first 10 rows):")
print(future_forecast_df.head(10))