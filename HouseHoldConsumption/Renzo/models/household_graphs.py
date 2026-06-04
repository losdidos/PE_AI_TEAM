import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# 1. Define File Paths
allCsv = [
    r'C:\Users\renzo\PyCharmMiscProject\.venv\pracitce enterprise\household_data\houses_datasets\CLEAN_House1.csv',
    r'C:\Users\renzo\PyCharmMiscProject\.venv\pracitce enterprise\household_data\houses_datasets\CLEAN_House2.csv',
    r'C:\Users\renzo\PyCharmMiscProject\.venv\pracitce enterprise\household_data\houses_datasets\CLEAN_House3.csv',
    r'C:\Users\renzo\PyCharmMiscProject\.venv\pracitce enterprise\household_data\houses_datasets\CLEAN_House4.csv',
    r'C:\Users\renzo\PyCharmMiscProject\.venv\pracitce enterprise\household_data\houses_datasets\CLEAN_House5.csv',
    r'C:\Users\renzo\PyCharmMiscProject\.venv\pracitce enterprise\household_data\houses_datasets\CLEAN_House6.csv',
    r'C:\Users\renzo\PyCharmMiscProject\.venv\pracitce enterprise\household_data\houses_datasets\CLEAN_House7.csv',
    r'C:\Users\renzo\PyCharmMiscProject\.venv\pracitce enterprise\household_data\houses_datasets\CLEAN_House8.csv',
    r'C:\Users\renzo\PyCharmMiscProject\.venv\pracitce enterprise\household_data\houses_datasets\CLEAN_House9.csv',
    r'C:\Users\renzo\PyCharmMiscProject\.venv\pracitce enterprise\household_data\houses_datasets\CLEAN_House10.csv',
    r'C:\Users\renzo\PyCharmMiscProject\.venv\pracitce enterprise\household_data\houses_datasets\CLEAN_House11.csv',
    r'C:\Users\renzo\PyCharmMiscProject\.venv\pracitce enterprise\household_data\houses_datasets\CLEAN_House12.csv',
    r'C:\Users\renzo\PyCharmMiscProject\.venv\pracitce enterprise\household_data\houses_datasets\CLEAN_House13.csv',
    r'C:\Users\renzo\PyCharmMiscProject\.venv\pracitce enterprise\household_data\houses_datasets\CLEAN_House15.csv',
    r'C:\Users\renzo\PyCharmMiscProject\.venv\pracitce enterprise\household_data\houses_datasets\CLEAN_House16.csv',
    r'C:\Users\renzo\PyCharmMiscProject\.venv\pracitce enterprise\household_data\houses_datasets\CLEAN_House17.csv',
    r'C:\Users\renzo\PyCharmMiscProject\.venv\pracitce enterprise\household_data\houses_datasets\CLEAN_House18.csv',
    r'C:\Users\renzo\PyCharmMiscProject\.venv\pracitce enterprise\household_data\houses_datasets\CLEAN_House19.csv',
    r'C:\Users\renzo\PyCharmMiscProject\.venv\pracitce enterprise\household_data\houses_datasets\CLEAN_House20.csv',
    r'C:\Users\renzo\PyCharmMiscProject\.venv\pracitce enterprise\household_data\houses_datasets\CLEAN_House21.csv',
]

dfs = []

# 2. Data Loading & Cleaning
print("--- Loading Data ---")
for file in allCsv:
    try:
        df = pd.read_csv(file)

        # Convert to datetime before setting index
        # utc=True handles potential duplicate hours during winter time transitions
        df["Time"] = pd.to_datetime(df["Time"], utc=True)

        # Check for missing values
        ms = df.isnull().sum().sum()
        print(f"Read House {allCsv.index(file) + 1}: {ms} missing values found.")

        # Sort and set index
        df = df.sort_values("Time").set_index("Time")
        dfs.append(df)
    except Exception as e:
        print(f"Error loading {file}: {e}")

# 3. Visualization
print("\n--- Generating Plots ---")
n = len(dfs)
min_time = min(df.index.min() for df in dfs)
max_time = max(df.index.max() for df in dfs)

fig, ax = plt.subplots(n, 1, figsize=(15, 3 * n), sharex=True)
if n == 1: ax = [ax]  # Handle single plot case

for i, df in enumerate(dfs):
    ax[i].plot(df.index, df["Aggregate"], alpha=0.6, color='tab:blue')
    ax[i].set_title(f"House {i + 1}")
    ax[i].set_ylabel("Aggregate")
    ax[i].set_xlim(min_time, max_time)

ax[-1].set_xlabel("Time")
plt.tight_layout()
plt.show()

# 4. Training XGBoost Models
print("\n--- Training Models (XGBoost) ---")
scaler = StandardScaler()

for idx, df in enumerate(dfs, start=1):
    # Clean features: Drop non-numeric or metadata columns
    # We drop Aggregate because it's our target (y)
    X = df.drop(columns=["Unix", "Issues", "Aggregate"], errors="ignore")
    y = df["Aggregate"]

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale Features
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and Train XGBoost
    model = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"House {idx:02d} | MAE: {mae:.4f} | R²: {r2:.4f}")