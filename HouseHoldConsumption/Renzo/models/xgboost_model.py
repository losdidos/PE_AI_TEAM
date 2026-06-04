import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

model = XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

scaler = StandardScaler()

houses_paths = [
    r'C:\Users\renzo\PyCharmMiscProject\.venv\pracitce enterprise\household_data\houses_datasets\CLEAN_House1.csv',
    r'C:\Users\renzo\PyCharmMiscProject\.venv\pracitce enterprise\household_data\houses_datasets\CLEAN_House2.csv',
    r'C:\Users\renzo\PyCharmMiscProject\.venv\pracitce enterprise\household_data\houses_datasets\CLEAN_House3.csv',
    r'C:\Users\renzo\PyCharmMiscProject\.venv\pracitce enterprise\household_data\houses_datasets\CLEAN_House4.csv',
    r'C:\Users\renzo\PyCharmMiscProject\.venv\pracitce enterprise\household_data\houses_datasets\CLEAN_House5.csv'
]

for idx, path in enumerate(houses_paths, start=1):
    print(f"\n=== Processing House {idx} ===")

    df = pd.read_csv(path)
    df = df.drop(columns=["Time", "Unix", "Issues"], errors="ignore")

    X = df.drop(columns=["Aggregate"])
    y = df["Aggregate"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train XGBoost
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False
    )

    # Predict and Evaluate
    y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")