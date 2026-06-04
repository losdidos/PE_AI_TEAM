import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import TextBox, Button
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib


# Load Data

df = joblib.load("./Powerconsumption.joblib")

dates_test = df["dates_test"]
y_test     = df["y_test"]
y_pred     = df["y_pred"]
forecast_df = df["forecast_df"]

# animation

test_df = pd.DataFrame({
    "DateUTC":   dates_test.values,
    "Actual":    y_test.values,
    "Predicted": y_pred
})
test_df["Date"] = pd.to_datetime(test_df["DateUTC"]).dt.date
unique_dates = sorted(test_df["Date"].unique())

print(f"Available dates: {unique_dates[0]} to {unique_dates[-1]}")
print(f"Total days     : {len(unique_dates)}")
print("-" * 60)

NUM_DAYS = 5 #number of days


user_input = input(f"\nEnter a start date (YYYY-MM-DD) to view {NUM_DAYS} days, or 'q' to quit: ")


try:
    start_date = pd.to_datetime(user_input).date()

    if start_date not in unique_dates:
        print(f"Date {start_date} not found. Available range: {unique_dates[0]} to {unique_dates[-1]}")


    start_idx     = unique_dates.index(start_date)
    selected_days = unique_dates[start_idx : start_idx + NUM_DAYS]

    if len(selected_days) < NUM_DAYS:
        print(f"Warning: only {len(selected_days)} days available from {start_date}.")

    day_data         = test_df[test_df["Date"].isin(selected_days)].reset_index(drop=True)
    hours            = [i * 0.25 for i in range(len(day_data))]
    actual_values    = day_data["Actual"].values
    predicted_values = day_data["Predicted"].values
    date_labels      = day_data["Date"].values

    fig, ax = plt.subplots(figsize=(18, 6))
    plt.subplots_adjust(bottom=0.15)

    ax.set_xlabel("Time (hours)", fontsize=11)
    ax.set_ylabel("Consumption", fontsize=11)

    all_vals = np.concatenate([actual_values, predicted_values])
    margin   = (all_vals.max() - all_vals.min()) * 0.1
    ax.set_ylim([all_vals.min() - margin, all_vals.max() + margin])
    ax.set_xlim([0, len(selected_days) * 24])
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)

    ax.set_xticks([d * 24 for d in range(len(selected_days) + 1)])
    ax.set_xticklabels(
        [str(selected_days[d]) if d < len(selected_days) else "" for d in range(len(selected_days) + 1)],
        rotation=30, ha="right", fontsize=9
    )

    for d in range(1, len(selected_days)):
        ax.axvline(d * 24, color="grey", linewidth=0.8, linestyle=":", alpha=0.6)

    ax.plot(hours, actual_values, color="steelblue", linewidth=1.2, alpha=0.6, label="Actual")
    line_pred, = ax.plot([], [], color="orange", linewidth=2.5, alpha=0.9, label="Predicted")

    ax.legend(loc="upper right")

    stats_text = f"Min: {predicted_values.min():.1f}  |  Max: {predicted_values.max():.1f}  |  Avg: {predicted_values.mean():.1f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    def animate(frame):
        n            = frame + 1
        current_val  = predicted_values[n - 1]
        current_hour = hours[n - 1] % 24
        current_day  = date_labels[n - 1]
        line_pred.set_data(hours[:n], predicted_values[:n])
        ax.set_title(
            f"Electricity Consumption  |  {current_day}  {current_hour:.2f}h  |  Value: {current_val:.1f}",
            fontsize=11, fontweight="bold"
        )
        return line_pred,

    ani = FuncAnimation(fig, animate, frames=len(hours), interval=40, blit=True, repeat=True)
    plt.show()

except Exception as e:
    print(f"Invalid input. Use format YYYY-MM-DD.")
    print(f"Error: {e}")

