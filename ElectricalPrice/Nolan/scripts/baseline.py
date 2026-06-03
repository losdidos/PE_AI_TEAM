import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv(r'C:\Users\nolan\OneDrive\Documents\GitHub\PE2-AI\ReadyDataSets\Price_Weather_MergedExtraFeatures.csv', parse_dates=['DateUTC'])
df = df.sort_values('DateUTC').reset_index(drop=True)


count = 0
for price in df['Price_EUR_MWh']:
    if price == 0.0:
        count += 1


print(count)


INTERVALS_PER_DAY = 96
df['baseline_pred'] = df['Price_EUR_MWh'].shift(INTERVALS_PER_DAY)

split_index = int(len(df) * 0.8)  
test_df = df.iloc[split_index:].dropna(subset=['baseline_pred'])

actual = test_df['Price_EUR_MWh']
predicted = test_df['baseline_pred']

# Metrics
mae = mean_absolute_error(actual, predicted)
r2  = r2_score(actual, predicted)


perfect_index = pd.date_range(
    start='2024-01-01 00:00:00',
    end='2025-09-30 23:00:00',
    freq='15min',
    tz='UTC'
)

print(f"Perfect index length:  {len(perfect_index):,}")
print(f"Your data length:      {len(df):,}")
print(f"Difference:            {len(perfect_index) - len(df):,} missing intervals")

# Find missing timestamps
your_timestamps = pd.DatetimeIndex(df['DateUTC'])
missing = perfect_index.difference(your_timestamps)

print(f"\nMissing timestamps: {len(missing)}")
if len(missing) > 0:
    print(missing)

# Find extra timestamps 
extra = your_timestamps.difference(perfect_index)
print(f"\nExtra/unexpected timestamps: {len(extra)}")
if len(extra) > 0:
    print(extra)


#plot

fig, ax = plt.subplots(figsize=(7, 7))

ax.scatter(actual, predicted, alpha=0.3, s=10, color='steelblue')

lims = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction')


ax.text(0.05, 0.92, f'R²:  {r2:.4f}\nMAE: {mae:.4f}',
        transform=ax.transAxes, fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_xlabel('Actual Price (EUR/MWh)')
ax.set_ylabel('Predicted Price (EUR/MWh)')
ax.set_title('Baseline Model — Predicted vs Actual (Test Set)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('baseline_pred_vs_actual.png', dpi=150)
plt.show()