import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math

# Read the data
df = pd.read_csv(r'C:\Users\renzo\PyCharmMiscProject\.venv\pracitce enterprise\household_data\dataset\clean_household_weather_extended.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Get all houses
houses = df['house_id'].unique()
n_houses = len(houses)

# Create a figure with subplots
cols = 3
rows = math.ceil(n_houses / cols)
fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows), sharex=False)
axes = axes.flatten()

# Dictionary to store gap information
gap_info = {}

for i, house in enumerate(houses):
    ax = axes[i]
    df_house = df[df['house_id'] == house].sort_values('timestamp').copy()

    # Calculate time differences between consecutive measurements
    df_house['time_diff'] = df_house['timestamp'].diff()

    # Identify gaps (gaps larger than 1 day, adjust threshold as needed)
    gap_threshold = pd.Timedelta(days=1)
    gaps = df_house[df_house['time_diff'] > gap_threshold]

    # Calculate data coverage
    date_range = df_house['timestamp'].max() - df_house['timestamp'].min()
    expected_points = date_range.days + 1  # Assuming daily data
    actual_points = len(df_house)
    coverage_pct = (actual_points / expected_points * 100) if expected_points > 0 else 0

    # Store gap information
    gap_info[house] = {
        'gaps': gaps,
        'coverage': coverage_pct,
        'n_gaps': len(gaps),
        'start_date': df_house['timestamp'].min(),
        'end_date': df_house['timestamp'].max(),
        'data_points': actual_points
    }

    # Plot the data
    ax.plot(df_house['timestamp'], df_house['aggregate_w'],
            marker='.', markersize=2, linewidth=1, alpha=0.7, label='Data')

    # Highlight gaps with red vertical spans
    for _, gap in gaps.iterrows():
        gap_start = gap['timestamp'] - gap['time_diff']
        gap_end = gap['timestamp']
        ax.axvspan(gap_start, gap_end, alpha=0.3, color='red', label='Gap' if _ == gaps.index[0] else "")

    # Formatting
    ax.set_title(f'House {house} - {coverage_pct:.1f}% coverage\n{actual_points} points, {len(gaps)} gaps')
    ax.set_ylabel('Energy (W)')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)

    # Add year indicator
    years = df_house['timestamp'].dt.year.unique()
    if len(years) > 0:
        ax.text(0.02, 0.98, f'Years: {", ".join(map(str, sorted(years)))}',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Remove empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle('Energy Consumption per House with Gap Visualization', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# Print summary of houses with complete yearly data
print("\n" + "=" * 80)
print("HOUSES WITH COMPLETE YEAR DATA (no gaps > 1 day)")
print("=" * 80)

# Find houses that have continuous data for at least one full year
complete_year_houses = []
for house, info in gap_info.items():
    if info['n_gaps'] == 0 and info['coverage'] >= 99.5:
        days_of_data = (info['end_date'] - info['start_date']).days
        if days_of_data >= 365:
            complete_year_houses.append({
                'house': house,
                'start_date': info['start_date'],
                'end_date': info['end_date'],
                'data_points': info['data_points'],
                'days': days_of_data
            })

if complete_year_houses:
    print(f"\nFound {len(complete_year_houses)} houses with complete year of data:")
    for h in complete_year_houses:
        print(f"House {h['house']}: {h['start_date'].date()} to {h['end_date'].date()} "
              f"({h['days']} days, {h['data_points']} points)")
else:
    print("\nNo houses with complete year of data found.")

# Alternative: Find the longest continuous period for each house
print("\n" + "=" * 80)
print("LONGEST CONTINUOUS PERIODS PER HOUSE")
print("=" * 80)

for house in houses[:10]:  # Show first 10 houses as example
    df_house = df[df['house_id'] == house].sort_values('timestamp').copy()
    df_house['time_diff'] = df_house['timestamp'].diff()

    # Identify gap boundaries
    gap_threshold = pd.Timedelta(days=2)  # Consider gaps > 2 days as breaks
    is_break = df_house['time_diff'] > gap_threshold
    df_house['segment'] = is_break.cumsum()

    # Calculate lengths of continuous segments
    segment_lengths = df_house.groupby('segment')['timestamp'].agg(['min', 'max', 'count'])
    segment_lengths['duration_days'] = (segment_lengths['max'] - segment_lengths['min']).dt.days

    # Find longest segment
    longest = segment_lengths.nlargest(1, 'duration_days').iloc[0]

    print(f"House {house}: Longest continuous period = {longest['duration_days']} days "
          f"({longest['min'].date()} to {longest['max'].date()}, "
          f"{longest['count']} points)")

# Create a summary DataFrame for all houses
summary_df = pd.DataFrame([
    {
        'house': house,
        'start_date': info['start_date'],
        'end_date': info['end_date'],
        'data_points': info['data_points'],
        'coverage_pct': info['coverage'],
        'n_gaps': info['n_gaps'],
        'total_days': (info['end_date'] - info['start_date']).days
    }
    for house, info in gap_info.items()
])

print("\n" + "=" * 80)
print("SUMMARY OF ALL HOUSES (sorted by coverage)")
print("=" * 80)
print(summary_df.sort_values('coverage_pct', ascending=False).to_string(index=False))