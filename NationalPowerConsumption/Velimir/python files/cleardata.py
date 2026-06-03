import pandas as pd
from pathlib import Path

input_file = "monthly_hourly_load_values_2024.csv"
output_file = "output.csv"

columns_to_drop = ["CreateDate"]
df = pd.read_csv(input_file, sep='\t')


df = df.drop(columns=columns_to_drop)

df.to_csv(output_file, index=False)

print("Columns dropped (non-existing columns ignored).")
