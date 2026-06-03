import pandas as pd

df = pd.read_csv('sprint2/CC_LCL-FullData.csv')
print(df.head())               # Should show four clean columns
print(df.info(memory_usage='deep'))   # Check real memory usagt