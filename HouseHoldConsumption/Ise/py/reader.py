import pandas as pd

df = pd.read_csv("sprint2/CC_LCL-FullData_V2.csv")

# Missing value count + percentage
missing = df.isnull().sum()

print(missing)
"""
missing_pct = (missing / len(df)) * 100

summary = pd.DataFrame({"missing_count": missing, "missing_%": missing_pct})
print(summary[summary["missing_count"] > 0])

df.drop(columns=["stdorToU"], inplace=True)


df.to_csv("sprint2/CC_LCL-FullData_V2.csv", index=False)

"""