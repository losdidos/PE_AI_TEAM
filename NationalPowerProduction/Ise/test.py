import pandas as pd
from statsmodels.tsa.stattools import adfuller


df = pd.read_csv("filtered_BE_1V2.csv", sep="\t")


print(df.head())

print(df.isnull())



print(df.isnull().sum())


#print(df.info())


series = df['Value']


result = adfuller(series)

print("ADF statistic:", result[0])
print("p-value:", result[1])
print("Critical values:", result[4])

#conclusi data is stationary!!!!!!!!!!!!!