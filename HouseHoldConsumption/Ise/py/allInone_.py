import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("User_MAC000002_data.csv")

df['DateTime']=pd.to_datetime(df['DateTime'])

df.set_index('DateTime', inplace=True)

df.sort_index(inplace=True)

print(df.isnull().sum())

df = df.groupby(df.index).agg({"KWH/hh (per half hour)":"mean","LCLid": "first"})



#Gemiddelde verdeling (bij totalen)

#Lineaire interpolatie

df_15 = df.resample('15min').asfreq()

df_15["KWH/hh (per half hour)"] = (df_15["KWH/hh (per half hour)"].ffill() / 2)
print(df_15.isnull().sum())
print(df_15)
print(df)







