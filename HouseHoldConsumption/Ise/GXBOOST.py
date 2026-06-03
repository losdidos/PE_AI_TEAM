import pandas as pd
import numpy as np
import xgboost
from requests.packages import target
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as pl



#__________/>/_Data ophaalen_\<\__________#
df = pd.read_csv("User_MAC000002_15min_met_weer.csv")
df["DateTime"] = pd.to_datetime(df["DateTime"])
df = df.sort_values("DateTime").reset_index(drop=True)


#__________/>/_tijt opsplitten in meerdere collomen_\<\__________#
df["MO"] = df["DateTime"].dt.month
df["DW"] = df["DateTime"].dt.dayofweek
df["U"] = df["DateTime"].dt.hour
df["MI"] = df["DateTime"].dt.minute


#_V2_________/>/_Lag opsplitten in meerdere collomen_\<\__________#

target = "KWH/hh (per kwartier)"

lags = [1,2,4,96]

for lag in lags:
    df[f"tar_lag_{lag}"] = df[target].shift(lag)
    df[f"Tem_lag_{lag}"] = df["temperature_2m"].shift(lag)






"""
df["tar_lag_1"] = df[target].shift(1)
df["tar_lag_4"] = df[target].shift(4)
df["tar_lag_96"] = df[target].shift(96)

df["Tem_lag_1"] = df["temperature_2m"].shift(1)
df["Tem_lag_4"] = df["temperature_2m"].shift(4)
df["Tem_lag_96"] = df["temperature_2m"].shift(96)
"""
df = df.dropna()



#__________/>/_variables maaken om straks makkelijker te assinen_\<\__________#
toegelaate_trijn_data = [
    "temperature_2m",
    "relative_humidity_2m",
    "snowfall",
    "rain",
    "MO",
    "DW",
    "U",
    "MI",
    "tar_lag_1",#bijfevoegt door V2
    "tar_lag_4",
    "tar_lag_96",
    "Tem_lag_1",
    "Tem_lag_4",
    "Tem_lag_96",
    ]
 #target = "KWH/hh (per kwartier)"


#__________/>/_Splitten_\<\__________#

split_Index = int(len(df) * 0.8)#geeft een int trug dat 80% is van de aantal rows det er zijn;°
trijn_set = df.iloc[:split_Index]#?????
test_Set = df.iloc[split_Index:]#?????

X_teijning_Set = trijn_set[toegelaate_trijn_data]
Y_teijning_Test_Set = trijn_set[target]

X_Test_Set = test_Set[toegelaate_trijn_data]
Y_Test_Set_testing = test_Set[target]














#__________/>/_Model aanmaakrn_\<\__________#
print("start model creat")

model1 = xgboost.XGBRegressor(
    n_estimators=2000,
    max_depth=None,
    learning_rate=0.3,
    random_state=42,
)
print("start model train")
model1.fit(X_teijning_Set, Y_teijning_Test_Set) #tijnt

#__________/>/_voorspelling_\<\__________#

y_voorspelling = model1.predict(X_Test_Set)

print("MAE:", mean_absolute_error(Y_Test_Set_testing,y_voorspelling))
print("RMSE:", np.sqrt(mean_squared_error(Y_Test_Set_testing,y_voorspelling)))
print("R²:", r2_score(Y_Test_Set_testing,y_voorspelling))

#__________/>/_plotting_\<\__________#

pl.figure(figsize=(12, 5))
pl.plot(test_Set["DateTime"],Y_Test_Set_testing, label="echte waarde")
pl.plot(test_Set["DateTime"],y_voorspelling, label="voorspelling")
pl.legend()
pl.show()












#__________/>/_Splitten_\<\__________#












#__________/>/_Splitten_\<\__________#












#__________/>/_Splitten_\<\__________#














