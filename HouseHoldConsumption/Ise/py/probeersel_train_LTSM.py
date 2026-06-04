import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler



from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def modelBouwer(hp):
    model = Sequential()

    model.add(LSTM(
        units=hp.Choise()))




df = pd.read_csv('User_MAC000002_15min_met_weer.csv', parse_dates=['DateTime'],index_col='DateTime')

window_size = 96 # (96 * 15) = 1440 / 60 = 24 = 1_dag

