import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('filtered_BE_1V2.csv', sep='\t')

df['DateUTC'] = pd.to_datetime(df['DateUTC'], dayfirst=True)


df_numeric = df[['DateUTC', 'Value']].copy()
df_numeric['Value'] = pd.to_numeric(df_numeric['Value'], errors='coerce')


df_numeric = df_numeric.sort_values('DateUTC')
df_numeric = df_numeric.set_index('DateUTC')


df_numeric = df_numeric.groupby(df_numeric.index).mean()


series_15min = df_numeric['Value'].resample('15min').interpolate(method='linear')


print(series_15min.head(1000))
print("Aantal rijen na 15-min resampels:", series_15min.shape)


plt.figure(figsize=(15,5))
plt.plot(series_15min)
plt.title("15-minuten geïnterpoleerde data")
plt.grid(True)
plt.show()

