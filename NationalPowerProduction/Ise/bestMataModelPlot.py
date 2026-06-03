import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, r2_score

def main():

    df = pd.read_csv('Power_Production_Weather.csv')
    df['DateUTC'] = pd.to_datetime(df['DateUTC'])
    df = df.sort_values('DateUTC').drop_duplicates('DateUTC').reset_index(drop=True)

    df_prophet = df.rename(columns={
        'DateUTC': 'ds',
        'Value':   'y',
        'temperature_2m': 'temperature'
    })[['ds', 'y', 'temperature']]

    df_prophet = (df_prophet
        .set_index('ds')
        .resample('15min')
        .mean()
        .interpolate('time')
        .reset_index())

    print(f"Rows: {len(df_prophet)}")


    split = int(len(df_prophet) * 0.8)
    train_df = df_prophet.iloc[:split]
    test_df  = df_prophet.iloc[split:]
    print(f"Train: {len(train_df)} rows  |  Test: {len(test_df)} rows")


    print("\nTraining with best parameters...")
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        changepoint_prior_scale=0.01,       )
    model.add_seasonality(
        name='intraday', period=1,
        fourier_order=4,
        prior_scale=0.1
    )
    model.add_regressor('temperature')
    model.fit(train_df)
    print("Model trained!")


    forecast_test = model.predict(test_df[['ds', 'temperature']])
    mae = mean_absolute_error(test_df['y'].values, forecast_test['yhat'].values)
    r2  = r2_score(test_df['y'].values, forecast_test['yhat'].values)
    print(f"\nTest results:")
    print(f"  MAE : {mae:.4f}")
    print(f"  R²  : {r2:.4f}")


    plot_train = train_df.iloc[-1344:]

    fig, ax = plt.subplots(figsize=(16, 6))

    # Actual values
    ax.plot(plot_train['ds'], plot_train['y'],
            color='steelblue', linewidth=0.8, label='Actual (train)')
    ax.plot(test_df['ds'], test_df['y'],
            color='green', linewidth=0.8, label='Actual (test)')


    ax.plot(forecast_test['ds'], forecast_test['yhat'],
            color='red', linewidth=0.8, linestyle='--', label='Predicted (test)')


    ax.fill_between(forecast_test['ds'],
                    forecast_test['yhat_lower'],
                    forecast_test['yhat_upper'],
                    alpha=0.2, color='red', label='Uncertainty band')


    ax.axvline(x=test_df['ds'].iloc[0], color='black',
               linestyle='--', linewidth=1.5, label='Train/Test split')

    ax.set_title(f'Prophet — Train vs Test  |  MAE: {mae:.1f}  |  R²: {r2:.4f}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Energy Production')
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('train_vs_test.png', dpi=150)
    plt.close()
    print("\nPlot saved as train_vs_test.png ✅")

if __name__ == '__main__':
    main()