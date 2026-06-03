import matplotlib.pyplot as plt
import pandas as pd
import itertools
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, r2_score

def train_and_evaluate(params, train_df, test_df):
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        changepoint_prior_scale=params['changepoint_prior_scale'],
    )
    model.add_seasonality(
        name='intraday', period=1,
        fourier_order=params['fourier_order'],
        prior_scale=params['prior_scale']
    )
    model.add_regressor('temperature')
    model.fit(train_df)

    forecast_test = model.predict(test_df[['ds', 'temperature']])
    mae = mean_absolute_error(test_df['y'].values, forecast_test['yhat'].values)
    r2  = r2_score(test_df['y'].values, forecast_test['yhat'].values)
    return mae, r2

def main():

    df = pd.read_csv('Power_Production_Weather.csv')
    df['DateUTC'] = pd.to_datetime(df['DateUTC'])
    df = df.sort_values('DateUTC').drop_duplicates('DateUTC').reset_index(drop=True)

    TEMP_COL = 'temperature_2m'

    df_prophet = df.rename(columns={
        'DateUTC': 'ds',
        'Value':   'y',
        TEMP_COL:  'temperature'
    })[['ds', 'y', 'temperature']]

    df_prophet = (df_prophet
        .set_index('ds')
        .resample('15min')
        .mean()
        .interpolate('time')
        .reset_index())

    print(f"Rows after resampling: {len(df_prophet)}")


    split = int(len(df_prophet) * 0.8)
    train_df = df_prophet.iloc[:split]
    test_df  = df_prophet.iloc[split:]
    print(f"Train: {len(train_df)} rows  |  Test: {len(test_df)} rows")

    param_grid = {
        'changepoint_prior_scale': [0.01, 0.05, 0.1, 0.3],
        'fourier_order':           [4, 8, 16],
        'prior_scale':             [0.01, 0.1, 0.5],
    }



    all_combinations = list(itertools.product(
        param_grid['changepoint_prior_scale'],
        param_grid['fourier_order'],
        param_grid['prior_scale']
    ))
    total = len(all_combinations)
    print(f"\nTesting {total} parameter combinations...")
    print("This will take a while \n")

    results = []
    for i, (cps, fo, ps) in enumerate(all_combinations):
        params = {
            'changepoint_prior_scale': cps,
            'fourier_order': fo,
            'prior_scale': ps,
        }
        print(f"[{i+1}/{total}] Testing: {params}")
        mae, r2 = train_and_evaluate(params, train_df, test_df)
        results.append({**params, 'mae': mae, 'r2': r2})
        print(f"         MAE: {mae:.2f}  |  R²: {r2:.4f}")


    results_df = pd.DataFrame(results).sort_values('mae')
    print("\n========== TOP 5 PARAMETER COMBINATIONS ==========")
    print(results_df.head(5).to_string(index=False))

    best = results_df.iloc[0]
    print(f"\nBEST PARAMETERS:")
    print(f"   changepoint_prior_scale : {best['changepoint_prior_scale']}")
    print(f"   fourier_order           : {int(best['fourier_order'])}")
    print(f"   prior_scale             : {best['prior_scale']}")
    print(f"   MAE                     : {best['mae']:.4f}")
    print(f"   R²                      : {best['r2']:.4f}")


    print("\nTraining final model with best parameters...")
    model_full = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        changepoint_prior_scale=best['changepoint_prior_scale'],
    )
    model_full.add_seasonality(
        name='intraday', period=1,
        fourier_order=int(best['fourier_order']),
        prior_scale=best['prior_scale']
    )
    model_full.add_regressor('temperature')
    model_full.fit(df_prophet)
    print("Final model trained!")


    future = model_full.make_future_dataframe(periods=672, freq='15min')
    avg_temp = df_prophet['temperature'].mean()
    future = future.merge(df_prophet[['ds', 'temperature']], on='ds', how='left')
    future['temperature'] = future['temperature'].fillna(avg_temp)
    forecast = model_full.predict(future)


    print("\nRunning cross-validation...")
    df_cv = cross_validation(
        model_full,
        initial='365 days',
        period='60 days',
        horizon='7 days',
        parallel=None
    )
    cv_metrics = performance_metrics(df_cv)


    fig1 = model_full.plot(forecast)
    plt.title('Energy Production Forecast (best model)')
    plt.tight_layout()
    plt.savefig('forecast.png')
    plt.close()

    fig2 = model_full.plot_components(forecast)
    plt.tight_layout()
    plt.savefig('components.png')
    plt.close()


    print("\n========== FINAL RESULTS ==========")
    print(f"  Best MAE  : {best['mae']:.4f}")
    print(f"  Best R²   : {best['r2']:.4f}")
    print("\nCross-validation metrics:")
    print(cv_metrics[['horizon', 'mae', 'rmse', 'mape']].to_string())
    print("\nPlots saved as forecast.png and components.png")
    print("===================================")

if __name__ == '__main__':
    main()


