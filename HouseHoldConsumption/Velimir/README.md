# HouseHoldConsumption - Velimir
This folder contains the household energy consumption datasets and the Python scripts used for preprocessing, merging weather data, training machine learning models, and predicting household electricity usage as well as some images that shows the predictions in a plot.

### Household datasets

The project contains household energy consumption data for multiple houses. These files are used as the main input for the prediction models. The data is time-based, meaning that every row represents energy consumption at a specific moment in time.

These datasets are used to:

- analyse household electricity usage patterns
- clean and prepare time series data
- test different machine learning models
- predict future electricity consumption
- compare prediction performance between different houses

### Weather data scripts

Some scripts are used to add or prepare weather data for the household datasets. Weather information is useful because electricity consumption can depend on temperature and other weather conditions.

Examples of this part of the workflow:

- adding weather data to household files
- interpolating missing weather values
- preparing weather data before merging it with consumption data

## General workflow

The typical workflow for these files is:

1. Start with the original household consumption datasets.
2. Check the time column and fix missing timestamps.
3. Interpolate missing values where needed.
4. Prepare and clean the weather data.
5. Merge household data with weather data and metadata.
6. Train machine learning models on the prepared data.
7. Predict household energy consumption.
8. Evaluate and compare the results.
9. Plot the original and processed data for visual inspection.