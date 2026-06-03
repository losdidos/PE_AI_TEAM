# Datasets - Velimir
This folder contains the datasets used in the project for energy consumption prediction and analysis.  
The data is split into two main parts:

1. **Belgium national energy consumption data**
2. **Individual household energy consumption datasets**

The datasets were used for preprocessing, feature engineering, model training, testing, and prediction tasks.

---

## Folder Contents

### 1. Household datasets

The files named similar to:

```text
CleerResampel_House_...
```

contain electricity consumption data for separate households.

These datasets were used to build and test prediction models for individual houses.  
Each household file contains time-based energy consumption values, which were cleaned and prepared before being used for machine learning models.

Typical work done with these files:

- checking missing values
- cleaning and resampling the data
- preparing time-based features
- training prediction models per household
- comparing prediction results between different houses

---

### 2. Household weather data

Files such as:

```
household_weather.csv
weather-data-household...
```

contain weather information used together with the household consumption datasets.

Weather data can help improve energy prediction because temperature and other weather conditions can influence electricity usage.

Typical use:

- merging weather data with household consumption data
- adding temperature/weather features to the model
- improving prediction accuracy

---

### 3. Belgium national energy consumption data

The folder/file group named similar to:

```text
national-consumption-data
Datasetfor2024_modified...
Datasetfor2025_modified...
electricity_consumption_...
merged-national-energy...
original-consumption-va...
output-formatted-nation...
weather-data-interpolat...
```

contains Belgium national energy consumption datasets and related processed files.

These files were used to predict energy consumption on a national level instead of only for separate households.

Typical work done with these files:

- cleaning national consumption values
- preparing 2024 and 2025 datasets
- merging national electricity consumption with weather data
- merging 2024 and 2025 data
- interpolating missing weather values
- formatting output files for model usage
- training and testing national-level prediction models


## Notes

- Some files are original raw datasets.
- Some files are modified or cleaned versions created during preprocessing.
- Some files are merged datasets that combine energy consumption and weather data.
- The data is intended for educational/project use and model development.
