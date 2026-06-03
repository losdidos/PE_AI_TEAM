# NationalPowerConsumption - Velimir
This folder contains the Python scripts used for preparing, cleaning, analysing, training, and predicting Belgian national electricity consumption data.

The goal of these files is to support the machine learning workflow for predicting energy consumption using historical national consumption data together with weather information.


### Python scripts

- `checkForStationary.py`  
  Used to check whether the electricity consumption time series is stationary. This helps decide whether extra preprocessing or transformations are needed before modelling.

- `cleardata.py`  
  Cleans the raw dataset by removing or fixing incorrect values, formatting columns, and preparing the data for the next processing steps.

- `dataset2024.py`  
  Prepares or filters the dataset for the year 2024. This file was used to work with a specific yearly part of the Belgian national consumption data.

- `improved_lstm_model.py`  
  Contains an improved LSTM model for energy consumption prediction. This was used to test a deep learning approach on the time-series data.

- `interpolating.py`  
  Handles missing values in the dataset by interpolating data points. This helps create a continuous time series without gaps.

- `joblibneuralnetworkenergy...py`  
  Used for saving or loading a neural network model with Joblib, so the trained model can be reused without training again.

- `merged.py`  
  Merges different datasets together, such as national electricity consumption data and weather data.

- `neuralnetworkpoweco...py`  
  Contains a neural network model for predicting electricity consumption.

- `plot.py`  
  Creates plots and visualisations of the datasets, predictions, and model results.

- `predict.py`  
  Uses a trained model to make predictions on new or prepared data.

- `pytorchNeuralNetwork.py`  
  Contains a PyTorch-based neural network model used for testing deep learning predictions.

- `xgboost-optimized-energy...py`  
  Contains the optimized XGBoost model for predicting Belgian electricity consumption. This was one of the main machine learning approaches used in the project.


  
## Notes

- Some scripts were experimental and were used to compare different modelling approaches.
- The XGBoost optimized model was used as one of the main prediction models.
- The result images are useful for explaining and presenting the model performance.