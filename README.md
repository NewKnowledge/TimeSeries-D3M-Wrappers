# Repository of D3M wrappers for time series classification and forecasting

**TimeSeriesD3MWrappers/primitives**: 

D3M primitives

1. **classification_knn.py**: wrapper for tslearn's KNeighborsTimeSeriesClassifier algorithm 

2. **classification_lstm.py**: wrapper for LSTM Fully Convolutional Networks for Time Series Classification paper, original repo (https://github.com/titu1994/MLSTM-FCN), paper (https://arxiv.org/abs/1801.04503)

3. **forecasting_deepar.py**: wrapper for DeepAR recurrent, autoregressive Time Series Forecasting algorithm (https://arxiv.org/abs/1704.04110). Custom implementation repo (git+https://github.com/NewKnowledge/deepar#egg=deepar-0.0.1)

4. **forecasting_var.py**: wrapper for **statsmodels**' implementation of vector autoregression for multivariate time series

**TimeSeriesD3MWrappers/pipelines**: 

Example pipelines for primitives. Latest are: 

1. **forecasting_pipeline_imputer.py**: pipeline for DeepAR primitive on all forecasting datasets

2. **forecasting_pipeline_var.py**: pipeline for VAR primitive on all forecasting datasets (except terra datasets)

3. **Kanine_pipeline.py**: pipeline for Kanine primitive on all classification datasets

4. **LSTM_FCN_pipeline.py**: pipeline for LSTM_FCN primitive on all classification datasets

Model utils

1. **layer_utils.py**: implementation of AttentionLSTM in tensorflow (compatible with 2), originally from https://github.com/houshd/LSTM-FCN

2. **lstm_model_utils.py**: functions to generate LSTM_FCN model architecture and data generators

3. **var_model_utils.py**: wrapper of the **auto_arima** method from **pmdarima.arima** with some specific parameters fixed





