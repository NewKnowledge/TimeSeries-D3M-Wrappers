from setuptools import setup

setup(name='TimeSeriesD3MWrappers',
    version='1.0.8',
    description='Five wrappers for interacting with New Knowledge time series tool Sloth',
    packages=['TimeSeriesD3MWrappers'],
    install_requires=["numpy<=1.17.3",
                      'scikit-learn<=0.21.3',
                      'tensorflow-gpu == 2.0.0',
                      'tslearn == 0.2.5'
                      ],
    entry_points = {
        'd3m.primitives': [
            'time_series_classification.k_neighbors.Kanine = TimeSeriesD3MWrappers.primitives.Kanine:Kanine',
            'time_series_forecasting.arima.Parrot = TimeSeriesD3MWrappers.primitives.Parrot:Parrot',
            'time_series_forecasting.vector_autoregression.VAR = TimeSeriesD3MWrappers.primitives.VAR:VAR',
            'time_series_classification.convolutional_neural_net.LSTM_FCN = TimeSeriesD3MWrappers.primitives.LSTM_FCN:LSTM_FCN',
            'data_transformation.data_cleaning.DistilTimeSeriesFormatter = TimeSeriesD3MWrappers.primitives.timeseries_formatter:TimeSeriesFormatterPrimitive'
        ],
    },
)
