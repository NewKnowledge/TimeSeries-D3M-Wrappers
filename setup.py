from setuptools import setup

setup(name='TimeSeriesD3MWrappers',
    version='1.0.6',
    description='Five wrappers for interacting with New Knowledge time series tool Sloth',
    packages=['TimeSeriesD3MWrappers'],
    install_requires=["typing",
                      "numpy == 1.15.4",
                      'scikit-learn == 0.20.3',
                      'Keras == 2.2.4',
                      "Sloth @ git+https://github.com/NewKnowledge/sloth@c331cec7f9c90642c8726f8cf673c2034493d08b#egg=Sloth-2.0.7",
                      ],
    entry_points = {
        'd3m.primitives': [
            'time_series_classification.shapelet_learning.Shallot = TimeSeriesD3MWrappers:Shallot',
            'time_series_classification.k_neighbors.Kanine = TimeSeriesD3MWrappers:Kanine',
            'time_series_forecasting.arima.Parrot = TimeSeriesD3MWrappers:Parrot',
            'time_series_forecasting.vector_autoregression.VAR = TimeSeriesD3MWrappers:VAR',
            'time_series_classification.convolutional_neural_net.LSTM_FCN = TimeSeriesD3MWrappers:LSTM_FCN'
        ],
    },
)
