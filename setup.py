from setuptools import setup

setup(name='TimeSeriesD3MWrappers',
    version='1.0.6',
    description='Five wrappers for interacting with New Knowledge time series tool Sloth',
    packages=['TimeSeriesD3MWrappers'],
    install_requires=["typing",
                      "numpy == 1.15.4",
                      'scikit-learn == 0.20.3',
                      'Keras == 2.2.4',
                      "Sloth @ git+https://github.com/NewKnowledge/sloth@4af4d9d7391cfc3e89957eb1cd364123d6d262cb#egg=Sloth-2.0.7",
                      ],
    entry_points = {
        'd3m.primitives': [
            'clustering.k_means.Sloth = TimeSeriesD3MWrappers:Storc',
            'clustering.hdbscan.Hdbscan = TimeSeriesD3MWrappers:Hdbscan',
            'time_series_classification.shapelet_learning.Shallot = TimeSeriesD3MWrappers:Shallot',
            'time_series_classification.k_neighbors.Kanine = TimeSeriesD3MWrappers:Kanine',
            'time_series_forecasting.arima.Parrot = TimeSeriesD3MWrappers:Parrot',
            'time_series_forecasting.vector_autoregression.VAR = TimeSeriesD3MWrappers:VAR',
            'time_series_classification.convolutional_neural_net.LSTM_FCN = TimeSeriesD3MWrappers:LSTM_FCN'
        ],
    },
)
