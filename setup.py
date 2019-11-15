from setuptools import setup

setup(name='TimeSeriesD3MWrappers',
    version='1.0.7',
    description='Five wrappers for interacting with New Knowledge time series tool Sloth',
    packages=['TimeSeriesD3MWrappers'],
    install_requires=["numpy<=1.17.3",
                      'scikit-learn<=0.21.3',
                      'Keras == 2.3.1',
                      "Sloth @ git+https://github.com/NewKnowledge/sloth@2c867f6ebba39657540f15a27f0fd466a7ce3d37#egg=Sloth-2.0.8",
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
