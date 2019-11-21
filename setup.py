from setuptools import setup

setup(name='TimeSeriesD3MWrappers',
    version='1.1.0',
    description='Five wrappers for interacting with New Knowledge time series tool Sloth',
    packages=['TimeSeriesD3MWrappers'],
    install_requires=['numpy>=1.15.4,<=1.17.3',
                      'statsmodels == 0.10.1',
                      'scikit-learn[alldeps]>=0.20.3,<=0.21.3',
                      'pandas>=0.23.4,<=0.25.2',
                      'tensorflow-gpu == 2.0.0',
                      'tslearn == 0.2.5',
                      'pmdarima==1.0.0',
                      'deepar @ git+https://github.com/NewKnowledge/deepar@285afa40a5adae0274ce44180643eb8dd5b11b31#egg=deepar-0.0.1'
                      ],
    entry_points = {
        'd3m.primitives': [
            'time_series_classification.k_neighbors.Kanine = TimeSeriesD3MWrappers.primitives.classification_knn:Kanine',
            'time_series_forecasting.vector_autoregression.VAR = TimeSeriesD3MWrappers.primitives.forecasting_var:VAR',
            'time_series_classification.convolutional_neural_net.LSTM_FCN = TimeSeriesD3MWrappers.primitives.classification_lstm:LSTM_FCN',
            'time_series_forecasting.convolutional_neural_net.DeepAR = TimeSeriesD3MWrappers.primitives.forecasting_deepar:DeepAR',
        ],
    },
)
