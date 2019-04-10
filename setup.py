from distutils.core import setup

setup(name='TimeSeriesD3MWrappers',
    version='1.0.3',
    description='Five wrappers for interacting with New Knowledge time series tool Sloth',
    packages=['TimeSeriesD3MWrappers'],
    install_requires=["typing",
                      "numpy",
                      "Cython",
                      "Sloth @ git+https://github.com/NewKnowledge/sloth@9c9d3a4df9a0339390569ad8028beba2dd2dd7b4#egg=Sloth-2.0.6",
                      "DistilTimeSeriesLoader @ git+https://github.com/uncharted-distil/distil-timeseries-loader@b781b140ec1328939e7bf6005251b2145d9ca20a#egg=DistilTimeSeriesLoader-0.1.2",
                      ],
    entry_points = {
        'd3m.primitives': [
            'clustering.k_means.Sloth = TimeSeriesD3MWrappers:Storc',
            'clustering.hdbscan.Hdbscan = TimeSeriesD3MWrappers:Hdbscan',
            'time_series_classification.shapelet_learning.Shallot = TimeSeriesD3MWrappers:Shallot',
            'time_series_classification.k_neighbors.Kanine = TimeSeriesD3MWrappers:Kanine',
            'time_series_forecasting.arima.Parrot = TimeSeriesD3MWrappers:Parrot',
            'time_series_forecasting.vector_autoregression.VAR = TimeSeriesD3MWrappers:VAR'
        ],
    },
)
