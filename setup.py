from setuptools import setup

setup(name='TimeSeriesD3MWrappers',
    version='1.0.3',
    description='Five wrappers for interacting with New Knowledge time series tool Sloth',
    packages=['TimeSeriesD3MWrappers'],
    install_requires=["typing",
                      "numpy",
                      "Sloth @ git+https://github.com/NewKnowledge/sloth@3f527314445bfdc8197ed40b279bd74016d77c1b#egg=Sloth-2.0.6",
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
