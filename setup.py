from distutils.core import setup

setup(name='TimeSeriesD3MWrappers',
    version='1.0.3',
    description='Five wrappers for interacting with New Knowledge time series tool Sloth',
    packages=['TimeSeriesD3MWrappers'],
    install_requires=["typing",
                      "Sloth==2.0.5",
                      "DistilTimeSeriesLoader],
    dependency_links=[
        "git+https://github.com/NewKnowledge/sloth@fd86004a67965065cf1687f9d756c2ed7493d1a9#egg=Sloth-2.0.5",
        "git+https://github.com/uncharted-distil/distil-timeseries-loader@e218c02ca41deccab94da30bf492bcb74eeb6c78#egg=DistilTimeSeriesLoader"
    ],
    entry_points = {
        'd3m.primitives': [
            'clustering.k_means.Sloth = TimeSeriesD3MWrappers:Storc',
            'clustering.hdbscan.Hdbscan = TimeSeriesD3MWrappers:Hdbscan',
            'time_series_classification.shapelet_learning.Shallot = TimeSeriesD3MWrappers:Shallot',
            'time_series_classification.k_neighbors.Kanine = TimeSeriesD3MWrappers:Kanine',
            'time_series_forecasting.arima.Parrot = TimeSeriesD3MWrappers:Parrot',
        ],
    },
)
