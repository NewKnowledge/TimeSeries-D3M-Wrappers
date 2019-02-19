from distutils.core import setup

setup(name='TimeSeriesD3MWrappers',
    version='1.0.2',
    description='Five wrappers for interacting with New Knowledge time series tool Sloth',
    packages=['TimeSeriesD3MWrappers'],
    install_requires=["typing",
                      "Sloth==2.0.5"],
    dependency_links=[
        "git+https://github.com/NewKnowledge/sloth@eeca19594f2922487e21357fba6d15488331f5e8#egg=Sloth-2.0.5"
    ],
    entry_points = {
        'd3m.primitives': [
            'clustering.kmeans.Sloth = TimeSeriesD3MWrappers:Storc',
            'clustering.time_series_clustering.Hdbscan = TimeSeriesD3MWrappers:Hdbscan',
            'time_series_classification.shapelet_learning.Shallot = TimeSeriesD3MWrappers:Shallot',
            'time_series_classification.k_neighbors.Kanine = TimeSeriesD3MWrappers:Kanine',
            'time_series_forecasting.arima.Parrot = TimeSeriesD3MWrappers:Parrot',
        ],
    },
)
