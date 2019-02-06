from distutils.core import setup

setup(name='TimeSeriesD3MWrappers',
    version='1.0.0',
    description='Three wrappesr for interacting with New Knowledge time series tool Sloth',
    packages=['TimeSeriesD3MWrappers'],
    install_requires=["typing",
                      "Sloth==2.0.3"],
    dependency_links=[
        "git+https://github.com/NewKnowledge/sloth@82a1e08049531270256f38ca838e6cc7d1119223#egg=Sloth-2.0.3"
    ],
    entry_points = {
        'd3m.primitives': [
            'time_series_segmentation.cluster.Sloth = TimeSeriesD3MWrappers:Storc',
            'time_series_classification.shapelet_learning.Shallot = TimeSeriesD3MWrappers:Shallot',
            'time_series_forecasting.arima.Parrot = TimeSeriesD3MWrappers:Parrot'
        ],
    },
)
