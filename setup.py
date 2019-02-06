from distutils.core import setup

setup(name='TimeSeries-D3M-Wrappers',
    version='1.0.0',
    description='Three wrappesr for interacting with New Knowledge time series tool Sloth',
    packages=['SlothD3MWrapper', 
             'ShallotD3MWrapper',
             'ParrotD3MWrapper'],
    install_requires=["Sloth==2.0.3"],
    dependency_links=[
        "git+https://github.com/NewKnowledge/sloth@82a1e08049531270256f38ca838e6cc7d1119223#egg=Sloth-2.0.3"
    ],
    entry_points = {
        'd3m.primitives': [
            'time_series_segmentation.cluster.Sloth = SlothD3MWrapper:Storc',
            'time_series_classification.shapelet_learning.Shallot = ShallotD3MWrapper:Shallot',
            'time_series_forecasting.arima.Parrot = ParrotD3MWrapper:Parrot'
        ],
    },
)
