paths = ['clustering.hdbscan.Hdbscan', 'clustering.k_means.Sloth', 'time_series_classification.k_neighbors.Kanine', 'time_series_classification.shapelet_learning.Shallot', 'time_series_forecasting.arima.Parrot', 'time_series_forecasting.vector_autoregression.VAR']
paths = ['d3m.primitives.' + p for path in paths]

for path in paths; do
    echo $path