#!/bin/bash -e

Paths=('d3m.primitives.clustering.hdbscan.Hdbscan' 'd3m.primitives.clustering.k_means.Sloth' 'd3m.primitives.time_series_classification.k_neighbors.Kanine' 'd3m.primitives.time_series_classification.shapelet_learning.Shallot' 'd3m.primitives.time_series_forecasting.arima.Parrot' 'd3m.primitives.time_series_forecasting.vector_autoregression.VAR')
cd /primitives/v2019.6.7/Distil
for i in "${Paths[@]}"; do
  cd $i
  echo $i
  cd *
  python3 -m d3m index describe -i 4 $i > primitive.json
done
