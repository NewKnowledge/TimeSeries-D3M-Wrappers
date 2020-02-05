#!/bin/bash -e

Paths=('d3m.primitives.time_series_classification.k_neighbors.Kanine' 'd3m.primitives.time_series_forecasting.vector_autoregression.VAR' 'd3m.primitives.time_series_forecasting.lstm.DeepAR' 'd3m.primitives.time_series_classification.convolutional_neural_net.LSTM_FCN')
for i in "${Paths[@]}"; do
  cd /primitives/v2019.11.10/Distil
  cd $i
  echo $i
  cd *
  python3 -m d3m index describe -i 4 $i > primitive.json
done
