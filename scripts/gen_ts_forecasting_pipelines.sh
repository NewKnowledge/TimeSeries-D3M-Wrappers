#!/bin/bash -e 

cd /primitives/v2019.6.7/Distil/d3m.primitives.time_series_forecasting.arima.Parrot/1.0.3/pipelines
# generate and save pipeline + metafile
rm /primitives/v2019.6.7/Distil/d3m.primitives.time_series_forecasting.arima.Parrot/1.0.3/pipelines/*
python3 /src/timeseriesd3mwrappers/TimeSeriesD3MWrappers/pipelines/Parrot_56_sunspots_pipeline.py 56_sunspots
python3 /src/timeseriesd3mwrappers/TimeSeriesD3MWrappers/pipelines/Parrot_pop_spawn_original_pipeline.py LL1_736_population_spawn
python3 /src/timeseriesd3mwrappers/TimeSeriesD3MWrappers/pipelines/Parrot_pop_spawn_pipeline.py LL1_736_population_spawn_simpler
python3 /src/timeseriesd3mwrappers/TimeSeriesD3MWrappers/pipelines/Parrot_stock_market_pipeline.py LL1_736_stock_market

cd /primitives/v2019.6.7/Distil/d3m.primitives.time_series_forecasting.vector_autoregression.VAR/1.0.1/pipelines
# generate and save pipeline + metafile
rm /primitives/v2019.6.7/Distil/d3m.primitives.time_series_forecasting.vector_autoregression.VAR/1.0.1/pipelines
python3 /src/timeseriesd3mwrappers/TimeSeriesD3MWrappers/pipelines/VAR_56_sunspots_pipeline.py 56_sunspots
python3 /src/timeseriesd3mwrappers/TimeSeriesD3MWrappers/pipelines/VAR_pop_spawn_original_pipeline.py LL1_736_population_spawn
python3 /src/timeseriesd3mwrappers/TimeSeriesD3MWrappers/pipelines/VAR_pop_spawn_pipeline.py LL1_736_population_spawn_simpler
python3 /src/timeseriesd3mwrappers/TimeSeriesD3MWrappers/pipelines/VAR_stock_market_pipeline.py LL1_736_stock_market


