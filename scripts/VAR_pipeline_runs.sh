#!/bin/bash -e 

Datasets = ('56_sunspots' '56_sunspots_monthly' 'LL1_736_population_spawn' 'LL1_736_population_spawn_simpler' 'LL1_736_stock_market' 'LL1_terra_canopy_height_long_form_s4_100' 'LL1_terra_canopy_height_long_form_s4_90' 'LL1_terra_canopy_height_long_form_s4_80' 'LL1_terra_canopy_height_long_form_s4_70' 'LL1_terra_leaf_angle_mean_long_form_s4')
cd /primitives
# git pull upstream master
# git branch forecasting_pipelines
# git checkout forecasting_pipelines
cd /primitives/v2019.11.10/Distil/d3m.primitives.time_series_forecasting.vector_autoregression.VAR/1.0.2
mkdir pipelines
cd pipelines
python3 "/src/timeseriesd3mwrappers/TimeSeriesD3MWrappers/pipelines/forecasting_pipeline_var.py"
cd ..
mkdir pipeline_runs
cd pipeline_runs

python3 "/src/timeseriesd3mwrappers/TimeSeriesD3MWrappers/pipelines/forecasting_pipeline_var.py"

for i in "${Datasets[@]}"; do

  python3 -m d3m runtime -d /datasets/ fit-score -p ../pipelines/*.json -i /datasets/seed_datasets_current/$i/TRAIN/dataset_TRAIN/datasetDoc.json -t /datasets/seed_datasets_current/$i/TEST/dataset_TEST/datasetDoc.json -a /datasets/seed_datasets_current/$i/SCORE/dataset_SCORE/datasetDoc.json -r /datasets/seed_datasets_current/$i/${i}_problem/problemDoc.json -O $i.yml

done

# zip pipeline runs individually
cd ..
gzip -r pipeline_runs